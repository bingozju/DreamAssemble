"""Multi-Density Neural Field backbone for DreamAssemble.

The forward pass produces ``M`` per-subspace densities together with a single
shared albedo, implementing Eq. (3) of the paper:

    (tau^1, ..., tau^M, rho) = MLP(mu; theta)

Compared to a vanilla NeRF, the only architectural change is enlarging the
final linear layer from ``3 + 1`` to ``3 + M`` outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from activation import biased_softplus, trunc_exp
from encoding import get_encoder

from .renderer import NeRFRenderer
from .utils import safe_normalize


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Plain feed-forward MLP with ReLU activations between hidden layers."""

    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.num_layers = num_layers

        layers = []
        for layer_idx in range(num_layers):
            in_dim = dim_in if layer_idx == 0 else dim_hidden
            out_dim = dim_out if layer_idx == num_layers - 1 else dim_hidden
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        for layer_idx, layer in enumerate(self.net):
            x = layer(x)
            if layer_idx != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


def _xavier_init(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight.data)
        init.constant_(module.bias.data, 0)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class NeRFNetwork(NeRFRenderer):
    """Multi-density grid-based NeRF.

    Args:
        opt: parsed argparse / OmegaConf namespace.
        num_layers: number of layers in the spatial MLP.
        hidden_dim: width of the spatial MLP.
        num_layers_bg: number of layers in the (optional) background MLP.
        hidden_dim_bg: width of the background MLP.
    """

    def __init__(self, opt,
                 num_layers: int = 3,
                 hidden_dim: int = 64,
                 num_layers_bg: int = 2,
                 hidden_dim_bg: int = 32):
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.part_nums = opt.part_nums

        # Per-subspace density centers (mu^j_center). Learned for flexibility.
        self.x_centers = nn.Parameter(torch.tensor(opt.part_centers, dtype=torch.float32),
                                      requires_grad=True)

        # Hash-grid spatial encoder (Instant-NGP style).
        self.encoder, self.in_dim = get_encoder(
            'hashgrid', input_dim=3, level_dim=4, log2_hashmap_size=19,
            desired_resolution=2048 * self.bound, interpolation='smoothstep',
        )

        # MLP outputs (rho_RGB, tau^1, ..., tau^M).
        self.sigma_net = MLP(
            self.in_dim, 3 + self.part_nums,
            hidden_dim + self.part_nums, num_layers, bias=True,
        )

        self.density_activation = (
            trunc_exp if self.opt.density_activation == 'exp' else biased_softplus
        )

        # Optional background network (frequency-encoded directional MLP).
        if self.opt.bg_radius > 0:
            self.encoder_bg, self.in_dim_bg = get_encoder(
                'frequency', input_dim=3, multires=6,
            )
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
        else:
            self.bg_net = None

        self.sigma_net.apply(_xavier_init)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def common_forward(self, x):
        """Run the spatial MLP and split into (sigma, albedo, sigma_parts).

        ``sigma_parts`` has shape ``[N, M]`` and stores the per-subspace
        densities tau^j. ``sigma`` is their (normalized) sum and corresponds
        to tau^0 in the paper.
        """

        enc = self.encoder(x, bound=self.bound, max_level=self.max_level)
        h = self.sigma_net(enc)

        # Per-subspace density blob initialization (Spatial Density Bias).
        density_blobs_parts = []
        for part_idx in range(len(self.x_centers)):
            offset = x - self.x_centers[part_idx]
            density_blobs_parts.append(self.density_blob(offset, part_idx)[..., None])
        density_blobs_parts = torch.cat(density_blobs_parts, dim=-1)  # [N, M]

        albedo = torch.sigmoid(h[..., :3])
        sigma_parts = self.density_activation(h[..., 3:] + density_blobs_parts)
        sigma_parts = sigma_parts / self.opt.part_nums  # normalize for stable summation
        sigma = torch.sum(sigma_parts, dim=-1)

        return sigma, albedo, sigma_parts

    def finite_difference_normal(self, x, epsilon=1e-2):
        """Approximate surface normal via central finite differences."""
        eps_x = torch.tensor([[epsilon, 0., 0.]], device=x.device)
        eps_y = torch.tensor([[0., epsilon, 0.]], device=x.device)
        eps_z = torch.tensor([[0., 0., epsilon]], device=x.device)

        def _sigma(p):
            return self.common_forward(p.clamp(-self.bound, self.bound))[0]

        dx = 0.5 * (_sigma(x + eps_x) - _sigma(x - eps_x)) / epsilon
        dy = 0.5 * (_sigma(x + eps_y) - _sigma(x - eps_y)) / epsilon
        dz = 0.5 * (_sigma(x + eps_z) - _sigma(x - eps_z)) / epsilon
        return -torch.stack([dx, dy, dz], dim=-1)

    def normal(self, x):
        n = self.finite_difference_normal(x)
        n = safe_normalize(n)
        return torch.nan_to_num(n)

    # ------------------------------------------------------------------
    # Standard NeRF API
    # ------------------------------------------------------------------

    def forward(self, x, d, l=None, ratio=1.0, shading='albedo'):
        """Compute (sigma, color) for a batch of points.

        Args:
            x: ``[N, 3]`` query positions in ``[-bound, bound]``.
            d: ``[N, 3]`` view directions in ``[-1, 1]``.
            l: ``[3]`` plane-light direction (used by Lambertian shading).
            ratio: ambient ratio for Lambertian shading.
            shading: one of ``albedo``, ``lambertian``, ``textureless``, ``normal``.
        """
        sigma, albedo, sigma_parts = self.common_forward(x)

        if shading == 'albedo':
            normal = None
            color = albedo
        else:
            normal = self.normal(x)
            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0)
            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else:  # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)

        return {
            'sigmas': sigma,
            'rgbs': color,
            'normals': normal,
            'sigma_parts': sigma_parts,
        }

    def density(self, x):
        sigma, albedo, sigma_parts = self.common_forward(x)
        return {'sigma': sigma, 'albedo': albedo, 'sigma_parts': sigma_parts}

    def background(self, d):
        h = self.encoder_bg(d)
        return torch.sigmoid(self.bg_net(h))

    # ------------------------------------------------------------------
    # Optimizer parameter groups
    # ------------------------------------------------------------------

    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]

        if self.opt.bg_radius > 0:
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        if self.opt.dmtet and not self.opt.lock_geo:
            params.append({'params': self.sdf, 'lr': lr * 5})
            params.append({'params': self.deform, 'lr': lr * 5})
            for part_idx in range(self.opt.part_nums):
                params.append({'params': getattr(self, f'sdf_{part_idx}'), 'lr': lr * 5})
                params.append({'params': getattr(self, f'deform_{part_idx}'), 'lr': lr * 5})

        return params
