"""Multi-Opacity volumetric and DMTet rendering.

The volumetric ``run`` implements Eq. (4)-(6) and Eq. (8) of the paper:

  alpha_i^j  = (1 - exp(-tau_i^j * delta_i)) * prod_{k<j} exp(-tau_i^k * delta_i)
  alpha_i^0  = 1 - exp(-(sum_j tau_i^j) * delta_i)

with the random material-ordering permutation (Eq. (13)) used for gradient
balancing across subspaces.
"""

import math
import os

import cv2
import mcubes
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from mesh_utils import clean_mesh, decimate_mesh

from .utils import custom_meshgrid, safe_normalize


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def sample_pdf(bins, weights, n_samples, det=False):
    """Inverse-CDF sampling along rays (vanilla NeRF hierarchical sampling)."""
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = torch.linspace(0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples,
                           device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=weights.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='cube', min_near=0.05):
    """Compute (near, far) along each ray for a sphere/cube scene bound."""
    radius = rays_o.norm(dim=-1, keepdim=True)
    if type == 'sphere':
        near = radius - bound
        far = radius + bound
    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15)
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        near = torch.clamp(near, min=min_near)
    else:
        raise ValueError(f"unknown bound type: {type}")
    return near, far


# ---------------------------------------------------------------------------
# DMTet utilities
# ---------------------------------------------------------------------------

class DMTet:
    """Marching tetrahedra extraction (from NVlabs nvdiffrec)."""

    def __init__(self, device):
        self.device = device
        self.triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [1,  0,  2, -1, -1, -1],
            [4,  0,  3, -1, -1, -1],
            [1,  4,  2,  1,  3,  4],
            [3,  1,  5, -1, -1, -1],
            [2,  3,  0,  2,  5,  3],
            [1,  4,  0,  1,  5,  4],
            [4,  2,  5, -1, -1, -1],
            [4,  5,  2, -1, -1, -1],
            [4,  1,  0,  4,  5,  1],
            [3,  2,  0,  3,  5,  2],
            [1,  3,  5, -1, -1, -1],
            [4,  1,  2,  4,  3,  1],
            [3,  0,  4, -1, -1, -1],
            [2,  0,  1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=torch.long, device=device)
        self.num_triangles_table = torch.tensor(
            [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0],
            dtype=torch.long, device=device,
        )
        self.base_tet_edges = torch.tensor(
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
            dtype=torch.long, device=device,
        )

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)
            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
        return torch.stack([a, b], -1)

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones(unique_edges.shape[0], dtype=torch.long,
                                 device=pos_nx3.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,
                                               device=pos_nx3.device)
            idx_map = mapping[idx_map]

            interp_v = unique_edges[mask_edges]

        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)
        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=pos_nx3.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        faces = torch.cat([
            torch.gather(input=idx_map[num_triangles == 1],
                        dim=1,
                        index=self.triangle_table[tetindex[num_triangles == 1]][:, :3])
            .reshape(-1, 3),
            torch.gather(input=idx_map[num_triangles == 2],
                        dim=1,
                        index=self.triangle_table[tetindex[num_triangles == 2]][:, :6])
            .reshape(-1, 3),
        ], dim=0)
        return verts, faces


def compute_edge_to_face_mapping(attr_idx):
    with torch.no_grad():
        edge_indices = torch.tensor(
            [[0, 1], [1, 2], [2, 0]], dtype=torch.long, device=attr_idx.device,
        )
        all_edges = attr_idx[:, edge_indices].reshape(-1, 2)
        all_edges_sorted = torch.sort(all_edges, dim=-1)[0]

        unique_edges, idx_map = torch.unique(all_edges_sorted, dim=0, return_inverse=True)

        tris_per_edge = torch.zeros((unique_edges.shape[0]), dtype=torch.long,
                                    device=attr_idx.device)
        edge_to_face = torch.zeros((unique_edges.shape[0], 2), dtype=torch.long,
                                   device=attr_idx.device)

        for i in range(idx_map.shape[0]):
            face_idx = i // 3
            edge_idx = idx_map[i]
            edge_to_face[edge_idx, tris_per_edge[edge_idx]] = face_idx
            tris_per_edge[edge_idx] += 1

        return edge_to_face[tris_per_edge == 2]


def normal_consistency(face_normals, t_pos_idx):
    """Smoothness loss between face-normals of edge-adjacent triangles."""
    edge_to_face = compute_edge_to_face_mapping(t_pos_idx)
    face_pairs = face_normals[edge_to_face]
    cos = torch.sum(face_pairs[:, 0] * face_pairs[:, 1], -1)
    return ((1 - cos) ** 2).mean()


def laplacian_uniform(verts, faces):
    V, F_ = verts.shape[0], faces.shape[0]

    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    diag_idx = adj[0]
    diag = torch.stack((diag_idx, diag_idx), dim=0)

    idx = torch.cat((adj, diag), dim=1)
    values = torch.cat((-adj_values, adj_values))

    return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()


def laplacian_smooth_loss(verts, faces):
    L = laplacian_uniform(verts.detach(), faces.long())
    loss = L.mm(verts)
    return loss.norm(dim=1).mean()


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class NeRFRenderer(nn.Module):
    """Volumetric + DMTet renderer with multi-density bookkeeping."""

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.bound = opt.bound
        self.cascade = 1 + math.ceil(math.log2(opt.bound))
        self.grid_size = 128
        self.max_level = None
        self.dmtet = opt.dmtet
        self.cuda_ray = opt.cuda_ray
        self.taichi_ray = opt.taichi_ray
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        # Axis-aligned bounding box used for clamping query positions.
        aabb_train = torch.FloatTensor(
            [-opt.bound, -opt.bound, -opt.bound, opt.bound, opt.bound, opt.bound]
        )
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_train.clone())

        self.glctx = None

        # CUDA raymarching extra buffers.
        if self.cuda_ray:
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3])
            density_bitfield = torch.zeros(
                self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8,
            )
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0

        # DMTet refinement: one global instance plus one per subspace.
        if self.dmtet:
            tet_path = os.path.join('tets', f'{self.opt.tet_grid_size}_tets.npz')
            tets = np.load(tet_path)

            self.verts = -torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * 2
            self.indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
            self.tet_scale = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')
            self.dmtet_model = DMTet('cuda')

            self.dmtet_model_parts = []
            self.verts_parts = []
            self.indices_parts = []
            self.tet_scale_parts = []
            for _ in range(self.opt.part_nums):
                tets_p = np.load(tet_path)
                self.verts_parts.append(
                    -torch.tensor(tets_p['vertices'], dtype=torch.float32, device='cuda') * 2
                )
                self.indices_parts.append(
                    torch.tensor(tets_p['indices'], dtype=torch.long, device='cuda')
                )
                self.tet_scale_parts.append(
                    torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')
                )
                self.dmtet_model_parts.append(DMTet('cuda'))

            self.register_parameter(
                'sdf', nn.Parameter(torch.zeros_like(self.verts[..., 0]), requires_grad=True)
            )
            self.register_parameter(
                'deform', nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
            )

            for part_idx in range(self.opt.part_nums):
                self.register_parameter(
                    f'sdf_{part_idx}',
                    nn.Parameter(torch.zeros_like(self.verts_parts[part_idx][..., 0]),
                                 requires_grad=True),
                )
                self.register_parameter(
                    f'deform_{part_idx}',
                    nn.Parameter(torch.zeros_like(self.verts_parts[part_idx]),
                                 requires_grad=True),
                )

            edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                                 dtype=torch.long, device='cuda')
            all_edges = self.indices[:, edges].reshape(-1, 2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

            if self.opt.h <= 2048 and self.opt.w <= 2048:
                self.glctx = dr.RasterizeCudaContext()
            else:
                self.glctx = dr.RasterizeGLContext()

    # ------------------------------------------------------------------
    # Density blob initialization (Spatial Density Bias from DreamFusion).
    # ------------------------------------------------------------------

    def density_blob(self, x, index=None):
        """Gaussian/linear density bump centered at the origin (or part center)."""
        d = (x ** 2).sum(-1)

        if index is not None:
            radius = self.opt.parts_blob_radius[index]
        else:
            radius = self.opt.blob_radius

        if self.opt.density_activation == 'exp':
            return self.opt.blob_density * torch.exp(-d / (2 * radius ** 2))
        return self.opt.blob_density * (1 - torch.sqrt(d) / radius)

    # ------------------------------------------------------------------
    # Required overrides (subclass implements these).
    # ------------------------------------------------------------------

    def forward(self, x, d):
        raise NotImplementedError

    def density(self, x):
        raise NotImplementedError

    def reset_extra_state(self):
        if not (self.cuda_ray or self.taichi_ray):
            return
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0

    # ------------------------------------------------------------------
    # Mesh export (marching cubes / DMTet).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def export_mesh(self, path, resolution=None, decimate_target=-1, S=128):
        if self.opt.dmtet:
            sdf = self.sdf
            deform = torch.tanh(self.deform) / self.opt.tet_grid_size
            vertices, triangles = self.dmtet_model(self.verts + deform, sdf, self.indices)
            vertices = vertices.detach().cpu().numpy()
            triangles = triangles.detach().cpu().numpy()
        else:
            if resolution is None:
                resolution = self.grid_size
            if self.cuda_ray:
                density_thresh = (
                    min(self.mean_density, self.density_thresh)
                    if np.greater(self.mean_density, 0) else self.density_thresh
                )
            else:
                density_thresh = self.density_thresh
            if self.opt.density_activation == 'softplus':
                density_thresh = density_thresh * 25

            sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)
            X = torch.linspace(-1, 1, resolution).split(S)
            Y = torch.linspace(-1, 1, resolution).split(S)
            Z = torch.linspace(-1, 1, resolution).split(S)
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1),
                                         zz.reshape(-1, 1)], dim=-1)
                        val = self.density(pts.to(self.aabb_train.device))
                        sigmas[
                            xi * S: xi * S + len(xs),
                            yi * S: yi * S + len(ys),
                            zi * S: zi * S + len(zs),
                        ] = val['sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()

            print(f'[INFO] marching cubes thresh: {density_thresh} '
                  f'({sigmas.min()} ~ {sigmas.max()})')
            vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
            vertices = vertices / (resolution - 1.0) * 2 - 1

        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.01)

        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        os.makedirs(path, exist_ok=True)
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        mesh.export(os.path.join(path, 'mesh.obj'))

    # ------------------------------------------------------------------
    # Volumetric rendering (training path).
    # ------------------------------------------------------------------

    def run(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0,
            shading='albedo', bg_color=None, perturb=False, **kwargs):
        """Volumetric rendering with multi-density alpha compositing.

        Returns a dict with the global image / depth / weights as well as one
        ``image_{j}``, ``weights_{j}``, ``weights_sum_{j}`` for each subspace.
        """

        part_nums = self.opt.part_nums
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_o.shape[0]
        device = rays_o.device

        results = {}
        aabb = self.aabb_train if self.training else self.aabb_infer

        nears, fars = near_far_from_bound(
            rays_o, rays_d, self.bound, type='sphere', min_near=self.min_near,
        )

        if light_d is None:
            light_d = safe_normalize(rays_o + torch.randn(3, device=device))

        # Coarse uniform sampling between near/far.
        z_vals = torch.linspace(0.0, 1.0, self.opt.num_steps, device=device).unsqueeze(0)
        z_vals = z_vals.expand((N, self.opt.num_steps))
        z_vals = nears + (fars - nears) * z_vals

        sample_dist = (fars - nears) / self.opt.num_steps
        if perturb:
            z_vals = z_vals + (torch.rand_like(z_vals) - 0.5) * sample_dist

        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])

        density_outputs = self.density(xyzs.reshape(-1, 3))
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, self.opt.num_steps, -1)

        # Hierarchical (fine) sampling using global visibility weights.
        if self.opt.upsample_steps > 0:
            with torch.no_grad():
                deltas = z_vals[..., 1:] - z_vals[..., :-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])],
                                   dim=-1)
                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1))
                alphas_shifted = torch.cat(
                    [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1,
                )
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]

                z_vals_mid = z_vals[..., :-1] + 0.5 * deltas[..., :-1]
                new_z_vals = sample_pdf(
                    z_vals_mid, weights[:, 1:-1],
                    self.opt.upsample_steps, det=not self.training,
                ).detach()
                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1)
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:])

            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, self.opt.upsample_steps, -1)

            z_vals = torch.cat([z_vals, new_z_vals], dim=1)
            z_vals, z_index = torch.sort(z_vals, dim=1)
            xyzs = torch.cat([xyzs, new_xyzs], dim=1)
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))
            for k in density_outputs:
                merged = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(
                    merged, dim=1, index=z_index.unsqueeze(-1).expand_as(merged),
                )

        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        light_d = light_d.view(-1, 1, 3).expand_as(xyzs)
        dirs = safe_normalize(dirs)

        ret = self(
            xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d.reshape(-1, 3),
            ratio=ambient_ratio, shading=shading,
        )
        rgbs, normals = ret['rgbs'], ret['normals']

        # ------------------------------------------------------------------
        # Multi-density alpha compositing (Eq. 4-6, Eq. 13).
        # ------------------------------------------------------------------
        sigma_parts = density_outputs['sigma_parts']  # [N, S, M]

        # Random material-ordering permutation for gradient equilibrium.
        perm = torch.randperm(part_nums)
        inverse_perm = torch.argsort(perm)
        sigma_parts_shuffled = sigma_parts[..., perm]

        alphas_parts = []
        for j in range(part_nums):
            term = 1 - torch.exp(-deltas * sigma_parts_shuffled[..., j])
            for k in range(j):
                term = term * torch.exp(-deltas * sigma_parts_shuffled[..., k])
            alphas_parts.append(term)
        alphas_parts = torch.stack(alphas_parts)[inverse_perm]  # [M, N, S]

        # Append the global alpha at the end so weights[-1] == global weights.
        global_alpha = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1))
        alphas_parts = torch.cat([alphas_parts, global_alpha[None, ...]], dim=0)

        alphas_shifted = torch.cat(
            [torch.ones_like(alphas_parts[..., :1]), 1 - alphas_parts + 1e-5], dim=-1,
        )
        weights = alphas_parts * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
        weights_sum = weights.sum(dim=-1).clamp(0, 1)
        depth = torch.sum(weights[-1] * z_vals, dim=-1)

        rgbs = rgbs.view(N, -1, 3)
        if normals is not None:
            normals = normals.view(N, -1, 3)

        if bg_color is None:
            bg_color = self.background(rays_d) if self.opt.bg_radius > 0 else 1

        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(1 + part_nums, prefix[-1], 3)
        weights_sum = weights_sum.view(1 + part_nums, prefix[-1])

        depth = depth.view(*prefix)

        if self.training:
            if self.opt.lambda_orient > 0 and normals is not None:
                loss_orient = weights[-1].detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.sum(-1).mean()

            if self.opt.lambda_3d_normal_smooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()

            if (self.opt.lambda_2d_normal_smooth > 0 or self.opt.lambda_normal > 0) \
                    and normals is not None:
                normal_image = torch.sum(
                    weights[-1].unsqueeze(-1) * (normals + 1) / 2, dim=-2,
                )
                results['normal_image'] = normal_image

        results['image'] = image[-1]
        results['depth'] = depth
        results['weights'] = weights[-1]
        results['weights_sum'] = weights_sum[-1]
        for j in range(part_nums):
            results[f'image_{j}'] = image[j]
            results[f'weights_{j}'] = weights[j]
            results[f'weights_sum_{j}'] = weights_sum[j]
        return results

    # ------------------------------------------------------------------
    # DMTet path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_tet(self, mesh=None):
        """Initialize global + per-subspace SDFs from the current density field."""
        if mesh is not None:
            scale = 0.8 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
            center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
            mesh.vertices = (mesh.vertices - center) * scale
            self.tet_scale = torch.from_numpy(
                np.array([np.abs(mesh.vertices).max()]) + 1e-1,
            ).to(self.verts.dtype).cuda()
            self.verts = self.verts * self.tet_scale
        else:
            density_thresh = (
                min(self.mean_density, self.density_thresh) if self.cuda_ray
                else self.density_thresh
            )
            if self.opt.density_activation == 'softplus':
                density_thresh = density_thresh * 25

            sigma = self.density(self.verts)['sigma']
            mask = sigma > density_thresh
            valid_verts = self.verts[mask]
            self.tet_scale = valid_verts.abs().amax(dim=0) + 1e-1
            self.verts = self.verts * self.tet_scale

            sigma = self.density(self.verts)['sigma']
            self.sdf.data += (sigma - density_thresh).clamp(-1, 1)

            for part_idx in range(self.opt.part_nums):
                ret = self.density(self.verts_parts[part_idx])
                sigma = ret['sigma_parts'][:, part_idx]

                mask = sigma > density_thresh
                valid_verts = self.verts_parts[part_idx][mask]
                self.tet_scale_parts[part_idx] = valid_verts.abs().amax(dim=0) + 1e-1
                self.verts_parts[part_idx] = self.verts_parts[part_idx] * self.tet_scale_parts[part_idx]

                ret = self.density(self.verts_parts[part_idx])
                sigma = ret['sigma_parts'][:, part_idx]

                sdf_p = getattr(self, f'sdf_{part_idx}')
                sdf_p.data += (sigma - density_thresh).clamp(-1, 1)

        print(f'[INFO] init dmtet: scale = {self.tet_scale}')

    def _render_dmtet_part(self, mvp, h, w, light_d, ambient_ratio, shading,
                          bg_color, sdf, deform, verts_buf, indices_buf, dmtet_module):
        """Rasterize a single DMTet instance (global or one subspace)."""
        deform = torch.tanh(deform) / self.opt.tet_grid_size
        verts, faces = dmtet_module(verts_buf + deform, sdf, indices_buf)

        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        faces = faces.int()

        face_normals = safe_normalize(torch.cross(v1 - v0, v2 - v0))
        vn = torch.zeros_like(verts)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)
        vn = torch.where(
            torch.sum(vn * vn, -1, keepdim=True) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )

        verts_clip = torch.bmm(
            F.pad(verts, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).repeat(
                mvp.shape[0], 1, 1),
            mvp.permute(0, 2, 1),
        ).float()

        rast, _ = dr.rasterize(self.glctx, verts_clip, faces, (h, w))
        alpha = (rast[..., 3:] > 0).float()
        xyzs, _ = dr.interpolate(verts.unsqueeze(0), rast, faces)
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
        normal = safe_normalize(normal)

        xyzs = xyzs.view(-1, 3)
        mask = (rast[..., 3:] > 0).view(-1).detach()

        albedo = torch.zeros_like(xyzs, dtype=torch.float32)
        if mask.any():
            albedo[mask] = self.density(xyzs[mask])['albedo'].float()
        albedo = albedo.view(-1, h, w, 3)

        if self.opt.lock_geo and shading in ['textureless', 'normal']:
            shading = 'lambertian'

        if shading == 'albedo':
            color = albedo
        elif shading == 'textureless':
            lambertian = ambient_ratio + (1 - ambient_ratio) * (
                normal * light_d).sum(-1).float().clamp(min=0)
            color = lambertian.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif shading == 'normal':
            color = (normal + 1) / 2
        else:  # lambertian
            lambertian = ambient_ratio + (1 - ambient_ratio) * (
                normal * light_d).sum(-1).float().clamp(min=0)
            color = albedo * lambertian.unsqueeze(-1)

        color = dr.antialias(color, rast, verts_clip, faces).clamp(0, 1)
        alpha = dr.antialias(alpha, rast, verts_clip, faces).clamp(0, 1)

        if torch.is_tensor(bg_color) and len(bg_color.shape) > 1:
            bg_color = bg_color.view(-1, h, w, 3)
        depth = rast[:, :, :, [2]]
        color = color + (1 - alpha) * bg_color
        return color, depth, alpha, verts, faces, face_normals, normal, rast, verts_clip

    def run_dmtet(self, rays_o, rays_d, mvp, h, w, light_d=None,
                  ambient_ratio=1.0, shading='albedo', bg_color=None, **kwargs):
        """Mesh-based rendering during the DMTet refinement stage."""
        device = mvp.device
        campos = rays_o[:, 0, :]

        if light_d is None:
            light_d = safe_normalize(campos + torch.randn_like(campos)).view(-1, 1, 1, 3)

        if bg_color is None:
            bg_color = self.background(rays_d) if self.opt.bg_radius > 0 else 1

        results = {}

        # Global mesh.
        color, depth, alpha, verts, faces, face_normals, normal, rast, verts_clip = (
            self._render_dmtet_part(
                mvp, h, w, light_d, ambient_ratio, shading, bg_color,
                self.sdf, self.deform, self.verts, self.indices, self.dmtet_model,
            )
        )
        results['image'] = color
        results['depth'] = depth
        results['weights_sum'] = alpha.squeeze(-1)

        # Per-subspace meshes.
        face_normals_parts, faces_parts, verts_parts = [], [], []
        for part_idx in range(self.opt.part_nums):
            color_p, depth_p, alpha_p, verts_p, faces_p, fn_p, _, _, _ = (
                self._render_dmtet_part(
                    mvp, h, w, light_d, ambient_ratio, shading, bg_color,
                    getattr(self, f'sdf_{part_idx}'),
                    getattr(self, f'deform_{part_idx}'),
                    self.verts_parts[part_idx],
                    self.indices_parts[part_idx],
                    self.dmtet_model_parts[part_idx],
                )
            )
            results[f'image_{part_idx}'] = color_p
            results[f'depth_{part_idx}'] = depth_p
            results[f'weights_sum_{part_idx}'] = alpha_p.squeeze(-1)
            face_normals_parts.append(fn_p)
            faces_parts.append(faces_p)
            verts_parts.append(verts_p)

        if self.opt.lambda_2d_normal_smooth > 0 or self.opt.lambda_normal > 0:
            normal_image = dr.antialias((normal + 1) / 2, rast, verts_clip, faces).clamp(0, 1)
            results['normal_image'] = normal_image

        if self.training:
            if self.opt.lambda_mesh_normal > 0:
                loss = normal_consistency(face_normals, faces)
                for part_idx in range(self.opt.part_nums):
                    loss = loss + normal_consistency(
                        face_normals_parts[part_idx], faces_parts[part_idx],
                    )
                results['normal_loss'] = loss

            if self.opt.lambda_mesh_laplacian > 0:
                loss = laplacian_smooth_loss(verts, faces)
                for part_idx in range(self.opt.part_nums):
                    loss = loss + laplacian_smooth_loss(
                        verts_parts[part_idx], faces_parts[part_idx],
                    )
                results['lap_loss'] = loss

        return results

    # ------------------------------------------------------------------
    # CUDA-ray density grid update (only used when ``--cuda_ray``).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        if not (self.cuda_ray or self.taichi_ray):
            return
        import raymarching

        tmp_grid = -torch.ones_like(self.density_grid)
        X = torch.arange(self.grid_size, dtype=torch.int32,
                         device=self.aabb_train.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32,
                         device=self.aabb_train.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32,
                         device=self.aabb_train.device).split(S)
        for xs in X:
            for ys in Y:
                for zs in Z:
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1),
                                        zz.reshape(-1, 1)], dim=-1)
                    indices = raymarching.morton3D(coords).long()
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1

                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid = bound / self.grid_size
                        cas_xyzs = xyzs * (bound - half_grid)
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                        tmp_grid[cas, indices] = sigmas

        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(
            self.density_grid[valid_mask] * decay, tmp_grid[valid_mask],
        )
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        density_thresh = min(self.mean_density, self.density_thresh)
        if self.cuda_ray:
            self.density_bitfield = raymarching.packbits(
                self.density_grid, density_thresh, self.density_bitfield,
            )

    # ------------------------------------------------------------------
    # Entry point used by the trainer.
    # ------------------------------------------------------------------

    def render(self, rays_o, rays_d, mvp, h, w, staged=False, max_ray_batch=4096, **kwargs):
        B, N = rays_o.shape[:2]
        device = rays_o.device

        if self.dmtet:
            return self.run_dmtet(rays_o, rays_d, mvp, h, w, **kwargs)

        if not staged:
            return self.run(rays_o, rays_d, **kwargs)

        # Chunked inference for large images during evaluation/test.
        depth = torch.empty((B, N), device=device)
        image = torch.empty((B, N, 3), device=device)
        weights_sum = torch.empty((B, N), device=device)
        image_parts = [torch.empty((B, N, 3), device=device)
                       for _ in range(self.opt.part_nums)]

        for b in range(B):
            head = 0
            while head < N:
                tail = min(head + max_ray_batch, N)
                results_ = self.run(
                    rays_o[b:b + 1, head:tail], rays_d[b:b + 1, head:tail], **kwargs,
                )
                depth[b:b + 1, head:tail] = results_['depth']
                weights_sum[b:b + 1, head:tail] = results_['weights_sum']
                image[b:b + 1, head:tail] = results_['image']
                for j in range(self.opt.part_nums):
                    image_parts[j][b:b + 1, head:tail] = results_[f'image_{j}']
                head += max_ray_batch

        results = {'depth': depth, 'image': image, 'weights_sum': weights_sum}
        for j in range(self.opt.part_nums):
            results[f'image_{j}'] = image_parts[j]
        return results
