"""Generic numeric / camera helpers used across the package."""

import math
import os
import random

import numpy as np
import torch
from packaging import version as pver


# ---------------------------------------------------------------------------
# Tiny tensor helpers
# ---------------------------------------------------------------------------

def custom_meshgrid(*args):
    """Backwards-compatible meshgrid that always uses 'ij' indexing."""
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    return torch.meshgrid(*args, indexing='ij')


def safe_normalize(x, eps: float = 1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ---------------------------------------------------------------------------
# Memory probes (used by the trainer for logging).
# ---------------------------------------------------------------------------

def get_cpu_mem() -> float:
    """Return resident-set memory of the current process in GiB."""
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3


def get_gpu_mem():
    """Return (total, per-device) used GPU memory in GiB."""
    num = torch.cuda.device_count()
    total, per_device = 0, []
    for i in range(num):
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        used = int(((mem_total - mem_free) / 1024 ** 3) * 1000) / 1000
        per_device.append(used)
        total += used
    return total, per_device


# ---------------------------------------------------------------------------
# Ray sampling
# ---------------------------------------------------------------------------

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    """Generate rays for a batch of camera poses.

    Args:
        poses: ``[B, 4, 4]`` cam2world matrices.
        intrinsics: ``(fx, fy, cx, cy)``.
        H, W: image height/width.
        N: number of rays per image (``-1`` means dense).
        error_map: per-pixel sample probability (optional).

    Returns:
        Dict with keys ``rays_o``, ``rays_d`` (both ``[B, N, 3]``) and ``inds``.
    """
    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}
    if N > 0:
        N = min(N, H * W)
        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)
            inds = inds.expand([B, N])
        else:
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False)
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y
            results['inds_coarse'] = inds_coarse
        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)
        results['inds'] = inds
    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = -torch.ones_like(i)
    xs = -(i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)

    rays_o = poses[..., :3, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    return results


# ---------------------------------------------------------------------------
# Perp-Neg helpers
# ---------------------------------------------------------------------------

def adjust_text_embeddings(embeddings, azimuth, opt):
    """Build a Perp-Neg input batch from per-direction text embeddings."""
    text_z_list = []
    weights_list = []
    K = 0
    for b in range(azimuth.shape[0]):
        text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth[b], opt)
        K = max(K, weights_.shape[0])
        text_z_list.append(text_z_)
        weights_list.append(weights_)

    # Interleave embeddings/weights from different views.
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0)

    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0)
    return text_embeddings, weights


def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    """Compute the (positive, side-neg, front-neg) embedding triple for one view."""
    if -90 <= azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front0']
        end_z = embeddings['side0']
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front0'], embeddings['side0']], dim=0)
        front_neg_w = 0.0 if r > 0.8 else math.exp(-r * opt.front_decay_factor) * opt.negative_w
        side_neg_w = 0.0 if r < 0.2 else math.exp(-(1 - r) * opt.side_decay_factor) * opt.negative_w
        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side0']
        end_z = embeddings['back0']
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side0'], embeddings['front0']], dim=0)
        front_neg_w = opt.negative_w
        side_neg_w = (
            0.0 if r > 0.8 else math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2
        )
        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)
