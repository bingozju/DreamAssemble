"""NeRF camera/ray dataset.

Generates random training cameras on a sphere around the origin, and fixed
turntable cameras for validation/test.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import get_rays, safe_normalize


# ---------------------------------------------------------------------------
# View direction labelling
# ---------------------------------------------------------------------------

def get_view_direction(thetas, phis, overhead, front):
    """Map (theta, phi) to one of {front, side-L, back, side-R, top, bottom}.

    Args:
        phis: azimuth in radians, range [0, 2*pi).
        thetas: polar angle in radians, range [0, pi].
    """
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0  # front
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1       # side-L
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2  # back
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3  # side-R
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


# ---------------------------------------------------------------------------
# Camera sampling
# ---------------------------------------------------------------------------

def rand_poses(size, device, opt,
               radius_range=(1.0, 1.5),
               theta_range=(0, 120),
               phi_range=(0, 360),
               return_dirs=False,
               angle_overhead=30,
               angle_front=60,
               uniform_sphere_rate=0.5):
    """Sample ``size`` random camera poses on an orbit around the origin."""
    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size, device=device),
                torch.abs(torch.randn(size, device=device)),
                torch.randn(size, device=device),
            ], dim=-1),
            p=2, dim=1,
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        phis[phis < 0] += 2 * np.pi
        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1)

    targets = torch.zeros_like(centers)

    if opt.jitter_pose:
        centers += torch.rand_like(centers) * opt.jitter_center - opt.jitter_center / 2.0
        targets += torch.randn_like(centers) * opt.jitter_target

    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    up_noise = torch.randn_like(up_vector) * opt.jitter_up if opt.jitter_pose else 0
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front) if return_dirs else None

    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180
    return poses, dirs, thetas, phis, radius


def circle_poses(device,
                 radius=torch.tensor([3.2]),
                 theta=torch.tensor([60.0]),
                 phi=torch.tensor([0.0]),
                 return_dirs=False,
                 angle_overhead=30,
                 angle_front=60):
    """Build look-at camera poses from spherical-coordinate inputs."""
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim=-1)

    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    dirs = get_view_direction(theta, phi, angle_overhead, angle_front) if return_dirs else None
    return poses, dirs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NeRFDataset:
    """Dataset wrapping random / circle / six-view sampling."""

    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        self.opt = opt
        self.device = device
        self.type = type
        self.H = H
        self.W = W
        self.size = size

        self.training = self.type in ['train', 'all']
        self.cx = self.H / 2
        self.cy = self.W / 2
        self.near = self.opt.min_near
        self.far = 1000

    def get_default_view_data(self):
        H = int(self.opt.known_view_scale * self.H)
        W = int(self.opt.known_view_scale * self.W)
        cx, cy = H / 2, W / 2

        radii = torch.FloatTensor(self.opt.ref_radii).to(self.device)
        thetas = torch.FloatTensor(self.opt.ref_polars).to(self.device)
        phis = torch.FloatTensor(self.opt.ref_azimuths).to(self.device)
        poses, dirs = circle_poses(
            self.device, radius=radii, theta=thetas, phi=phis,
            return_dirs=True, angle_overhead=self.opt.angle_overhead,
            angle_front=self.opt.angle_front,
        )
        focal = H / (2 * np.tan(np.deg2rad(self.opt.default_fovy) / 2))
        intrinsics = np.array([focal, focal, cx, cy])
        projection = self._projection_matrix(focal, H, W).unsqueeze(0).repeat(len(radii), 1, 1)
        mvp = projection @ torch.inverse(poses)

        rays = get_rays(poses, intrinsics, H, W, -1)
        return {
            'H': H, 'W': W,
            'rays_o': rays['rays_o'], 'rays_d': rays['rays_d'],
            'dir': dirs, 'mvp': mvp,
            'polar': self.opt.ref_polars,
            'azimuth': self.opt.ref_azimuths,
            'radius': self.opt.ref_radii,
        }

    def _projection_matrix(self, focal, H, W):
        return torch.tensor([
            [2 * focal / W, 0, 0, 0],
            [0, -2 * focal / H, 0, 0],
            [0, 0, -(self.far + self.near) / (self.far - self.near),
             -(2 * self.far * self.near) / (self.far - self.near)],
            [0, 0, -1, 0],
        ], dtype=torch.float32, device=self.device)

    def collate(self, index):
        if self.training:
            poses, dirs, thetas, phis, radius = rand_poses(
                len(index), self.device, self.opt,
                radius_range=self.opt.radius_range,
                theta_range=self.opt.theta_range,
                phi_range=self.opt.phi_range,
                return_dirs=True,
                angle_overhead=self.opt.angle_overhead,
                angle_front=self.opt.angle_front,
                uniform_sphere_rate=self.opt.uniform_sphere_rate,
            )
            fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) \
                + self.opt.fovy_range[0]
        elif self.type == 'six_views':
            thetas_six = [90, 90, 90, 90, 1e-3, 179.999]
            phis_six = [0, 90, 180, -90, 0, 0]
            thetas = torch.FloatTensor([thetas_six[index[0]]]).to(self.device)
            phis = torch.FloatTensor([phis_six[index[0]]]).to(self.device)
            radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)
            poses, dirs = circle_poses(
                self.device, radius=radius, theta=thetas, phi=phis,
                return_dirs=True,
                angle_overhead=self.opt.angle_overhead,
                angle_front=self.opt.angle_front,
            )
            fov = self.opt.default_fovy
        else:
            thetas = torch.FloatTensor([self.opt.default_polar]).to(self.device)
            phis = torch.FloatTensor([(index[0] / self.size) * 360]).to(self.device)
            radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)
            poses, dirs = circle_poses(
                self.device, radius=radius, theta=thetas, phi=phis,
                return_dirs=True,
                angle_overhead=self.opt.angle_overhead,
                angle_front=self.opt.angle_front,
            )
            fov = self.opt.default_fovy

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])
        projection = self._projection_matrix(focal, self.H, self.W).unsqueeze(0)
        mvp = projection @ torch.inverse(poses)

        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        delta_polar = thetas - self.opt.default_polar
        delta_azimuth = phis - self.opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360
        delta_radius = radius - self.opt.default_radius

        return {
            'H': self.H, 'W': self.W,
            'rays_o': rays['rays_o'], 'rays_d': rays['rays_d'],
            'dir': dirs, 'mvp': mvp,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius,
            'projection': projection,
            'poses': poses,
        }

    def dataloader(self, batch_size=None):
        batch_size = batch_size or self.opt.batch_size
        loader = DataLoader(
            list(range(self.size)),
            batch_size=batch_size, collate_fn=self.collate,
            shuffle=self.training, num_workers=0,
        )
        loader._data = self
        return loader
