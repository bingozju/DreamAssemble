"""DreamAssemble Trainer.

Wraps the model, guidance and optimizer; implements the parallelized
distillation sampling loss with edge-sparsity regularization (Eq. 11).
"""

import glob
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from PIL import Image
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from .utils import (
    adjust_text_embeddings, get_cpu_mem, get_gpu_mem, get_rays,
)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """End-to-end trainer for DreamAssemble.

    The trainer is intentionally backbone-agnostic: it consumes a NeRF
    model that exposes ``render`` and ``get_params``, and a guidance dict
    of diffusion priors with ``train_step`` / ``train_step_perpneg`` methods.
    """

    def __init__(self,
                 argv,
                 name,
                 opt,
                 model,
                 guidance,
                 criterion=None,
                 optimizer=None,
                 ema_decay=None,
                 lr_scheduler=None,
                 metrics=(),
                 local_rank=0,
                 world_size=1,
                 device=None,
                 mute=False,
                 fp16=False,
                 max_keep_ckpt=2,
                 workspace='workspace',
                 best_mode='min',
                 use_loss_as_metric=True,
                 report_metric_at_train=False,
                 use_checkpoint='latest',
                 use_tensorboardX=False,
                 scheduler_update_every_step=False):
        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = list(metrics)
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device or torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu',
        )
        self.console = Console()

        # -------- model / guidance --------
        model.to(self.device)
        if self.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        self.guidance = guidance
        self.embeddings = {}
        if self.guidance is not None:
            for key in self.guidance:
                for p in self.guidance[key].parameters():
                    p.requires_grad = False
                self.embeddings[key] = {}
            self.prepare_embeddings()

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # -------- optimizer --------
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=5e-4)
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1)
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        self.ema = (
            ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
            if ema_decay is not None else None
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # -------- bookkeeping --------
        self.total_train_t = 0
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            'loss': [], 'valid_loss': [], 'results': [],
            'checkpoints': [], 'best_result': None,
        }
        self.azimuth = 0
        if len(self.metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # -------- workspace --------
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f'log_{self.name}.txt')
            self.log_ptr = open(self.log_path, 'a+')
            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f'{self.ckpt_path}/{self.name}.pth'
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] opt: {self.opt}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | '
                 f'{self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: '
                 f'{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

        if self.workspace is not None:
            self._maybe_load_checkpoint()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _maybe_load_checkpoint(self):
        if self.use_checkpoint == 'scratch':
            self.log('[INFO] Training from scratch ...')
        elif self.use_checkpoint == 'latest':
            self.log('[INFO] Loading latest checkpoint ...')
            self.load_checkpoint()
        elif self.use_checkpoint == 'latest_model':
            self.log('[INFO] Loading latest checkpoint (model only) ...')
            self.load_checkpoint(model_only=True)
        elif self.use_checkpoint == 'best':
            if os.path.exists(self.best_path):
                self.log('[INFO] Loading best checkpoint ...')
                self.load_checkpoint(self.best_path)
            else:
                self.log(f'[INFO] {self.best_path} not found, loading latest ...')
                self.load_checkpoint()
        else:
            self.log(f'[INFO] Loading {self.use_checkpoint} ...')
            self.load_checkpoint(self.use_checkpoint)

    @torch.no_grad()
    def prepare_embeddings(self):
        """Pre-compute view-conditioned text embeddings for all sub-prompts."""
        if self.opt.text is None:
            return

        part_texts = self.opt.part_texts

        for guidance_name in ('SD', 'IF'):
            if guidance_name not in self.guidance:
                continue
            encoder = self.guidance[guidance_name]
            self.embeddings[guidance_name]['default'] = encoder.get_text_embeds([self.opt.text])
            self.embeddings[guidance_name]['uncond'] = encoder.get_text_embeds([self.opt.negative])

            for part_idx in range(len(part_texts)):
                self.embeddings[f'part{part_idx}'] = {}

            for direction in ('front', 'side', 'back'):
                self.embeddings[guidance_name][f'{direction}0'] = encoder.get_text_embeds(
                    [self._directional_prompt(self.opt.text, direction)]
                )
                for part_idx, part_text in enumerate(part_texts):
                    self.embeddings[f'part{part_idx}'][f'{direction}0'] = encoder.get_text_embeds(
                        [self._directional_prompt(part_text, direction)]
                    )

    @staticmethod
    def _directional_prompt(prompt: str, direction: str) -> str:
        """Wrap ``prompt`` with a richer view qualifier."""
        if direction == 'side':
            return f'{prompt}, side view, profile, lateral view.'
        if direction == 'back':
            return f'{prompt}, back view, rear view, from behind.'
        return f'{prompt}, front view'

    # ------------------------------------------------------------------
    # Logging / lifecycle
    # ------------------------------------------------------------------

    def __del__(self):
        if getattr(self, 'log_ptr', None):
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank != 0:
            return
        if not self.mute:
            self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()

    # ==================================================================
    # Forward step (single training iteration)
    # ==================================================================

    def train_step(self, data, save_guidance_path: Path = None):
        """Run one distillation-sampling iteration.

        Returns a dict containing ``pred_rgb``, ``pred_rgb_parts``,
        ``pred_depth``, ``loss`` and the per-subspace render snapshots.
        """

        # ---------- 1. progressive scheduling ----------
        exp_iter_ratio = (self.global_step - self.opt.exp_start_iter) / max(
            1, self.opt.exp_end_iter - self.opt.exp_start_iter,
        )

        if self.opt.progressive_view and not self.opt.dont_override_stuff:
            r = min(1.0, self.opt.progressive_view_init_ratio + 2.0 * exp_iter_ratio)
            self.opt.phi_range = [
                self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[0] * r,
                self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[1] * r,
            ]
            self.opt.theta_range = [
                self.opt.default_polar * (1 - r) + self.opt.full_theta_range[0] * r,
                self.opt.default_polar * (1 - r) + self.opt.full_theta_range[1] * r,
            ]
            self.opt.radius_range = [
                self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r,
            ]
            self.opt.fovy_range = [
                self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r,
            ]
        if self.opt.progressive_level:
            self.model.max_level = min(1.0, 0.25 + 2.0 * exp_iter_ratio)

        # ---------- 2. ray sampling & shading mode ----------
        rays_o = data['rays_o']
        rays_d = data['rays_d']
        mvp = data['mvp']
        B, _ = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if exp_iter_ratio <= self.opt.latent_iter_ratio:
            ambient_ratio, shading, as_latent, bg_color = 1.0, 'normal', True, None
        else:
            if exp_iter_ratio <= self.opt.albedo_iter_ratio:
                ambient_ratio, shading = 1.0, 'albedo'
            else:
                ambient_ratio = self.opt.min_ambient_ratio + (
                    1.0 - self.opt.min_ambient_ratio) * random.random()
                if random.random() >= 1.0 - self.opt.textureless_ratio:
                    shading = 'textureless'
                else:
                    shading = 'lambertian'
            if self.opt.bg_radius > 0 and random.random() > 0.5:
                bg_color = None
            else:
                bg_color = torch.rand(3, device=self.device)
            as_latent = False

        # ---------- 3. forward render ----------
        outputs = self.model.render(
            rays_o, rays_d, mvp, H, W,
            staged=False, perturb=True,
            bg_color=bg_color, ambient_ratio=ambient_ratio,
            shading=shading, binarize=False,
        )
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_normal = outputs['normal_image'].reshape(B, H, W, 3) if 'normal_image' in outputs else None

        part_nums = self.opt.part_nums
        pred_rgb, pred_rgb_parts, pred_rgb_parts_save, margin_loss = self._build_pred_rgbs(
            outputs, data, B, H, W, as_latent,
        )

        # ---------- 4. distillation losses ----------
        loss = 0
        if 'SD' in self.guidance:
            loss = loss + self._compute_sd_loss(
                data, exp_iter_ratio, pred_rgb, pred_rgb_parts, as_latent, save_guidance_path,
            )
        if 'IF' in self.guidance:
            loss = loss + self._compute_if_loss(
                data, exp_iter_ratio, pred_rgb, pred_rgb_parts,
            )

        # ---------- 5. regularizations ----------
        if not self.opt.dmtet:
            loss = loss + self.opt.lambda_edge_sparsity * margin_loss

            if self.opt.lambda_opacity > 0:
                loss = loss + self.opt.lambda_opacity * (outputs['weights_sum'] ** 2).mean()

            if self.opt.lambda_entropy > 0:
                lambda_entropy = self.opt.lambda_entropy * min(
                    1, 5 * self.global_step / self.opt.iters,
                )
                for j in range(part_nums):
                    alphas_p = outputs[f'weights_{j}'].clamp(1e-5, 1 - 1e-5)
                    entropy = -(alphas_p * torch.log2(alphas_p)
                                + (1 - alphas_p) * torch.log2(1 - alphas_p)).mean()
                    loss = loss + lambda_entropy * entropy

            if self.opt.lambda_2d_normal_smooth > 0 and pred_normal is not None:
                smooth = (
                    (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean()
                    + (pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]).square().mean()
                )
                loss = loss + self.opt.lambda_2d_normal_smooth * smooth

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
                loss = loss + self.opt.lambda_orient * outputs['loss_orient']

            if self.opt.lambda_3d_normal_smooth > 0 and 'loss_normal_perturb' in outputs:
                loss = loss + self.opt.lambda_3d_normal_smooth * outputs['loss_normal_perturb']
        else:
            if self.opt.lambda_mesh_normal > 0:
                loss = loss + self.opt.lambda_mesh_normal * outputs['normal_loss']
            if self.opt.lambda_mesh_laplacian > 0:
                loss = loss + self.opt.lambda_mesh_laplacian * outputs['lap_loss']

        return {
            'pred_rgb': pred_rgb,
            'pred_rgb_parts': pred_rgb_parts,
            'pred_depth': pred_depth,
            'loss': loss,
            'pred_rgb_parts_save': pred_rgb_parts_save,
        }

    # ------------------------------------------------------------------
    # train_step helpers
    # ------------------------------------------------------------------

    def _build_pred_rgbs(self, outputs, data, B, H, W, as_latent):
        """Compute the global RGB and per-subspace cropped RGBs.

        Implements the radial cropping around each subspace's projected
        center (mu^j_center) and accumulates the edge-sparsity term
        (Eq. 11 in the paper).
        """
        part_nums = self.opt.part_nums

        if as_latent:
            # Use weight_sum as the alpha channel (Fantasia3D-style).
            global_image = torch.cat(
                [outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1,
            ).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous()
        else:
            global_image = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

        pred_rgb_parts = []
        pred_rgb_parts_save = []
        margin_loss = 0

        for part_idx in range(part_nums):
            if as_latent:
                rgb_p = torch.cat(
                    [outputs[f'image_{part_idx}'],
                     outputs[f'weights_sum_{part_idx}'].unsqueeze(-1)], dim=-1,
                ).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous()
            else:
                rgb_p = outputs[f'image_{part_idx}'].reshape(B, H, W, -1) \
                    .permute(0, 3, 1, 2).contiguous()

            pred_rgb_parts_save.append(rgb_p[0])

            # Project mu^j_center onto the image plane to get (u, v).
            part_center = self.model.x_centers[part_idx]
            point_hom = torch.tensor(
                [part_center[0], part_center[1], part_center[2], 1],
                device=rgb_p.device, dtype=rgb_p.dtype,
            )
            point_proj = data['projection'][0] @ torch.inverse(data['poses'])[0] @ point_hom
            x_proj, y_proj, w_proj = point_proj[0], point_proj[1], point_proj[3]
            u = int(x_proj / w_proj * H // 2 + H // 2)
            v = int(y_proj / w_proj * W // 2 + W // 2)

            # Radial crop radius limited by image boundary and per-part scale.
            min_dist = min(u, (H - 1) - u, v, (W - 1) - v)
            min_dist = int(min_dist * self.opt.part_scales[part_idx])

            if min_dist > 5:
                # Edge sparsity regularization: penalise opacity outside the crop.
                weights_sum_p = outputs[f'weights_sum_{part_idx}'] \
                    .reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                mask = torch.ones_like(weights_sum_p)
                mask[:, :, v - min_dist: v + min_dist + 1,
                     u - min_dist: u + min_dist + 1] = 0
                margin_loss = margin_loss + torch.nan_to_num(
                    torch.sum(weights_sum_p * mask.detach())
                )

                rgb_p = rgb_p[
                    :, :,
                    v - min_dist: v + min_dist + 1,
                    u - min_dist: u + min_dist + 1,
                ]
                rgb_p = F.interpolate(rgb_p, (W, H), mode='bilinear', align_corners=False)

            pred_rgb_parts.append(rgb_p)

        pred_rgb_parts = torch.cat(pred_rgb_parts, dim=0)
        return global_image, pred_rgb_parts, pred_rgb_parts_save, margin_loss

    def _build_view_text(self, guidance_key, azimuth, part_nums):
        """Concatenate per-view text embeddings for the global prompt and each sub-prompt."""
        # Unconditional anchors.
        text_z = [self.embeddings[guidance_key]['uncond']] * azimuth.shape[0]
        text_p = {f'{j}': [self.embeddings[guidance_key]['uncond']] * azimuth.shape[0]
                  for j in range(part_nums)}

        weights_parts = []
        if self.opt.perpneg:
            text_z_comp, weights = adjust_text_embeddings(
                self.embeddings[guidance_key], azimuth, self.opt,
            )
            text_z.append(text_z_comp)
            for j in range(part_nums):
                z_comp_p, w_p = adjust_text_embeddings(
                    self.embeddings[f'part{j}'], azimuth, self.opt,
                )
                weights_parts.append(w_p)
                text_p[f'{j}'].append(z_comp_p)
        else:
            weights = None
            for b in range(azimuth.shape[0]):
                if -90 <= azimuth[b] < 90:
                    if azimuth[b] >= 0:
                        r = 1 - azimuth[b] / 90
                    else:
                        r = 1 + azimuth[b] / 90
                    direction = 'side'
                    start_z = self.embeddings[guidance_key]['front0']
                    end_z = self.embeddings[guidance_key][f'{direction}0']
                    for j in range(part_nums):
                        text_p[f'{j}'].append(
                            self.embeddings[f'part{j}']['front0'] * r
                            + self.embeddings[f'part{j}'][f'{direction}0'] * (1 - r)
                        )
                else:
                    if azimuth[b] >= 0:
                        r = 1 - (azimuth[b] - 90) / 90
                    else:
                        r = 1 + (azimuth[b] + 90) / 90
                    direction = 'side'
                    start_z = self.embeddings[guidance_key][f'{direction}0']
                    end_z = self.embeddings[guidance_key]['back0']
                    for j in range(part_nums):
                        text_p[f'{j}'].append(
                            self.embeddings[f'part{j}'][f'{direction}0'] * r
                            + self.embeddings[f'part{j}']['back0'] * (1 - r)
                        )
                text_z.append(r * start_z + (1 - r) * end_z)

        text_z = torch.cat(text_z, dim=0)
        text_parts = torch.cat(
            [torch.cat(text_p[f'{j}'], dim=0) for j in range(part_nums)], dim=0,
        )
        return text_z, text_parts, weights, weights_parts

    def _select_input_pair(self, exp_iter_ratio, text_z, text_parts,
                          pred_rgb, pred_rgb_parts, weights, weights_parts,
                          part_nums, perpneg, period=10):
        """Decide which (text, image) pair to feed the diffusion guidance.

        DreamAssemble alternates between the global prompt and randomly
        chosen sub-prompts to balance global coherence and local fidelity.
        Cf. Sec. 4.3 ("Parallelized Distillation Sampling") of the paper.
        """
        if self.opt.train_all:
            input_text = torch.cat([text_z, text_parts], dim=0)
            input_rgb = torch.cat([pred_rgb, pred_rgb_parts], dim=0)
            if perpneg:
                weights = torch.stack([weights] + weights_parts, dim=0)
                input_text = torch.stack(
                    [text_z] + list(torch.chunk(text_parts, chunks=part_nums)), dim=0,
                )
            return input_text, input_rgb, weights

        p_i = (self.global_step // period) % part_nums
        q_i = ((self.global_step + 1) // period) % part_nums

        if exp_iter_ratio < 0.2:
            input_text = torch.cat(
                [text_z, text_parts[2 * p_i:2 * p_i + 2], text_parts[2 * q_i:2 * q_i + 2]], dim=0,
            )
            input_rgb = torch.cat(
                [pred_rgb, pred_rgb_parts[p_i:p_i + 1], pred_rgb_parts[q_i:q_i + 1]], dim=0,
            )
            if perpneg:
                input_text = torch.stack(
                    [text_z, text_parts[4 * p_i:4 * p_i + 4]], dim=0,
                )
                weights = torch.stack([weights, weights_parts[p_i]], dim=0)
        else:
            if self.global_step % 100 >= 50:
                input_text = torch.cat(
                    [text_z, text_parts[2 * p_i:2 * p_i + 2]], dim=0,
                )
                input_rgb = torch.cat(
                    [pred_rgb, pred_rgb_parts[p_i:p_i + 1]], dim=0,
                )
                if perpneg:
                    weights = torch.stack([weights], dim=0)
                    input_text = torch.stack([text_z], dim=0)
            else:
                input_text = text_parts
                input_rgb = pred_rgb_parts
                if perpneg:
                    input_text = torch.chunk(text_parts, chunks=part_nums)
                    weights = torch.stack(weights_parts, dim=0)

        return input_text, input_rgb, weights

    def _compute_sd_loss(self, data, exp_iter_ratio,
                        pred_rgb, pred_rgb_parts, as_latent, save_guidance_path):
        azimuth = data['azimuth']
        part_nums = self.opt.part_nums
        text_z, text_parts, weights, weights_parts = self._build_view_text(
            'SD', azimuth, part_nums,
        )

        input_text, input_rgb, weights = self._select_input_pair(
            exp_iter_ratio, text_z, text_parts,
            pred_rgb, pred_rgb_parts, weights, weights_parts,
            part_nums, self.opt.perpneg,
        )

        loss = 0
        if self.opt.perpneg:
            for i in range(weights.shape[0]):
                loss = loss + self.guidance['SD'].train_step_perpneg(
                    input_text[i], weights[i], input_rgb[i:i + 1],
                    as_latent=as_latent,
                    guidance_scale=self.opt.guidance_scale,
                    grad_scale=self.opt.lambda_guidance,
                    save_guidance_path=save_guidance_path,
                ) / weights.shape[0]
        else:
            loss = loss + self.guidance['SD'].train_step(
                input_text, input_rgb,
                as_latent=as_latent,
                exp_iter_ratio=exp_iter_ratio,
                dmtet=self.opt.dmtet,
                guidance_scale=self.opt.guidance_scale,
                grad_scale=self.opt.lambda_guidance,
                save_guidance_path=save_guidance_path,
            )
        return loss

    def _compute_if_loss(self, data, exp_iter_ratio, pred_rgb, pred_rgb_parts):
        azimuth = data['azimuth']
        part_nums = self.opt.part_nums
        text_z, text_parts, weights, weights_parts = self._build_view_text(
            'IF', azimuth, part_nums,
        )

        input_text, input_rgb, weights = self._select_input_pair(
            exp_iter_ratio, text_z, text_parts,
            pred_rgb, pred_rgb_parts, weights, weights_parts,
            part_nums, self.opt.perpneg,
        )

        loss = 0
        if self.opt.perpneg:
            for i in range(weights.shape[0]):
                loss = loss + self.guidance['IF'].train_step_perpneg(
                    input_text[i], weights[i], input_rgb[i:i + 1],
                    guidance_scale=self.opt.guidance_scale,
                    grad_scale=self.opt.lambda_guidance,
                ) / weights.shape[0]
        else:
            loss = loss + self.guidance['IF'].train_step(
                input_text, input_rgb,
                guidance_scale=self.opt.guidance_scale,
                grad_scale=self.opt.lambda_guidance,
                exp_iter_ratio=exp_iter_ratio,
                dmtet=self.opt.dmtet,
            )
        return loss

    # ==================================================================
    # Eval / test step
    # ==================================================================

    def post_train_step(self):
        self.scaler.unscale_(self.optimizer)
        if self.opt.grad_clip >= 0:
            nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

        if not self.opt.dmtet and self.opt.backbone == 'grid':
            if self.opt.lambda_tv > 0:
                lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_tv
                self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
            if self.opt.lambda_wd > 0:
                self.model.encoder.grad_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):
        rays_o = data['rays_o']
        rays_d = data['rays_d']
        mvp = data['mvp']
        B, _ = rays_o.shape[:2]
        H, W = data['H'], data['W']

        bg_color = torch.ones(3, device=self.device)
        shading = data.get('shading', 'albedo')
        ambient_ratio = data.get('ambient_ratio', 1.0)
        light_d = data.get('light_d', None)

        outputs = self.model.render(
            rays_o, rays_d, mvp, H, W,
            staged=True, perturb=False, bg_color=bg_color, light_d=light_d,
            ambient_ratio=ambient_ratio, shading=shading,
        )
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        pred_rgb_parts = [pred_rgb] + [
            outputs[f'image_{j}'].reshape(B, H, W, 3)
            for j in range(self.opt.part_nums)
        ]
        pred_rgb_parts = torch.cat(pred_rgb_parts, dim=2)
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)
        return pred_rgb, pred_depth, loss, pred_rgb_parts

    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']
        rays_d = data['rays_d']
        mvp = data['mvp']
        B, _ = rays_o.shape[:2]
        H, W = data['H'], data['W']

        bg_color = torch.ones(3, device=self.device)
        shading = data.get('shading', 'albedo')
        ambient_ratio = data.get('ambient_ratio', 1.0)
        light_d = data.get('light_d', None)

        outputs = self.model.render(
            rays_o, rays_d, mvp, H, W,
            staged=True, perturb=perturb, light_d=light_d,
            ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color,
        )
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        pred_rgb_parts = [pred_rgb] + [
            outputs[f'image_{j}'].reshape(B, H, W, 3)
            for j in range(self.opt.part_nums)
        ]
        pred_rgb_parts = torch.cat(pred_rgb_parts, dim=2)
        return pred_rgb, pred_depth, pred_rgb_parts

    def save_mesh(self, save_path=None):
        save_path = save_path or os.path.join(self.workspace, 'mesh')
        self.log(f'==> Saving mesh to {save_path}')
        os.makedirs(save_path, exist_ok=True)
        self.model.export_mesh(
            save_path,
            resolution=self.opt.mcubes_resolution,
            decimate_target=self.opt.decimate_target,
        )
        self.log('==> Finished saving mesh.')

    # ==================================================================
    # Training driver
    # ==================================================================

    def train(self, train_loader, valid_loader, test_loader, max_epochs):
        start_t = time.time()
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(train_loader, max_epochs)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.opt.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

            if self.epoch % self.opt.test_interval == 0 and self.epoch < max_epochs - 10:
                self.test(test_loader)

        self.test(test_loader)
        self.total_train_t += time.time() - start_t
        self.log(f'[INFO] training takes {self.total_train_t / 60:.4f} minutes.')

    def evaluate(self, loader, name=None):
        self.evaluate_one_epoch(loader, name)

    def test(self, loader, save_path=None, name=None, write_video=True):
        save_path = save_path or os.path.join(self.workspace, 'results')
        name = name or f'{self.name}_ep{self.epoch:04d}'
        os.makedirs(save_path, exist_ok=True)
        self.log(f'==> Start Test, save results to {save_path}')

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} '
                                    '[{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_rgb, all_depth, all_parts = [], [], []
        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_parts = self.test_step(data)

                pred = (preds[0].detach().cpu().numpy() * 255).astype(np.uint8)
                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (
                    pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)
                pred_parts = (preds_parts[0].detach().cpu().numpy() * 255).astype(np.uint8)

                if write_video:
                    all_rgb.append(pred)
                    all_depth.append(pred_depth)
                    all_parts.append(pred_parts)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'),
                                cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'),
                                pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_rgb = np.stack(all_rgb, axis=0)
            all_depth = np.stack(all_depth, axis=0)
            all_parts = np.stack(all_parts, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'),
                             all_rgb, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'),
                             all_depth, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_parts.mp4'),
                             all_parts, fps=25, quality=8, macro_block_size=1)

        self.log('==> Finished Test.')

    # ------------------------------------------------------------------
    # Per-epoch loops
    # ------------------------------------------------------------------

    def train_one_epoch(self, loader, max_epochs):
        self.log(f'==> [{time.strftime("%Y-%m-%d_%H-%M-%S")}] '
                 f'Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, '
                 f'lr={self.optimizer.param_groups[0]["lr"]:.6f} ...')

        total_loss = 0
        self.model.train()
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} '
                                        '[{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        save_guidance_folder = None
        if self.opt.save_guidance:
            save_guidance_folder = Path(self.workspace) / 'guidance'
            save_guidance_folder.mkdir(parents=True, exist_ok=True)

        for data in loader:
            if (self.model.cuda_ray or self.model.taichi_ray) \
                    and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                save_guidance_path = (
                    save_guidance_folder / f'step_{self.global_step:07d}.png'
                    if save_guidance_folder
                    and (self.global_step % self.opt.save_guidance_interval == 0)
                    else None
                )
                ret = self.train_step(data, save_guidance_path=save_guidance_path)
                pred_rgb = ret['pred_rgb']
                loss = ret['loss']

                if self.global_step % 50 == 0 or (
                        self.global_step < 100 and self.global_step % 5 == 0):
                    self._save_debug_images(ret)

            if self.opt.grad_clip_rgb >= 0:
                pred_rgb.register_hook(self._make_rgb_grad_hook())

            self.scaler.scale(loss).backward()
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                pbar.set_description(
                    f'loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), '
                    f'lr={self.optimizer.param_groups[0]["lr"]:.6f}'
                )
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats['loss'].append(average_loss)

        if self.local_rank == 0:
            pbar.close()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        cpu_mem, gpu_mem = get_cpu_mem(), get_gpu_mem()[0]
        self.log(f'==> [{time.strftime("%Y-%m-%d_%H-%M-%S")}] '
                 f'Finished Epoch {self.epoch}/{max_epochs}. '
                 f'CPU={cpu_mem:.1f}GB, GPU={gpu_mem:.1f}GB.')

    def _make_rgb_grad_hook(self):
        def _hook(grad):
            if self.opt.fp16:
                grad_scale = self.scaler._get_scale_async()
                return grad.clamp(grad_scale * -self.opt.grad_clip_rgb,
                                  grad_scale * self.opt.grad_clip_rgb)
            return grad.clamp(-self.opt.grad_clip_rgb, self.opt.grad_clip_rgb)
        return _hook

    def _save_debug_images(self, ret):
        pred_rgb = ret['pred_rgb']
        pred_rgb_parts = ret['pred_rgb_parts']
        pred_rgb_parts_save = ret['pred_rgb_parts_save']

        rgb_np = (255. * pred_rgb[0, :3].detach().cpu().permute(1, 2, 0).numpy()) \
            .round().astype(np.uint8)
        Image.fromarray(rgb_np).save(f'{self.workspace}/debug_{self.global_step}.jpg')

        all_preds = [pred_rgb[0, :3]] + [
            pred_rgb_parts[j, :3] for j in range(self.opt.part_nums)
        ]
        all_preds = torch.cat(all_preds, dim=2)
        all_np = (255. * all_preds.detach().cpu().permute(1, 2, 0).numpy()) \
            .round().astype(np.uint8)
        Image.fromarray(all_np).save(f'{self.workspace}/debug_{self.global_step}_all.jpg')

        parts_save = torch.cat(pred_rgb_parts_save, dim=2)
        parts_np = (255. * parts_save[:3].detach().cpu().permute(1, 2, 0).numpy()) \
            .round().astype(np.uint8)
        Image.fromarray(parts_np).save(f'{self.workspace}/debug_{self.global_step}_debug.jpg')

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f'++> Evaluate {self.workspace} at epoch {self.epoch} ...')
        name = name or f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        self.model.eval()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} '
                                        '[{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            for data in loader:
                self.local_step += 1
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, loss, preds_parts = self.eval_step(data)

                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                loss_val = loss.item()
                total_loss += loss_val

                if self.local_rank == 0:
                    save_dir = os.path.join(self.workspace, 'validation')
                    os.makedirs(save_dir, exist_ok=True)

                    pred = (preds[0].detach().cpu().numpy() * 255).astype(np.uint8)
                    pred_parts = (preds_parts[0].detach().cpu().numpy() * 255).astype(np.uint8)
                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    az = data['azimuth'].item()
                    cv2.imwrite(os.path.join(save_dir, f'{name}_{self.local_step:04d}_{az}_rgb.png'),
                                cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_dir, f'{name}_{self.local_step:04d}_depth.png'),
                                pred_depth)
                    cv2.imwrite(os.path.join(save_dir, f'{name}_{self.local_step:04d}_parts.png'),
                                cv2.cvtColor(pred_parts, cv2.COLOR_RGB2BGR))

                    pbar.set_description(
                        f'loss={loss_val:.4f} ({total_loss / self.local_step:.4f})'
                    )
                    pbar.update(loader.batch_size)

        average_loss = total_loss / max(1, self.local_step)
        self.stats['valid_loss'].append(average_loss)
        if self.local_rank == 0:
            pbar.close()
            self.stats['results'].append(average_loss)

        if self.ema is not None:
            self.ema.restore()
        self.log(f'++> Evaluate epoch {self.epoch} Finished.')

    # ==================================================================
    # Checkpointing
    # ==================================================================

    def save_checkpoint(self, name=None, full=False, best=False):
        name = name or f'{self.name}_ep{self.epoch:04d}'
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }
        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density
        if self.opt.dmtet:
            state['tet_scale'] = self.model.tet_scale.cpu().numpy()
        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:
            state['model'] = self.model.state_dict()
            file_path = f'{name}.pth'
            self.stats['checkpoints'].append(file_path)
            if len(self.stats['checkpoints']) > self.max_keep_ckpt:
                old = os.path.join(self.ckpt_path, self.stats['checkpoints'].pop(0))
                if os.path.exists(old):
                    os.remove(old)
            torch.save(state, os.path.join(self.ckpt_path, file_path))
        else:
            if len(self.stats['results']) == 0:
                self.log('[WARN] no evaluated results found, skip saving best checkpoint.')
                return
            if self.ema is not None:
                self.ema.store()
                self.ema.copy_to()
            state['model'] = self.model.state_dict()
            if self.ema is not None:
                self.ema.restore()
            torch.save(state, self.best_path)

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            ckpt_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if not ckpt_list:
                self.log('[WARN] No checkpoint found, model randomly initialized.')
                return
            checkpoint = ckpt_list[-1]
            self.log(f'[INFO] Latest checkpoint is {checkpoint}')

        ckpt = torch.load(checkpoint, map_location=self.device)
        if 'model' not in ckpt:
            self.model.load_state_dict(ckpt)
            self.log('[INFO] loaded model.')
            return

        missing, unexpected = self.model.load_state_dict(ckpt['model'], strict=False)
        self.log('[INFO] loaded model.')
        if missing:
            self.log(f'[WARN] missing keys: {missing}')
        if unexpected:
            self.log(f'[WARN] unexpected keys: {unexpected}')

        if self.ema is not None and 'ema' in ckpt:
            try:
                self.ema.load_state_dict(ckpt['ema'])
            except Exception:
                self.log('[WARN] failed to load EMA.')

        if self.model.cuda_ray and 'mean_density' in ckpt:
            self.model.mean_density = ckpt['mean_density']

        if self.opt.dmtet and 'tet_scale' in ckpt:
            new_scale = torch.from_numpy(ckpt['tet_scale']).to(self.device)
            self.model.verts *= new_scale / self.model.tet_scale
            self.model.tet_scale = new_scale

        if model_only:
            return

        self.stats = ckpt['stats']
        self.epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.log(f'[INFO] load at epoch {self.epoch}, global step {self.global_step}')

        for attr, key in [('optimizer', 'optimizer'),
                          ('lr_scheduler', 'lr_scheduler'),
                          ('scaler', 'scaler')]:
            obj = getattr(self, attr, None)
            if obj is not None and key in ckpt:
                try:
                    obj.load_state_dict(ckpt[key])
                except Exception:
                    self.log(f'[WARN] Failed to load {key}.')
