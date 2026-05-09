"""
DreamAssemble: Complex Multi-object Text-to-3D Generation
via Multi-Density Neural Fields.

Main training / testing entry point.
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from nerf.provider import NeRFDataset
from nerf.trainer import Trainer
from nerf.utils import seed_everything


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Make a training run deterministic across torch / numpy / random."""
    import random as _random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DreamAssemble training entry.")

    # ---- required ----
    parser.add_argument("--config", type=str, required=True,
                        help="Path to yaml config under trainfiles/.")
    parser.add_argument("--workspace", type=str, default="workspace",
                        help="Run name; output goes under results/<workspace>.")
    parser.add_argument("--seed", type=int, default=42)

    # ---- mode ----
    parser.add_argument("--test", action="store_true", help="test mode (no training)")
    parser.add_argument("--six_views", action="store_true", help="render six canonical views")
    parser.add_argument("--save_mesh", action="store_true", help="export textured mesh")
    parser.add_argument("--train_all", action="store_true",
                        help="always optimize global and sub-prompts together")

    # ---- guidance ----
    parser.add_argument("--guidance", type=str, nargs="*", default=["SD"],
                        choices=["SD", "IF"], help="diffusion guidance models")
    parser.add_argument("--guidance_scale", type=float, default=100,
                        help="classifier-free guidance scale")
    parser.add_argument("--negative", type=str, default="", help="negative prompt")
    parser.add_argument("--hf_key", type=str, default=None,
                        help="HuggingFace SD checkpoint key (overrides default)")

    # ---- DMTet refinement ----
    parser.add_argument("--dmtet", action="store_true", help="enable DMTet stage")
    parser.add_argument("--tet_grid_size", type=int, default=256,
                        choices=[32, 64, 128, 256])
    parser.add_argument("--init_with", type=str, default="",
                        help="checkpoint to initialize DMTet (auto-resolved if empty)")
    parser.add_argument("--lock_geo", action="store_true",
                        help="freeze DMTet geometry, only learn texture")

    # ---- Perp-Neg ----
    parser.add_argument("--perpneg", action="store_true",
                        help="use Perp-Neg sampler (default: True at runtime)")
    parser.add_argument("--negative_w", type=float, default=-2)
    parser.add_argument("--front_decay_factor", type=float, default=2)
    parser.add_argument("--side_decay_factor", type=float, default=10)

    # ---- training ----
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt", type=str, default="latest",
                        choices=["latest", "scratch", "best", "latest_model"])
    parser.add_argument("--optim", type=str, default="adan", choices=["adan", "adam"])
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vram_O", action="store_true", help="trade speed for VRAM")
    parser.add_argument("--cuda_ray", action="store_true",
                        help="use CUDA raymarching (faster, requires extension)")
    parser.add_argument("--taichi_ray", action="store_true")
    parser.add_argument("--max_steps", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--upsample_steps", type=int, default=32)
    parser.add_argument("--update_extra_interval", type=int, default=16)
    parser.add_argument("--max_ray_batch", type=int, default=4096)

    # ratios for shading scheduling
    parser.add_argument("--latent_iter_ratio", type=float, default=0.2)
    parser.add_argument("--albedo_iter_ratio", type=float, default=0.0)
    parser.add_argument("--min_ambient_ratio", type=float, default=0.1)
    parser.add_argument("--textureless_ratio", type=float, default=0.2)

    # camera jitter
    parser.add_argument("--jitter_pose", action="store_true")
    parser.add_argument("--jitter_center", type=float, default=0.2)
    parser.add_argument("--jitter_target", type=float, default=0.2)
    parser.add_argument("--jitter_up", type=float, default=0.02)
    parser.add_argument("--uniform_sphere_rate", type=float, default=0.0)

    # gradient clipping
    parser.add_argument("--grad_clip", type=float, default=-1)
    parser.add_argument("--grad_clip_rgb", type=float, default=-1)

    # network backbone (only one is supported in this release)
    parser.add_argument("--backbone", type=str, default="grid_mask",
                        choices=["grid_mask"], help="multi-density grid backbone")

    # rendering resolution during training
    parser.add_argument("--w", type=int, default=64, help="train render width")
    parser.add_argument("--h", type=int, default=64, help="train render height")
    parser.add_argument("--known_view_scale", type=float, default=1.5)
    parser.add_argument("--known_view_noise_scale", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=1)

    # GUI / test render resolution
    parser.add_argument("--W", type=int, default=800, help="test render width")
    parser.add_argument("--H", type=int, default=800, help="test render height")
    parser.add_argument("--gui", action="store_true")

    # dataset / scene bounds
    parser.add_argument("--bound", type=float, default=1)
    parser.add_argument("--dt_gamma", type=float, default=0)
    parser.add_argument("--min_near", type=float, default=0.01)
    parser.add_argument("--angle_overhead", type=float, default=30)
    parser.add_argument("--angle_front", type=float, default=60)
    parser.add_argument("--t_range", type=float, nargs="*", default=[0.02, 0.98])
    parser.add_argument("--dont_override_stuff", action="store_true",
                        help="do not let dmtet/image branches override t_range etc.")
    parser.add_argument("--progressive_view", action="store_true")
    parser.add_argument("--progressive_view_init_ratio", type=float, default=0.2)
    parser.add_argument("--progressive_level", action="store_true")

    # regularization weights
    parser.add_argument("--lambda_entropy", type=float, default=1e-1)
    parser.add_argument("--lambda_opacity", type=float, default=0)
    parser.add_argument("--lambda_orient", type=float, default=1e-2)
    parser.add_argument("--lambda_tv", type=float, default=0)
    parser.add_argument("--lambda_wd", type=float, default=0)
    parser.add_argument("--lambda_mesh_normal", type=float, default=0.5)
    parser.add_argument("--lambda_mesh_laplacian", type=float, default=0.5)
    parser.add_argument("--lambda_guidance", type=float, default=1)
    parser.add_argument("--lambda_rgb", type=float, default=1000)
    parser.add_argument("--lambda_mask", type=float, default=500)
    parser.add_argument("--lambda_normal", type=float, default=0)
    parser.add_argument("--lambda_depth", type=float, default=10)
    parser.add_argument("--lambda_2d_normal_smooth", type=float, default=0)
    parser.add_argument("--lambda_3d_normal_smooth", type=float, default=0)
    parser.add_argument("--lambda_edge_sparsity", type=float, default=0.1,
                        help="weight k of the edge sparsity regularization (Eq. 11).")

    # debugging / logging
    parser.add_argument("--save_guidance", action="store_true")
    parser.add_argument("--save_guidance_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=3)
    parser.add_argument("--test_interval", type=int, default=100)

    # dataset sizes
    parser.add_argument("--dataset_size_train", type=int, default=100)
    parser.add_argument("--dataset_size_valid", type=int, default=16)
    parser.add_argument("--dataset_size_test", type=int, default=100)

    # progressive view experiment range
    parser.add_argument("--exp_start_iter", type=int, default=None)
    parser.add_argument("--exp_end_iter", type=int, default=None)

    # mesh extraction
    parser.add_argument("--mcubes_resolution", type=int, default=256)
    parser.add_argument("--decimate_target", type=int, default=int(5e4))

    # ---- scene / camera defaults (typically overridden by YAML) ----
    parser.add_argument("--text", type=str, default=None, help="global prompt")
    parser.add_argument("--part_texts", type=str, default=None,
                        help="semicolon-separated sub-prompts")
    parser.add_argument("--part_centers", type=float, nargs="*", default=None,
                        help="(unused; expected from YAML as a list of triples)")
    parser.add_argument("--part_scales", type=float, nargs="*", default=None)
    parser.add_argument("--parts_blob_radius", type=float, nargs="*", default=None)

    parser.add_argument("--IF", action="store_true", default=False,
                        help="use DeepFloyd IF instead of Stable Diffusion")
    parser.add_argument("--sd_version", type=str, default="2.1",
                        choices=["1.5", "2.1"])

    parser.add_argument("--density_activation", type=str, default="exp",
                        choices=["exp", "softplus"])
    parser.add_argument("--radius_range", type=float, nargs=2, default=[3.0, 3.5])
    parser.add_argument("--default_radius", type=float, default=3.2)
    parser.add_argument("--fovy_range", type=float, nargs=2, default=[19, 21])
    parser.add_argument("--default_fovy", type=float, default=20.0)
    parser.add_argument("--theta_range", type=float, nargs=2, default=[45, 105])
    parser.add_argument("--phi_range", type=float, nargs=2, default=[-180, 180])
    parser.add_argument("--default_polar", type=float, default=90.0)
    parser.add_argument("--default_azimuth", type=float, default=0.0)
    parser.add_argument("--bg_radius", type=float, default=1.4)
    parser.add_argument("--density_thresh", type=float, default=1.0)
    parser.add_argument("--blob_density", type=float, default=5.0)
    parser.add_argument("--blob_radius", type=float, default=0.2)

    # reference views (used by NeRFDataset.get_default_view_data)
    parser.add_argument("--ref_radii", type=float, nargs="*", default=[3.2])
    parser.add_argument("--ref_polars", type=float, nargs="*", default=[90.0])
    parser.add_argument("--ref_azimuths", type=float, nargs="*", default=[0.0])

    return parser


def configure_run(opt):
    """Apply DreamAssemble-specific defaults and post-process the merged config."""

    # By default, DreamAssemble uses Perp-Neg.
    opt.perpneg = True

    # Decompose semicolon-separated sub-prompts into a list, e.g.
    # "A boy; A girl; A tiger" -> ["A boy", "A girl", "A tiger"].
    opt.part_texts = [p.strip() for p in opt.part_texts.split(";")]
    opt.part_nums = len(opt.part_texts)

    # If the user didn't provide a --negative, fall back to "".
    if "negative" not in opt or opt.negative is None:
        opt.negative = ""

    # DeepFloyd-IF replaces SD if the config asks for it.
    if getattr(opt, "IF", False):
        if "SD" in opt.guidance:
            opt.guidance = [g for g in opt.guidance if g != "SD"]
        if "IF" not in opt.guidance:
            opt.guidance.append("IF")
        opt.latent_iter_ratio = 0  # cannot use as_latent with IF

    # Suffix workspace with _perpneg to make multi-run logs distinguishable.
    if opt.perpneg:
        opt.workspace = opt.workspace + "_perpneg"

    # Resolve workspace path: results/<name>  or  results_dmtet/<name>.
    if opt.dmtet:
        if not opt.init_with or not opt.init_with.endswith("df.pth"):
            opt.init_with = os.path.join("results", opt.workspace, "checkpoints", "df.pth")
        opt.workspace = os.path.join("results_dmtet", opt.workspace)
    else:
        opt.workspace = os.path.join("results", opt.workspace)

    # Image-conditioned generation is not part of the public DreamAssemble release.
    opt.images = None
    opt.image = None
    opt.image_config = None

    # Progressive view / level: keep user choice but back up full ranges if enabled.
    if opt.progressive_view:
        opt.full_radius_range = list(opt.radius_range)
        opt.full_theta_range = list(opt.theta_range)
        opt.full_phi_range = list(opt.phi_range)
        opt.full_fovy_range = list(opt.fovy_range)
        if not opt.dont_override_stuff:
            opt.jitter_pose = False
        opt.uniform_sphere_rate = 0

    # Experiment iteration window for progressive scheduling.
    opt.exp_start_iter = opt.exp_start_iter or 0
    opt.exp_end_iter = opt.exp_end_iter or opt.iters

    # DMTet defaults: high-resolution rendering and lower noise levels.
    if opt.dmtet:
        opt.h = 512
        opt.w = 512
        opt.known_view_scale = 1
        if not opt.dont_override_stuff:
            opt.t_range = [0.02, 0.50]  # ref: Magic3D
            opt.lambda_normal = 0
            opt.lambda_depth = 0
        opt.latent_iter_ratio = 0
        opt.albedo_iter_ratio = 0
        opt.progressive_view = False

    return opt


def build_model(opt, device):
    """Instantiate the multi-density NeRF backbone."""
    if opt.backbone == "grid_mask":
        from nerf.network import NeRFNetwork
    else:
        raise NotImplementedError(f"backbone {opt.backbone} is not supported.")

    model = NeRFNetwork(opt).to(device)

    # Initialize DMTet from a coarse-stage checkpoint (or a mesh).
    if opt.dmtet and opt.init_with:
        if opt.init_with.endswith(".pth"):
            state_dict = torch.load(opt.init_with, map_location=device)
            model.load_state_dict(state_dict["model"], strict=False)
            if opt.cuda_ray and "mean_density" in state_dict:
                model.mean_density = state_dict["mean_density"]
            model.init_tet()
        else:
            import trimesh
            mesh = trimesh.load(opt.init_with, force="mesh", skip_material=True, process=False)
            model.init_tet(mesh=mesh)

    return model


def build_optimizer(opt):
    if opt.optim == "adan":
        from optimizer import Adan
        return lambda m: Adan(
            m.get_params(5 * opt.lr),
            eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False,
        )
    return lambda m: torch.optim.Adam(m.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)


def build_guidance(opt, device):
    guidance = nn.ModuleDict()
    if "SD" in opt.guidance:
        from guidance.stable_diffusion import StableDiffusion
        guidance["SD"] = StableDiffusion(
            device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key, opt.t_range,
        )
    if "IF" in opt.guidance:
        from guidance.deep_floyd_if import DeepFloydIF
        guidance["IF"] = DeepFloydIF(device, opt.vram_O, opt.t_range)
    return guidance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_arg_parser()
    args, _unknown = parser.parse_known_args()

    # Merge yaml config with CLI args (CLI wins).
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.create(vars(args)))
    opt = configure_run(opt)

    set_seed(int(opt.seed))
    seed_everything(int(opt.seed))

    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(opt, device)
    print(model)

    # ---- inference modes ----
    if opt.six_views or opt.test:
        trainer = Trainer(
            " ".join(sys.argv), "df", opt, model, guidance=None,
            device=device, workspace=opt.workspace,
            fp16=opt.fp16, use_checkpoint=opt.ckpt,
        )

        view_type = "six_views" if opt.six_views else "test"
        size = 6 if opt.six_views else opt.dataset_size_test
        loader = NeRFDataset(opt, device=device, type=view_type,
                             H=opt.H, W=opt.W, size=size).dataloader(batch_size=1)
        trainer.test(loader, write_video=not opt.six_views)

        if opt.save_mesh:
            trainer.save_mesh()
        return

    # ---- training mode ----
    # Fresh-start: wipe previous workspace.
    if os.path.exists(opt.workspace):
        shutil.rmtree(opt.workspace)

    optimizer = build_optimizer(opt)
    scheduler = lambda o: optim.lr_scheduler.LambdaLR(o, lambda step: 1)
    guidance = build_guidance(opt, device)

    trainer = Trainer(
        " ".join(sys.argv), "df", opt, model, guidance,
        device=device, workspace=opt.workspace,
        optimizer=optimizer, lr_scheduler=scheduler,
        ema_decay=0.95, fp16=opt.fp16,
        use_checkpoint=opt.ckpt, scheduler_update_every_step=True,
    )

    train_loader = NeRFDataset(
        opt, device=device, type="train",
        H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size,
    ).dataloader()

    valid_loader = NeRFDataset(
        opt, device=device, type="val",
        H=opt.H, W=opt.W, size=opt.dataset_size_valid,
    ).dataloader(batch_size=1)

    test_loader = NeRFDataset(
        opt, device=device, type="test",
        H=opt.H, W=opt.W, size=opt.dataset_size_test,
    ).dataloader(batch_size=1)

    trainer.default_view_data = train_loader._data.get_default_view_data()

    max_epoch = int(np.ceil(opt.iters / len(train_loader)))
    trainer.train(train_loader, valid_loader, test_loader, max_epoch)

    if opt.save_mesh:
        trainer.save_mesh()


if __name__ == "__main__":
    main()
