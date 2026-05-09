"""Stable Diffusion guidance for distillation sampling (SDS / Perp-Neg)."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision.utils import save_image
from transformers import logging as hf_logging

from .perp_neg import weighted_perpendicular_aggregator

hf_logging.set_verbosity_error()


# Default HuggingFace keys per supported SD version.
DEFAULT_HF_KEYS = {
    '2.1': 'stabilityai/stable-diffusion-2-1-base',
    '1.5': 'runwayml/stable-diffusion-v1-5',
}


class _SpecifyGradient(torch.autograd.Function):
    """Detached gradient pass-through used by SDS.

    Forward returns ``1`` so AMP's grad-scaler scales the gradient correctly,
    backward injects the precomputed ``gt_grad``.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        return gt_grad * grad_scale, None


class StableDiffusion(nn.Module):
    """Stable Diffusion text-to-2D guidance with SDS and Perp-Neg samplers."""

    def __init__(self, device, fp16, vram_O,
                 sd_version='2.1', hf_key=None, t_range=(0.02, 0.98)):
        super().__init__()
        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f'[INFO] Using custom Stable Diffusion key: {hf_key}')
            model_key = hf_key
        elif sd_version in DEFAULT_HF_KEYS:
            model_key = DEFAULT_HF_KEYS[sd_version]
        else:
            raise ValueError(f'Stable Diffusion version {sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32
        print(f'[INFO] loading Stable Diffusion ({model_key}) ...')

        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.precision_t, safety_checker=None,
        )
        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder='scheduler', torch_dtype=self.precision_t,
        )
        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        print('[INFO] loaded Stable Diffusion!')

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
        )
        return self.text_encoder(inputs.input_ids.to(self.device))[0]

    def encode_imgs(self, imgs):
        """Encode an RGB image tensor in [0, 1] into a VAE latent."""
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        return posterior.sample() * self.vae.config.scaling_factor

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        return (imgs / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------
    # SDS train step
    # ------------------------------------------------------------------

    def train_step(self, text_embeddings, pred_rgb,
                   guidance_scale=100, as_latent=False, grad_scale=1.0,
                   exp_iter_ratio=0, dmtet=False, **kwargs):
        """Standard Score Distillation Sampling step."""
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear',
                                    align_corners=False) * 2 - 1
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear',
                                         align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)

        if dmtet:
            t = torch.randint(20, 500, (1,), dtype=torch.long, device=self.device)
        else:
            t = torch.randint(100, 900, (1,), dtype=torch.long, device=self.device)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # Stack [uncond, cond] for each latent in the batch.
            latent_model_input = torch.cat(
                [torch.cat([latents_noisy[i:i + 1]] * 2) for i in range(latents.shape[0])]
            )
            tt = torch.cat([t] * 2 * latents.shape[0])
            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=text_embeddings,
            ).sample

            preds = []
            for chunk in noise_pred.chunk(latents.shape[0]):
                noise_pred_uncond, noise_pred_cond = chunk.chunk(2)
                preds.append(
                    noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                )
            noise_pred = torch.cat(preds, dim=0)

        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        return _SpecifyGradient.apply(latents, grad)

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb,
                           guidance_scale=100, as_latent=False, grad_scale=1.0,
                           save_guidance_path: Path = None):
        """Perp-Neg variant of the SDS step."""
        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear',
                                    align_corners=False) * 2 - 1
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear',
                                         align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)

        t = torch.randint(self.min_step, self.max_step + 1,
                          (latents.shape[0],), dtype=torch.long, device=self.device)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * (1 + K))
            tt = torch.cat([t] * (1 + K))
            unet_output = self.unet(
                latent_model_input, tt, encoder_hidden_states=text_embeddings,
            ).sample

            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(
                delta_noise_preds, weights, B,
            )

        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            self._save_guidance(latents, latents_noisy, noise_pred, t, save_guidance_path,
                                pred_rgb_512=pred_rgb_512 if not as_latent else None,
                                as_latent=as_latent)

        return _SpecifyGradient.apply(latents, grad)

    @torch.no_grad()
    def _save_guidance(self, latents, latents_noisy, noise_pred, t,
                       save_path, pred_rgb_512=None, as_latent=False):
        """Save (input | noisy | denoised) triptych for debugging."""
        if as_latent or pred_rgb_512 is None:
            pred_rgb_512 = self.decode_latents(latents)

        alphas = self.scheduler.alphas.to(latents)
        total_t = self.max_step - self.min_step + 1
        index = total_t - t.to(latents.device) - 1
        b = noise_pred.shape[0]
        a_t = alphas[index].reshape(b, 1, 1, 1).to(self.device)
        sqrt_one_minus = torch.sqrt(1 - alphas)[index].reshape(b, 1, 1, 1).to(self.device)
        pred_x0 = (latents_noisy - sqrt_one_minus * noise_pred) / a_t.sqrt()

        denoised = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))
        noisier = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))
        save_image(torch.cat([pred_rgb_512, noisier, denoised], dim=0), save_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512,
                        num_inference_steps=50, guidance_scale=7.5, latents=None):
        if latents is None:
            latents = torch.randn(
                (text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                device=self.device,
            )
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings,
            )['sample']
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512,
                      num_inference_steps=50, guidance_scale=7.5):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        pos = self.get_text_embeds(prompts)
        neg = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg, pos], dim=0)
        latents = self.produce_latents(
            text_embeds, height=height, width=width,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        )
        imgs = self.decode_latents(latents)
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        return (imgs * 255).round().astype('uint8')
