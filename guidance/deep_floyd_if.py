"""DeepFloyd IF text-to-image guidance for distillation sampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import IFPipeline
from torch.cuda.amp import custom_bwd, custom_fwd
from transformers import logging as hf_logging

from .perp_neg import weighted_perpendicular_aggregator

hf_logging.set_verbosity_error()

DEFAULT_IF_KEY = 'DeepFloyd/IF-I-XL-v1.0'


class _SpecifyGradient(torch.autograd.Function):
    """Detached gradient pass-through (see ``stable_diffusion.py``)."""

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


class DeepFloydIF(nn.Module):
    """DeepFloyd IF-I-XL guidance model with SDS and Perp-Neg samplers.

    IF operates directly in pixel space at ``64x64`` resolution.
    """

    def __init__(self, device, vram_O, t_range=(0.02, 0.98), hf_key=None):
        super().__init__()
        self.device = device
        model_key = hf_key or DEFAULT_IF_KEY

        print(f'[INFO] loading DeepFloyd IF ({model_key}) ...')
        is_torch2 = torch.__version__[0] == '2'

        pipe = IFPipeline.from_pretrained(
            model_key, variant='fp16', torch_dtype=torch.float16, device_map='balanced',
        )
        if not is_torch2:
            pipe.enable_xformers_memory_efficient_attention()
        if vram_O:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        self.unet = pipe.unet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.scheduler = pipe.scheduler
        self.pipe = pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        print('[INFO] loaded DeepFloyd IF!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        prompt = self.pipe._text_preprocessing(prompt, clean_caption=False)
        inputs = self.tokenizer(
            prompt, padding='max_length', max_length=77,
            truncation=True, add_special_tokens=True, return_tensors='pt',
        )
        return self.text_encoder(inputs.input_ids.to(self.device))[0]

    # ------------------------------------------------------------------
    # SDS train step
    # ------------------------------------------------------------------

    def train_step(self, text_embeddings, pred_rgb,
                   guidance_scale=100, grad_scale=1.0,
                   exp_iter_ratio=0, dmtet=False, **kwargs):
        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

        if dmtet:
            t = torch.randint(20, 500, (1,), dtype=torch.long, device=self.device)
        else:
            t = torch.randint(100, 900, (1,), dtype=torch.long, device=self.device)

        with torch.no_grad():
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)
            model_input = torch.cat(
                [torch.cat([images_noisy[i:i + 1]] * 2) for i in range(images.shape[0])]
            )
            model_input = self.scheduler.scale_model_input(model_input, torch.cat([t] * 2))
            tt = torch.cat([t] * 2 * images.shape[0])
            noise_pred = self.unet(
                model_input, tt, encoder_hidden_states=text_embeddings,
            ).sample

            preds = []
            for chunk in noise_pred.chunk(images.shape[0]):
                noise_pred_uncond, noise_pred_text = chunk.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred_text, _ = noise_pred_text.split(model_input.shape[1], dim=1)
                preds.append(
                    noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                )
            noise_pred = torch.cat(preds, dim=0)

        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        return _SpecifyGradient.apply(images, grad)

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb,
                           guidance_scale=100, grad_scale=1.0):
        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1

        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        t = torch.randint(self.min_step, self.max_step + 1,
                          (images.shape[0],), dtype=torch.long, device=self.device)

        with torch.no_grad():
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)
            model_input = torch.cat([images_noisy] * (1 + K))
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * (1 + K))

            unet_output = self.unet(
                model_input, tt, encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, _ = noise_pred_text.split(model_input.shape[1], dim=1)

            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(
                delta_noise_preds, weights, B,
            )

        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        return _SpecifyGradient.apply(images, grad)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def produce_imgs(self, text_embeddings, height=64, width=64,
                     num_inference_steps=50, guidance_scale=7.5):
        images = torch.randn(
            (1, 3, height, width),
            device=text_embeddings.device, dtype=text_embeddings.dtype,
        )
        images = images * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            model_input = torch.cat([images] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)
            noise_pred = self.unet(
                model_input, t, encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(
                model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
            images = self.scheduler.step(noise_pred, t, images).prev_sample

        return (images + 1) / 2

    def prompt_to_img(self, prompts, negative_prompts='', height=64, width=64,
                      num_inference_steps=50, guidance_scale=7.5):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        pos = self.get_text_embeds(prompts)
        neg = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg, pos], dim=0)
        imgs = self.produce_imgs(text_embeds, height=height, width=width,
                                 num_inference_steps=num_inference_steps,
                                 guidance_scale=guidance_scale)
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        return (imgs * 255).round().astype('uint8')
