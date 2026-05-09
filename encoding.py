"""Positional / hash-grid encoders used by the NeRF backbone."""

import torch
import torch.nn as nn


class FrequencyEncoder(nn.Module):
    """Plain PyTorch implementation of NeRF-style frequency encoding."""

    def __init__(self, input_dim, max_freq_log2, num_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        self.num_freqs = num_freqs

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim
        self.output_dim += self.input_dim * num_freqs * len(self.periodic_fns)

        if log_sampling:
            freq_bands = 2 ** torch.linspace(0, max_freq_log2, num_freqs)
        else:
            freq_bands = torch.linspace(2 ** 0, 2 ** max_freq_log2, num_freqs)
        self.freq_bands = freq_bands.numpy().tolist()

    def forward(self, x, max_level=None, **kwargs):
        if max_level is None:
            max_level = self.num_freqs
        else:
            max_level = int(max_level * self.num_freqs)

        out = []
        if self.include_input:
            out.append(x)

        for i in range(max_level):
            freq = self.freq_bands[i]
            for fn in self.periodic_fns:
                out.append(fn(x * freq))

        # zero-pad the disabled levels so output_dim stays constant.
        if self.num_freqs - max_level > 0:
            pad_dim = (self.num_freqs - max_level) * 2 * x.shape[-1]
            out.append(torch.zeros(*x.shape[:-1], pad_dim, device=x.device, dtype=x.dtype))

        return torch.cat(out, dim=-1)


def get_encoder(encoding,
                input_dim=3,
                multires=6,
                degree=4,
                num_levels=16,
                level_dim=2,
                base_resolution=16,
                log2_hashmap_size=19,
                desired_resolution=2048,
                align_corners=False,
                interpolation="linear",
                **kwargs):
    """Factory dispatch for the supported encoders.

    DreamAssemble only relies on ``hashgrid`` for the spatial encoder and
    ``frequency`` for the background MLP.
    """

    if encoding == "None":
        return lambda x, **kw: x, input_dim

    if encoding == "frequency_torch":
        encoder = FrequencyEncoder(
            input_dim=input_dim, max_freq_log2=multires - 1,
            num_freqs=multires, log_sampling=True,
        )
    elif encoding == "frequency":
        from freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)
    elif encoding == "hashgrid":
        from gridencoder import GridEncoder
        encoder = GridEncoder(
            input_dim=input_dim, num_levels=num_levels, level_dim=level_dim,
            base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution, gridtype="hash",
            align_corners=align_corners, interpolation=interpolation,
        )
    elif encoding == "tiledgrid":
        from gridencoder import GridEncoder
        encoder = GridEncoder(
            input_dim=input_dim, num_levels=num_levels, level_dim=level_dim,
            base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution, gridtype="tiled",
            align_corners=align_corners, interpolation=interpolation,
        )
    else:
        raise NotImplementedError(
            f"Unknown encoding mode: {encoding}. Choose from "
            "[None, frequency, frequency_torch, hashgrid, tiledgrid]."
        )

    return encoder, encoder.output_dim
