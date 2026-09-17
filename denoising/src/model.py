"""Residual mapper operating on the frozen Geo-VAE's 16-channel latents."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResidualLatentRefiner(nn.Module):
    """Zero-initialized output starts at identity in latent space.

    Parameter names and shapes match the released denoise_latent.pt weights.
    """

    def __init__(self, channels=16, width=96, depth=8, checkpointing=False):
        super().__init__()
        if channels != 16:
            raise ValueError('Geo-VAE denoising requires 16 latent channels')
        self.checkpointing = checkpointing
        self.inp = nn.Conv3d(channels, width, 3, padding=1)
        blocks = []
        for _ in range(depth):
            blocks.extend([nn.GroupNorm(8, width), nn.SiLU(),
                           nn.Conv3d(width, width, 3, padding=1)])
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv3d(width, channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.inp(x)
        for i in range(0, len(self.blocks), 3):
            def residual(value, index=i):
                return self.blocks[index + 2](self.blocks[index + 1](
                    self.blocks[index](value)))

            if self.checkpointing and self.training and torch.is_grad_enabled():
                update = checkpoint(residual, h, use_reentrant=False)
            else:
                update = residual(h)
            h = h + update
        return x + self.out(h)


def build_mapper(config):
    return ResidualLatentRefiner(
        channels=int(config.get('latent_channels', 16)),
        width=int(config.get('width', 96)),
        depth=int(config.get('depth', 8)),
        checkpointing=bool(config.get('gradient_checkpointing', False)),
    )
