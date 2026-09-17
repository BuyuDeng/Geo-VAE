"""Noisy-input amplitude normalization and Geo-VAE axis conversion."""
from __future__ import annotations

import torch.nn.functional as F


def normalize_amplitude(x, centre, scale):
    return ((x - centre) / scale).clamp(-1, 1)


def robust_normalize(x):
    """Historical name: use each noisy cube's mean and population std."""
    flat = x.reshape(x.shape[0], -1)
    centre = flat.mean(1)[:, None, None, None]
    scale = (flat.std(1, correction=0)[:, None, None, None] + 1e-6) * 3.2
    return normalize_amplitude(x, centre, scale), centre, scale


def denormalize(x, centre, scale):
    """Restore amplitude units; normalization clipping remains lossy."""
    return x * scale + centre


def to_vae(x):
    """B,time,crossline,inline -> B,RGB,crossline,inline,time."""
    from vae.inference import causal_resize
    if x.ndim != 4:
        raise ValueError('expected B,time,crossline,inline')
    return causal_resize(x.permute(0, 2, 3, 1)[:, None].repeat(1, 3, 1, 1, 1).contiguous())


def from_vae(x, target_shape=None):
    y = x.mean(1).permute(0, 3, 1, 2).contiguous()
    if target_shape is not None and tuple(y.shape[-3:]) != tuple(target_shape):
        y = F.interpolate(y[:, None], size=target_shape,
                          mode='trilinear', align_corners=False)[:, 0]
    return y
