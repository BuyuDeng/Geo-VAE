"""Shared denoising path: frozen VAE encoder -> latent mapper -> VAE decoder."""
from __future__ import annotations

from contextlib import nullcontext

import torch

from vae.wan_video_vae import WanVideoVAE
from .model import build_mapper
from .vae_io import from_vae, to_vae


def checkpoint_state(checkpoint):
    """Accept released tensor-only weights or a resumable training checkpoint."""
    return checkpoint['ema']['model'] if checkpoint.get('ema') else checkpoint.get('model', checkpoint)


def load_vae(checkpoint, device):
    vae = WanVideoVAE().to(device)
    vae.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return vae.eval().requires_grad_(False)


def load_models(config, checkpoint, device):
    vae = load_vae(config['vae_checkpoint'], device)
    mapper = build_mapper(config).to(device)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    mapper.load_state_dict(checkpoint_state(state), strict=True)
    return vae, mapper.eval(), int(state.get('epoch', -1))


def _tiling(config):
    return dict(tiled=bool(config.get('tiled', False)),
                tile_size=tuple(config.get('tile_size', (34, 34))),
                tile_stride=tuple(config.get('tile_stride', (18, 16))))


def forward_denoise(vae, mapper, normalized, device, encode_config=None,
                    decode_config=None):
    """Encode once, map the complete latent once, and decode once.

    VAE weights are frozen by load_vae. Only encoding disables autograd:
    decoded-domain training losses must still reach the mapper through decoding.
    Internal VAE tiling is independent of the single mapper call.
    """
    device = torch.device(device)
    decode_config = decode_config or {}
    with torch.no_grad():
        latent = vae.encode(to_vae(normalized), device=device,
                            **_tiling(encode_config or {}))
    corrected = mapper(latent.to(device))
    if decode_config.get('tiled', False):
        corrected = corrected.cpu()
    offload = bool(decode_config.get('activation_offload', False)) and torch.is_grad_enabled()
    context = torch.autograd.graph.save_on_cpu(pin_memory=device.type == 'cuda') if offload else nullcontext()
    with context:
        decoded = vae.decode(corrected, device=device, **_tiling(decode_config))
    return from_vae(decoded, normalized.shape[-3:]).to(normalized.device).float()


@torch.inference_mode()
def predict_normalized(volume, vae, mapper, device, config):
    """Denoise a normalized NumPy volume in time,crossline,inline order."""
    device = torch.device(device)
    value = torch.from_numpy(volume).unsqueeze(0)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        prediction = forward_denoise(
            vae, mapper, value, device,
            encode_config=config.get('vae_encode'),
            decode_config=config.get('vae_decode'),
        )
    return prediction[0].cpu().numpy()
