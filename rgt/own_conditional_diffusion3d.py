"""Self-contained 3-D conditional DDPM used for FX+RGT -> seismic synthesis."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        frequency = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        phase = t[:, None].float() * frequency[None]
        return torch.cat((phase.sin(), phase.cos()), dim=1)


class FiLMResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, condition_channels=None, groups=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, 2 * out_channels))
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.condition = nn.Conv3d(condition_channels, out_channels, 1) if condition_channels is not None else None
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time, condition=None):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(time)[:, :, None, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        if self.condition is not None:
            if condition is None:
                raise ValueError("condition feature is required")
            h = h + self.condition(condition)
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class BottleneckAttention3D(nn.Module):
    """Attention only at 9^3 latent resolution; never at the 34^3 input grid."""
    def __init__(self, channels, heads=8):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must divide heads")
        self.heads, self.width = heads, channels // heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        n = d * h * w
        q = q.reshape(b, self.heads, self.width, n).transpose(-1, -2)
        k = k.reshape(b, self.heads, self.width, n)
        v = v.reshape(b, self.heads, self.width, n).transpose(-1, -2)
        # Autocast leaves Q/K in fp16.  As the denoiser sharpens, the 3-D
        # dot products can overflow before softmax; score/softmax must remain
        # fp32 even though convolutions retain mixed precision.
        weights = (q.float() @ k.float() / math.sqrt(self.width)).softmax(dim=-1).to(v.dtype)
        output = (weights @ v).transpose(-1, -2).reshape(b, c, d, h, w)
        return x + self.proj(output)


class ConditionalUNet3D(nn.Module):
    """Multi-scale 3-D conditional U-Net for z_sx given geological latents.

    The GeoVAE condition latent is projected to a three-level pyramid and is
    injected into every residual block at its native resolution.  Thus sparse
    fault information remains accessible during high-resolution decoding.
    """
    in_channels = 16

    def __init__(self, width=192, blocks_per_scale=4, condition_channels=32, dropout=0.0):
        super().__init__()
        self.time = nn.Sequential(TimeEmbedding(width), nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, 4 * width))
        time_channels = 4 * width
        w0, w1, w2 = width, width * 2, width * 2
        # Noisy SX is encoded separately; either one 16-channel GeoVAE
        # condition (FX or RGT) or the 32-channel concatenated condition is
        # projected and re-injected at every scale.
        self.input = nn.Conv3d(16, w0, 3, padding=1)
        self.condition0 = nn.Sequential(nn.Conv3d(condition_channels, w0, 3, padding=1), nn.SiLU(), nn.Conv3d(w0, w0, 3, padding=1))
        self.condition_down0 = nn.Conv3d(w0, w1, 4, stride=2, padding=1)
        self.condition_down1 = nn.Conv3d(w1, w2, 4, stride=2, padding=1)
        self.enc0 = nn.ModuleList([FiLMResBlock3D(w0, w0, time_channels, condition_channels=w0, dropout=dropout) for _ in range(blocks_per_scale)])
        self.down0 = nn.Conv3d(w0, w1, 4, stride=2, padding=1)
        self.enc1 = nn.ModuleList([FiLMResBlock3D(w1, w1, time_channels, condition_channels=w1, dropout=dropout) for _ in range(blocks_per_scale)])
        self.down1 = nn.Conv3d(w1, w2, 4, stride=2, padding=1)
        self.mid1 = FiLMResBlock3D(w2, w2, time_channels, condition_channels=w2, dropout=dropout)
        self.mid_attn = BottleneckAttention3D(w2)
        self.mid2 = FiLMResBlock3D(w2, w2, time_channels, condition_channels=w2, dropout=dropout)
        self.dec1 = nn.ModuleList([FiLMResBlock3D(w2 + w1 if i == 0 else w1, w1, time_channels, condition_channels=w1, dropout=dropout) for i in range(blocks_per_scale)])
        self.up1 = nn.Conv3d(w2, w1, 3, padding=1)
        self.dec0 = nn.ModuleList([FiLMResBlock3D(w0 + w0 if i == 0 else w0, w0, time_channels, condition_channels=w0, dropout=dropout) for i in range(blocks_per_scale)])
        self.up0 = nn.Conv3d(w1, w0, 3, padding=1)
        self.output = nn.Sequential(nn.GroupNorm(8, w0), nn.SiLU(), nn.Conv3d(w0, 16, 3, padding=1))

    def forward(self, noisy, timestep, condition):
        time = self.time(timestep)
        condition0 = self.condition0(condition)
        condition1 = self.condition_down0(condition0)
        condition2 = self.condition_down1(condition1)
        x = self.input(noisy)
        for block in self.enc0:
            x = block(x, time, condition0)
        skip0 = x
        x = self.down0(x)
        for block in self.enc1:
            x = block(x, time, condition1)
        skip1 = x
        x = self.down1(x)
        x = self.mid2(self.mid_attn(self.mid1(x, time, condition2)), time, condition2)
        x = F.interpolate(x, size=skip1.shape[-3:], mode="nearest")
        x = self.up1(x)
        x = torch.cat((x, skip1), dim=1)
        for block in self.dec1:
            x = block(x, time, condition1)
        x = F.interpolate(x, size=skip0.shape[-3:], mode="nearest")
        x = self.up0(x)
        x = torch.cat((x, skip0), dim=1)
        for block in self.dec0:
            x = block(x, time, condition0)
        return self.output(x)


class ConditionalDDPM(nn.Module):
    def __init__(self, model, timesteps=1000, prediction_type="epsilon"):
        super().__init__()
        self.model, self.timesteps = model, int(timesteps)
        if prediction_type not in {"epsilon", "v"}:
            raise ValueError(f"unsupported prediction type: {prediction_type}")
        self.prediction_type = prediction_type
        s = 0.008
        x = torch.linspace(0, self.timesteps, self.timesteps + 1)
        alpha_bar = torch.cos(((x / self.timesteps + s) / (1 + s)) * math.pi / 2).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-4, 0.9999)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alpha_bar.float())
        self.register_buffer("sqrt_alphas_cumprod", alpha_bar.sqrt().float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alpha_bar).sqrt().float())

    def _extract(self, values, time, shape):
        return values.gather(0, time).reshape(time.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, time, noise):
        return self._extract(self.sqrt_alphas_cumprod, time, x0.shape) * x0 + self._extract(self.sqrt_one_minus_alphas_cumprod, time, x0.shape) * noise

    def forward(self, x0, condition):
        time = torch.randint(self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        alpha = self._extract(self.sqrt_alphas_cumprod, time, x0.shape)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, time, x0.shape)
        predicted = self.model(alpha * x0 + sigma * noise, time, condition)
        target = noise if self.prediction_type == "epsilon" else alpha * noise - sigma * x0
        return F.mse_loss(predicted, target)
