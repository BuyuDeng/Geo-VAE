"""Existing Lightning trainer, restored to the causal paper configuration."""

import os
import time
import uuid
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

try:
    from .model import SeismicVAE3D
except ImportError:  # 作为脚本直接运行时(无父包)
    from model import SeismicVAE3D


# ============================================================================
# Discriminator
# ============================================================================

class Discriminator3D(nn.Module):
    """3D PatchGAN Discriminator."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = [
            nn.Conv3d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base_channels
        for i in range(1, num_layers):
            out_channels = min(channels * 2, 512)
            layers.extend([
                nn.Conv3d(channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            channels = out_channels

        layers.append(nn.Conv3d(channels, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AdversarialLoss(nn.Module):
    """相对论GAN损失 (Relativistic average GAN)

    'rahinge' 与 orgvae 的 compute_generator_loss / compute_discriminator_loss
    逐字等价。
    """

    def __init__(self, type='ralsgan'):
        super().__init__()
        self.type = type.lower()

    def __call__(self, d_real, d_fake, is_disc=True):
        """
        d_real: 判别器对真实样本输出 (D(real))
        d_fake: 判别器对生成样本输出 (D(fake))
        is_disc: 是否是用于训练判别器阶段
        """
        if self.type == 'rasgan':
            if is_disc:
                adv_loss = (F.binary_cross_entropy_with_logits(d_real - d_fake.mean(), torch.ones_like(d_real)) +
                            F.binary_cross_entropy_with_logits(d_fake - d_real.mean(), torch.zeros_like(d_fake))) / 2
            else:
                adv_loss = (F.binary_cross_entropy_with_logits(d_real - d_fake.mean(), torch.zeros_like(d_real)) +
                            F.binary_cross_entropy_with_logits(d_fake - d_real.mean(), torch.ones_like(d_fake))) / 2
            return adv_loss

        elif self.type == 'ralsgan':
            if is_disc:
                adv_loss = (((d_real - d_fake.mean() - 1) ** 2).mean() +
                            ((d_fake - d_real.mean() + 1) ** 2).mean()) / 2
            else:
                adv_loss = (((d_real - d_fake.mean() + 1) ** 2).mean() +
                            ((d_fake - d_real.mean() - 1) ** 2).mean()) / 2
            return adv_loss

        elif self.type == 'rahinge':
            if is_disc:
                adv_loss = (F.relu(1.0 - (d_real - d_fake.mean())).mean() +
                            F.relu(1.0 + (d_fake - d_real.mean())).mean()) / 2
            else:
                adv_loss = (F.relu(1.0 + (d_real - d_fake.mean())).mean() +
                            F.relu(1.0 - (d_fake - d_real.mean())).mean()) / 2
            return adv_loss
        else:
            raise NotImplementedError(f"Unsupported loss type: {self.type}")


# ============================================================================
# Perceptual Loss using pretrained networks
# ============================================================================

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features with random slice sampling."""

    def __init__(self, device: str = "cuda", k_slices: int = 16):
        super().__init__()
        self.k_slices = k_slices

        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except ImportError:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features

        # Extract features at different layers
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),   # relu1_2
            vgg[4:9].eval(),  # relu2_2
            vgg[9:18].eval(), # relu3_4
            vgg[18:27].eval() # relu4_4
        ])

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.weights = [1.0, 1.0, 1.0, 1.0]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet normalization."""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        return (x - self.mean) / self.std

    def compute_vgg_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute VGG perceptual loss for 2D slices."""
        x = self.normalize(x)
        y = self.normalize(y)

        loss = 0.0
        for block, weight in zip(self.blocks, self.weights):
            x = block(x)
            y = block(y)
            loss += weight * F.l1_loss(x, y)

        return loss

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between x and y using random slices from 3 directions.

        Args:
            x: Reconstructed cube [B, C, T, H, W]
            y: Original cube [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        k = self.k_slices

        losses = []

        # Direction 1: Depth slices (T direction) -> [B, C, H, W] frames
        t_indices = torch.randperm(T, device=x.device)[:min(k, T)]
        x_t = x[:, :, t_indices]  # [B, C, k, H, W]
        y_t = y[:, :, t_indices]
        x_t = rearrange(x_t, "b c k h w -> (b k) c h w")
        y_t = rearrange(y_t, "b c k h w -> (b k) c h w")
        losses.append(self.compute_vgg_loss(x_t, y_t))

        # Direction 2: Height slices (H direction) -> [B, C, T, W] slices
        h_indices = torch.randperm(H, device=x.device)[:min(k, H)]
        x_h = x[:, :, :, h_indices, :]  # [B, C, T, k, W]
        y_h = y[:, :, :, h_indices, :]
        x_h = rearrange(x_h, "b c t k w -> (b k) c t w")
        y_h = rearrange(y_h, "b c t k w -> (b k) c t w")
        losses.append(self.compute_vgg_loss(x_h, y_h))

        # Direction 3: Width slices (W direction) -> [B, C, T, H] slices
        w_indices = torch.randperm(W, device=x.device)[:min(k, W)]
        x_w = x[:, :, :, :, w_indices]  # [B, C, T, H, k]
        y_w = y[:, :, :, :, w_indices]
        x_w = rearrange(x_w, "b c t h k -> (b k) c t h")
        y_w = rearrange(y_w, "b c t h k -> (b k) c t h")
        losses.append(self.compute_vgg_loss(x_w, y_w))

        # Average loss across all three directions
        return sum(losses) / len(losses)


class LinearDecayPositionEmbedding(nn.Module):
    """
    线性递减三通道正余弦位置编码 (移植自 framework.py)。

    输入: rgt 体 (B, 1, T, H, W)，值域 [-1, 1]
    输出: 位置编码 (B, 3, T, H, W)，由三种频率的 sin/cos 组成
    频率: 2.0, 1.0, 0.5（线性递减 2 倍）
    用途: 把 RGT 的连续 scalar 投影到一个区分度更高的多通道空间，
          再在该空间上算回归与感知损失。
    """

    def __init__(self, frequencies=(2.0, 1.0, 0.5), discret: int = 128):
        super().__init__()
        self.frequencies = list(frequencies)
        self.discret = discret
        self.num_channels = len(self.frequencies)

    def forward(self, rgt_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgt_pos: (B, 1, T, H, W)，值域 [-1, 1]
        Returns:
            (B, 3, T, H, W)
        """
        b, _, t, h, w = rgt_pos.shape
        device = rgt_pos.device
        # [-1, 1] -> [0, 1] -> [1, discret]
        normalized_pos = (rgt_pos + 1) / 2
        scaled_pos = normalized_pos * (self.discret - 1) + 1
        scaled_pos = scaled_pos.squeeze(1)  # (B, T, H, W)

        pos_embedding = torch.zeros(b, self.num_channels, t, h, w,
                                    device=device, dtype=rgt_pos.dtype)
        for c, freq in enumerate(self.frequencies):
            angles = scaled_pos * freq
            if c % 2 == 0:
                pos_embedding[:, c] = torch.sin(angles)
            else:
                pos_embedding[:, c] = torch.cos(angles)
        return pos_embedding


class LPIPSLoss(nn.Module):
    """
    3D 感知损失 (LPIPS)：
    输入：cube1, cube2 的形状为 (N, C, D, H, W)
    在 D、H、W 三个维度上各随机抽取 k 个切片，以 2D LPIPS 计算感知差异。
    """

    def __init__(self, net: str = 'vgg', k: int = 20):
        """
        Args:
            net: 使用的 LPIPS 后端网络，可选 'alex', 'vgg', 'squeeze'
            k: 每个维度抽取的切片数
        """
        super().__init__()
        self.lpips_fun = lpips.LPIPS(net=net).eval().requires_grad_(False)
        self.k = k

    def forward(self, cube1: torch.Tensor, cube2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cube1, cube2: 两个 3D 体数据，shape = (N, C, D, H, W)
        Returns:
            平均的 3D 感知损失标量
        """
        N, C, D, H, W = cube1.shape
        device = cube1.device
        # —— 深度方向切片 (shape: N*k, C, H, W)
        idx_d = torch.randperm(D, device=device)[:self.k]
        slices1_d = rearrange(cube1[:, :, idx_d, :, :], 'b c k h w -> (b k) c h w')
        slices2_d = rearrange(cube2[:, :, idx_d, :, :], 'b c k h w -> (b k) c h w')
        loss_d = self.lpips_fun(slices1_d, slices2_d).mean()
        # —— 高度方向切片 (shape: N*k, C, D, W)
        idx_h = torch.randperm(H, device=device)[:self.k]
        slices1_h = rearrange(cube1[:, :, :, idx_h, :], 'b c d k w -> (b k) c d w')
        slices2_h = rearrange(cube2[:, :, :, idx_h, :], 'b c d k w -> (b k) c d w')
        loss_h = self.lpips_fun(slices1_h, slices2_h).mean()
        # —— 宽度方向切片 (shape: N*k, C, D, H)
        idx_w = torch.randperm(W, device=device)[:self.k]
        slices1_w = rearrange(cube1[:, :, :, :, idx_w], 'b c d h k -> (b k) c d h')
        slices2_w = rearrange(cube2[:, :, :, :, idx_w], 'b c d h k -> (b k) c d h')
        loss_w = self.lpips_fun(slices1_w, slices2_w).mean()
        # 三个方向的损失再平均
        loss = (loss_d + loss_h + loss_w) / 3.0
        return loss.mean()


# ============================================================================
# Perceptual-loss checkpoint compatibility
# ============================================================================

def _rank_zero_print(*args, **kwargs):
    """DDP 下只让 rank0 打印（不 import train.py，避免循环依赖）。"""
    if (os.environ.get("LOCAL_RANK", "0") == "0"
            and os.environ.get("NODE_RANK", "0") == "0"
            and os.environ.get("RANK", "0") == "0"):
        print(*args, **kwargs)


# 感知损失（LPIPS / VGG19）用的是冻结的预训练权重，不参与训练。
# 把它们存进 ckpt 没有意义，而且会制造后端冲突：LPIPS 的 alex 后端有 22 个
# 张量、vgg 有 38 个，其中 9 个同名但形状不同（如 net.slice1.0.weight 是
# 11x11 vs 3x3）。这种情况下 load_state_dict(strict=False) 也会抛
# "size mismatch"——strict=False 只放过多/少的 key，不放过形状不符。
# 于是换后端后旧 ckpt 直接加载失败。
_PERCEPTUAL_PREFIX = "perceptual_loss."


def strip_perceptual_keys(state_dict):
    """从 state_dict 里移除感知损失权重（原地修改）。Returns: 移除的 key 数。"""
    stale = [k for k in state_dict if k.startswith(_PERCEPTUAL_PREFIX)]
    for k in stale:
        del state_dict[k]
    return len(stale)


def align_perceptual_keys(state_dict, model, log=None):
    """让 ckpt 里的感知损失权重与当前模型对齐，使旧 ckpt 能正常加载。

    做法：丢掉 ckpt 里的 perceptual_loss.*（可能来自别的后端、形状不符），
    再把当前模型自己的那份填回去。这样 key 集合与模型完全一致，
    主干权重仍可走严格校验，不需要放宽 strict。

    Args:
        state_dict: ckpt 的 state_dict，原地修改
        model: 当前 LightningModule
        log: 可选的打印函数
    """
    dropped = strip_perceptual_keys(state_dict)

    own = model.state_dict()
    restored = 0
    for k, v in own.items():
        if k.startswith(_PERCEPTUAL_PREFIX):
            state_dict[k] = v
            restored += 1

    # 只在真的丢弃了旧权重时报告（那说明 ckpt 来自别的后端）；
    # 新 ckpt 本就不含这些 key，静默补回即可，不必刷屏。
    if log is not None and dropped:
        log(f"[perceptual] ckpt 里有 {dropped} 个感知损失张量（可能来自别的后端），"
            f"已丢弃并改用当前模型的 {restored} 个冻结预训练权重")
    return dropped, restored


# ============================================================================
# Training VAE Model (Trainable version)
# ============================================================================

class TrainableSeismicVAE(nn.Module):
    """Trainable wrapper of SeismicVAE3D with proper KL divergence support.

    Cached causal encoder/decoder with posterior parameters for KL training.
    """

    def __init__(
            self,
            dim: int = 96,
            z_dim: int = 16,
            dim_mult: List[int] = [1, 2, 4, 4],
            num_res_blocks: int = 2,
            attn_scales: List[float] = [],
            temperal_downsample: List[bool] = [True, True, True],
            dropout: float = 0.0,
            grad_checkpoint: str = 'none',
    ):
        super().__init__()

        self.z_dim = z_dim
        self.temporal_factor = 2 ** sum(temperal_downsample)

        self.vae = SeismicVAE3D(
            dim=dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            grad_checkpoint=grad_checkpoint,
        )

        # Learnable scale parameters (optional)
        self.register_buffer("mean", torch.zeros(z_dim))
        self.register_buffer("std", torch.ones(z_dim))

    @property
    def scale(self):
        return [self.mean, 1.0 / self.std]

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode cube to latent distribution parameters."""
        from .inference import causal_resize
        return self.vae.encode(causal_resize(x, self.temporal_factor))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to cube."""
        return self.vae.decode(z)

    def reparameterize(
            self,
            mu: torch.Tensor,
            log_var: torch.Tensor,
            deterministic: bool = False,
    ) -> torch.Tensor:
        """Reparameterization trick."""
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
            self,
            x: torch.Tensor,
            deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input cube [B, C, T, H, W]
            deterministic: If True, don't add noise to latent

        Returns:
            recon: Reconstructed cube
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, deterministic)
        recon = self.decode(z)
        if recon.shape[-3:] != x.shape[-3:]:
            recon = F.interpolate(recon, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return recon, mu, log_var


# ============================================================================
# Lightning Module
# ============================================================================

class SeismicVAELightning(LightningModule):
    """PyTorch Lightning module for training SeismicVAE3D."""

    def __init__(
            self,
            # Model params
            dim: int = 96,
            z_dim: int = 16,
            dim_mult: List[int] = [1, 2, 4, 4],
            num_res_blocks: int = 2,
            attn_scales: List[float] = [],
            temperal_downsample: List[bool] = [True, True, True],
            dropout: float = 0.0,
            grad_checkpoint: str = 'shallow',  # 'none' | 'shallow' | 'full'
            # Loss params
            recon_loss_type: str = "l1",  # "l1", "l2", "mixed"
            recon_loss_weight: float = 1.0,
            kl_weight: float = 1e-4,
            perceptual_weight: float = 1.0,
            perceptual_type: str = "lpips",  # "vgg", "lpips"
            use_gan: bool = True,
            gan_weight: float = 0.5,
            gan_loss_type: str = "ralsgan",  # 'rahinge', 'ralsgan', 'rasgan'
            disc_start_step: int = 100000,
            disc_num_layers: int = 3,
            disc_channels: int = 64,
            disc_lr_multiplier: float = 1.0,
            # Optimizer params
            learning_rate: float = 1e-5,
            end_lr: float = 5e-6,
            decay_steps: int = 200000,
            betas: Tuple[float, float] = (0.5, 0.9),
            # Training params
            gradient_clip_val: float = 1.0,
            ema_decay: float = 0.9999,
            use_ema: bool = True,
            # RGT-specific extra losses (applied only to data_type=='rgt' samples)
            rgt_pos_reg_weight: float = 0.0,
            rgt_pos_perc_weight: float = 0.0,
            rgt_pos_discret: int = 128,
            stage: str = "full",
            weight_decay: float = 0.01,
    ):
        super().__init__()

        print('========================')
        print('recon_loss_weight:', recon_loss_weight)
        print('perceptual_weight:', perceptual_weight)
        print('kl_weight:', kl_weight)
        print('gan_weight:', gan_weight)
        print('gan_loss_type:', gan_loss_type)
        print('grad_checkpoint:', grad_checkpoint)
        print('========================')

        self.save_hyperparameters()

        # Build VAE model
        self.vae = TrainableSeismicVAE(
            dim=dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            grad_checkpoint=grad_checkpoint,
        )

        self.vae.vae.set_training_stage(stage)
        self.stage = stage
        self.register_buffer("generator_steps", torch.tensor(0,dtype=torch.long))
        self.grad_checkpoint = grad_checkpoint
        n_on, n_total, _ = self.vae.vae.grad_checkpoint_summary()
        print(f'grad_checkpoint blocks: {n_on}/{n_total} enabled')

        # Loss functions
        self.recon_loss_type = recon_loss_type
        self.recon_loss_weight = recon_loss_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight

        # Perceptual loss
        if perceptual_weight > 0:
            if perceptual_type == "lpips":
                self.perceptual_loss = LPIPSLoss()
            else:
                self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

        # RGT-specific position embedding (sin/cos 映射)，对 data_type=='rgt' 的样本启用
        self.rgt_pos_reg_weight = rgt_pos_reg_weight
        self.rgt_pos_perc_weight = rgt_pos_perc_weight
        self.pos_emb = LinearDecayPositionEmbedding(discret=rgt_pos_discret)

        # GAN components
        self.use_gan = use_gan
        self.gan_weight = gan_weight
        self.gan_loss_type = gan_loss_type
        self.disc_start_step = disc_start_step
        self.disc_lr_multiplier = disc_lr_multiplier

        if use_gan:
            self.discriminator = Discriminator3D(
                in_channels=3,
                base_channels=disc_channels,
                num_layers=disc_num_layers,
            )
            self.adv_loss = AdversarialLoss(type=gan_loss_type)
        else:
            self.discriminator = None
            self.adv_loss = None

        # EMA
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        if use_ema:
            self.ema_vae = None  # Will be initialized in on_fit_start

        # Optimizer params
        self.learning_rate = learning_rate
        self.end_lr = end_lr
        self.betas = betas
        self.decay_steps = decay_steps
        self.gradient_clip_val = gradient_clip_val

        # For automatic optimization with multiple optimizers
        self.automatic_optimization = not use_gan

    def _init_ema(self):
        """Create the EMA copy if it does not exist yet."""
        if self.use_ema and self.ema_vae is None:
            import copy
            self.ema_vae = copy.deepcopy(self.vae)
            self.ema_vae.requires_grad_(False)

    def on_fit_start(self):
        """Initialize EMA model and ensure CUDA stream consistency."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._init_ema()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.no_grad()
    def update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema or self.ema_vae is None:
            return

        for ema_param, param in zip(
                self.ema_vae.parameters(), self.vae.parameters()
        ):
            ema_param.data.mul_(self.ema_decay).add_(
                param.data, alpha=1 - self.ema_decay
            )

    def compute_reconstruction_loss(
            self,
            recon: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss."""
        if self.recon_loss_type == "l1":
            return F.l1_loss(recon, target)
        elif self.recon_loss_type == "l2":
            return F.mse_loss(recon, target)
        elif self.recon_loss_type == "mixed":
            return F.l1_loss(recon, target) + F.mse_loss(recon, target)
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")

    def compute_kl_loss(
            self,
            mu: torch.Tensor,
            log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss."""
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return kl

    def _select_rgt_indices(self, data_type) -> Optional[torch.Tensor]:
        """从 batch 的 data_type 字段中筛出 'rgt' 样本索引。

        Args:
            data_type: list[str] 或 tuple[str]（pytorch dataloader 默认 collate 行为）
        Returns:
            LongTensor 索引；若 batch 内无 rgt 样本则返回 None。
        """
        if data_type is None:
            return None
        if isinstance(data_type, str):
            data_type = [data_type]
        idx = [i for i, t in enumerate(data_type) if t == 'rgt']
        if not idx:
            return None
        return torch.as_tensor(idx, dtype=torch.long, device=self.device)

    def _compute_rgt_pos_extra_loss(
            self,
            recon: torch.Tensor,
            seismic: torch.Tensor,
            rgt_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 batch 中 data_type=='rgt' 的样本，计算 pos-emb 映射后的额外回归 + 感知损失。

        Args:
            recon, seismic: (B, C, T, H, W)
            rgt_idx: LongTensor，选出 rgt 样本的索引
        Returns:
            (pos_reg_loss, pos_perc_loss) —— 都是标量 tensor。
            若 perceptual_loss 未启用，pos_perc_loss 返回 0。
        """
        # 取出 rgt 子 batch；C=3 三通道是 repeat 副本，mean 取出单通道 RGT 表示
        recon_rgt = recon.index_select(0, rgt_idx).mean(dim=1, keepdim=True)  # (Br, 1, T, H, W)
        seismic_rgt = seismic.index_select(0, rgt_idx).mean(dim=1, keepdim=True)

        # sin/cos 位置编码
        recon_pos = self.pos_emb(recon_rgt)   # (Br, 3, T, H, W)
        seismic_pos = self.pos_emb(seismic_rgt)

        # 映射后的回归 (L1) 损失
        pos_reg_loss = F.l1_loss(recon_pos, seismic_pos)

        # 映射后的感知损失
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            pos_perc_loss = self.perceptual_loss(recon_pos, seismic_pos)
        else:
            pos_perc_loss = torch.zeros((), device=recon.device, dtype=recon.dtype)

        return pos_reg_loss, pos_perc_loss

    def compute_generator_loss(
            self,
            d_real: torch.Tensor,
            d_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute generator GAN loss."""
        return self.adv_loss(d_real, d_fake, is_disc=False)

    def compute_discriminator_loss(
            self,
            d_real: torch.Tensor,
            d_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminator GAN loss."""
        return self.adv_loss(d_real, d_fake, is_disc=True)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step."""
        seismic = batch["img"]  # [B, C, T, H, W]
        data_type = batch.get("data_type", None)
        # Dataset applies uniform axis permutations once, before RGB replication.

        if self.use_gan:
            return self._training_step_gan(seismic, batch_idx, data_type=data_type)
        else:
            return self._training_step_vae(seismic, batch_idx, data_type=data_type)

    def _training_step_vae(
            self,
            seismic: torch.Tensor,
            batch_idx: int,
            data_type=None,
    ) -> torch.Tensor:
        """VAE-only training step."""
        # Forward pass
        recon, mu, log_var = self.vae(seismic)

        # Compute losses
        recon_loss = self.compute_reconstruction_loss(recon, seismic)
        kl_loss = self.compute_kl_loss(mu, log_var)

        total_loss = recon_loss * self.recon_loss_weight + self.kl_weight * kl_loss

        # Perceptual loss
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(recon, seismic)
            total_loss = total_loss + self.perceptual_weight * perceptual_loss
            self.log("train/perceptual_loss", perceptual_loss, prog_bar=False)

        # RGT-specific extra losses: pos-emb 映射后的回归 + 感知损失
        rgt_idx = self._select_rgt_indices(data_type)
        if rgt_idx is not None and (self.rgt_pos_reg_weight > 0 or self.rgt_pos_perc_weight > 0):
            pos_reg_loss, pos_perc_loss = self._compute_rgt_pos_extra_loss(recon, seismic, rgt_idx)
            total_loss = total_loss + self.rgt_pos_reg_weight * pos_reg_loss \
                                    + self.rgt_pos_perc_weight * pos_perc_loss
            self.log("train/rgt_pos_reg_loss", pos_reg_loss, prog_bar=False)
            self.log("train/rgt_pos_perc_loss", pos_perc_loss, prog_bar=False)

        # Logging
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/kl_loss", kl_loss, prog_bar=False)

        return total_loss

    def _training_step_gan(
            self,
            seismic: torch.Tensor,
            batch_idx: int,
            data_type=None,
    ) -> None:
        """GAN training step with manual optimization."""
        opt_vae, opt_disc = self.optimizers()
        # =================== Train Generator (VAE) ===================
        opt_vae.zero_grad()

        # Forward pass
        recon, mu, log_var = self.vae(seismic)

        # Compute VAE losses
        recon_loss = self.compute_reconstruction_loss(recon, seismic)
        kl_loss = self.compute_kl_loss(mu, log_var)

        vae_loss = self.recon_loss_weight * recon_loss + self.kl_weight * kl_loss

        # Perceptual loss
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(recon, seismic)
            vae_loss = vae_loss + self.perceptual_weight * perceptual_loss
            self.log("train/perceptual_loss", perceptual_loss, prog_bar=False)

        # RGT-specific extra losses: pos-emb 映射后的回归 + 感知损失 (不参与 GAN 对抗)
        rgt_idx = self._select_rgt_indices(data_type)
        if rgt_idx is not None and (self.rgt_pos_reg_weight > 0 or self.rgt_pos_perc_weight > 0):
            pos_reg_loss, pos_perc_loss = self._compute_rgt_pos_extra_loss(recon, seismic, rgt_idx)
            vae_loss = vae_loss + self.rgt_pos_reg_weight * pos_reg_loss \
                                + self.rgt_pos_perc_weight * pos_perc_loss
            self.log("train/rgt_pos_reg_loss", pos_reg_loss, prog_bar=False)
            self.log("train/rgt_pos_perc_loss", pos_perc_loss, prog_bar=False)

        # Generator loss (only after warmup)
        if self.stage == "full" and int(self.generator_steps) >= self.disc_start_step:
            self.discriminator.requires_grad_(False)
            with torch.no_grad():
                real_logits = self.discriminator(seismic)
            fake_logits = self.discriminator(recon)
            gen_loss = self.compute_generator_loss(real_logits, fake_logits)
            fidelity = vae_loss - self.kl_weight * kl_loss
            last = self.vae.vae.decoder.head[-1].weight
            if not last.requires_grad:
                last = self.vae.vae.decoder.upsamples[11].time_conv.weight
            rec_grad = torch.autograd.grad(fidelity,last,retain_graph=True)[0]
            adv_grad = torch.autograd.grad(gen_loss,last,retain_graph=True)[0]
            adaptive = (rec_grad.norm()/(adv_grad.norm()+1e-4)).clamp(0,1e4).detach()
            vae_loss = vae_loss + self.gan_weight * adaptive * gen_loss
            self.log("train/adaptive_gan_weight", adaptive, prog_bar=False)
            self.log("train/g_loss", gen_loss, prog_bar=False)

        if not torch.isfinite(vae_loss).all():
            print(f"[NaN] vae_loss at step {self.global_step}")
            opt_vae.zero_grad(set_to_none=True)
            opt_disc.zero_grad(set_to_none=True)
            return None

        self.manual_backward(vae_loss)
        self.clip_gradients(opt_vae, gradient_clip_val=self.gradient_clip_val)
        opt_vae.step()

        # =================== Train Discriminator ===================
        if self.stage == "full" and int(self.generator_steps) >= self.disc_start_step:
            self.discriminator.requires_grad_(True)
            opt_disc.zero_grad()
            real_logits = self.discriminator(seismic)
            fake_logits = self.discriminator(recon.detach())
            disc_loss = self.compute_discriminator_loss(real_logits, fake_logits)
            self.manual_backward(disc_loss)
            self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val)
            opt_disc.step()
            self.log("train/d_loss", disc_loss, prog_bar=False)
        self.generator_steps.add_(1)
        # Logging
        self.log("train/loss", vae_loss.detach(), prog_bar=True)
        self.log("train/recon_loss", recon_loss.detach(), prog_bar=True)
        self.log("train/kl_loss", kl_loss.detach(), prog_bar=False)

        for name, t in [("recon", recon), ("mu", mu), ("log_var", log_var)]:
            if not torch.isfinite(t).all():
                print(f"[NaN] {name} non-finite at step {self.global_step}")

        # Update EMA
        if self.use_ema:
            self.update_ema()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        seismic = batch["img"]

        # Forward pass (deterministic for validation)
        eval_vae = self.ema_vae if self.use_ema and self.ema_vae is not None else self.vae
        recon, mu, log_var = eval_vae(seismic, deterministic=True)

        # Compute losses
        recon_loss = self.compute_reconstruction_loss(recon, seismic)
        kl_loss = self.compute_kl_loss(mu, log_var)

        total_loss = recon_loss + self.kl_weight * kl_loss

        # Logging
        self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val/recon_loss", recon_loss, prog_bar=True, sync_dist=True)
        self.log("val/kl_loss", kl_loss, prog_bar=False, sync_dist=True)

        # Log sample reconstructions (first batch only)
        if batch_idx == 0:
            self._log_reconstructions(seismic, recon)

        return total_loss

    def _log_reconstructions(
            self,
            seismic: torch.Tensor,
            recon: torch.Tensor,
            num_samples: int = 4,
    ):
        """Log sample reconstructions to tensorboard."""
        if self.logger is None:
            return

        # Take first few samples
        seismic = seismic[:num_samples]
        recon = recon[:num_samples]

        # Denormalize: [-1, 1] -> [0, 1]
        seismic = (seismic + 1) / 2
        recon = (recon + 1) / 2

        # Get middle frame
        t = seismic.shape[2] // 2
        seismic_frame = seismic[:, :, t]  # [B, C, H, W]
        recon_frame = recon[:, :, t]

        # Concatenate original and reconstruction
        comparison = torch.cat([seismic_frame, recon_frame], dim=-1)

        # Log to tensorboard
        if hasattr(self.logger, "experiment"):
            for i, img in enumerate(comparison):
                self.logger.experiment.add_image(
                    f"val/sample_{i}",
                    img.clamp(0, 1),
                    self.global_step,
                )

    def configure_optimizers(self):
        vae_optimizer = torch.optim.AdamW(
            [p for p in self.vae.parameters() if p.requires_grad],
            lr=self.learning_rate, betas=self.betas, weight_decay=self.hparams.weight_decay)
        if not self.use_gan:
            return vae_optimizer
        disc_optimizer = torch.optim.AdamW(self.discriminator.parameters(),
            lr=self.learning_rate*self.disc_lr_multiplier,betas=self.betas,
            weight_decay=self.hparams.weight_decay)
        return [vae_optimizer,disc_optimizer]

    def on_before_zero_grad(self, optimizer):
        # Automatic optimization updates EMA after an optimizer step.
        if not self.use_gan and self.trainer.global_step > 0:
            self.update_ema()
            self.generator_steps.add_(1)

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler,"set_epoch"):
            sampler.set_epoch(self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        """Save EMA weights if available."""
        if self.use_ema and self.ema_vae is not None:
            checkpoint["ema_state_dict"] = self.ema_vae.state_dict()

        # 不保存感知损失权重：它们是冻结的预训练权重，与训练进度无关，
        # 存进去只会让 ckpt 变大（LPIPS-vgg 约 59MB）并在换后端时导致加载失败。
        # PL 先构建 state_dict 再调本钩子，所以这里剔除是有效的。
        if "state_dict" in checkpoint:
            strip_perceptual_keys(checkpoint["state_dict"])

    def on_load_checkpoint(self, checkpoint):
        """Load EMA weights if available."""
        if self.use_ema and "ema_state_dict" in checkpoint:
            self._init_ema()
            self.ema_vae.load_state_dict(checkpoint["ema_state_dict"])

        # 让 ckpt 的感知损失 key 与当前模型对齐，兼容两种情况：
        #   - 旧 ckpt 存了别的后端（如 alex）的权重 -> 形状不符，丢弃换成当前的
        #   - 新 ckpt 已不含这些 key              -> 补上，避免 strict 报 missing
        # 本钩子在 load_model_state_dict 之前调用（PL 2.x），所以改动生效。
        if "state_dict" in checkpoint:
            align_perceptual_keys(checkpoint["state_dict"], self, log=_rank_zero_print)


class LossPrintCallback(Callback):
    """每隔 k 步打印这 k 步的平均损失和耗时（仅主进程打印）"""

    def __init__(self, every_k_steps: int = 100):
        super().__init__()
        self.every_k_steps = every_k_steps

        self.loss_accumulator = defaultdict(list)
        self.last_print_step = 0
        self.last_print_time = None

        self.loss_items = [
            ('train/loss', 'Total', 4),
            ('train/recon_loss', 'Recon', 4),
            ('train/kl_loss', 'KL', 6),
            ('train/perceptual_loss', 'Perceptual', 4),
            ('train/rgt_pos_reg_loss', 'RGT_PosReg', 4),
            ('train/rgt_pos_perc_loss', 'RGT_PosPerc', 4),
            ('train/g_loss', 'G_loss', 4),
            ('train/d_loss', 'D_loss', 4),
            ('train/d_real', 'D_real', 3),
            ('train/d_fake', 'D_fake', 3),
        ]

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.last_print_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 只在主进程做任何事情
        if not trainer.is_global_zero:
            return

        metrics = trainer.logged_metrics
        global_step = trainer.global_step

        for key, _, _ in self.loss_items:
            if key in metrics:
                value = metrics[key]
                if hasattr(value, "item"):
                    value = value.item()
                self.loss_accumulator[key].append(value)

        if global_step - self.last_print_step >= self.every_k_steps and global_step > 0:
            current_time = time.time()
            elapsed = current_time - (self.last_print_time or current_time)

            # 用任意一个已有 key 的长度作为实际累积步数
            actual_steps = 0
            for k in self.loss_accumulator:
                actual_steps = len(self.loss_accumulator[k])
                break

            loss_strs = []
            for key, name, decimals in self.loss_items:
                values = self.loss_accumulator.get(key, [])
                if values:
                    avg_value = sum(values) / len(values)
                    loss_strs.append(f"{name}: {avg_value:.{decimals}f}")

            if loss_strs and actual_steps > 0:
                steps_per_sec = actual_steps / elapsed if elapsed > 0 else 0
                print(
                    f"\n[Step {global_step}] ({actual_steps} steps in {elapsed:.1f}s, {steps_per_sec:.2f} it/s) "
                    + " | ".join(loss_strs)
                )

            self.loss_accumulator.clear()
            self.last_print_step = global_step
            self.last_print_time = current_time


class CubeVisualizationCallback(Callback):
    """
    每隔 k 个 global_step 可视化一次输入与重建 cube（2x3 网格：Input vs Recon，三方向切片）

    设计目标：DDP安全、原子写入、防裁切、防半文件、防状态污染、低开销、训练不中断。
    """

    def __init__(
        self,
        save_dir: str,
        every_k_steps: int = 1000,
        batch_key: str = "img",
        cmap: str = "seismic",
        figsize: Tuple[int, int] = (15, 10),
        dpi: int = 150,
        symmetric_color: bool = True,
        percentile_clip: Optional[Tuple[float, float]] = None,  # e.g. (1, 99) 或 None
        max_items_per_batch: int = 1,  # 默认只取第一个样本做可视化/重建
        enabled: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            save_dir: 保存图片目录
            every_k_steps: 每隔多少 global_step 保存一次（>=1）
            batch_key: batch dict 中 cube 的 key
            cmap: matplotlib colormap
            figsize: 图片大小
            dpi: 分辨率
            symmetric_color: 是否使用对称 vmin/vmax（适合 seismic）
            percentile_clip: 用分位数裁剪显示范围，缓解极端值影响（例如 (1, 99)），None 表示不用
            max_items_per_batch: 取 batch 前多少个样本（建议 1）
            enabled: 是否启用
            verbose: 是否打印日志
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.every_k_steps = int(every_k_steps)
        self.batch_key = batch_key
        self.cmap = cmap
        self.figsize = figsize
        self.dpi = dpi
        self.symmetric_color = symmetric_color
        self.percentile_clip = percentile_clip
        self.max_items_per_batch = int(max_items_per_batch)
        self.enabled = enabled
        self.verbose = verbose

        if self.every_k_steps < 1:
            raise ValueError("every_k_steps must be >= 1")
        if self.max_items_per_batch < 1:
            raise ValueError("max_items_per_batch must be >= 1")

    # -------------------------
    # Lightning hooks
    # -------------------------
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # 只需 rank0 创建目录（其他rank不写图）
        if trainer.is_global_zero:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        if not self.enabled:
            return

        # DDP：只让 rank0 保存，避免并发写同名文件导致 PNG 截断/只保存一半
        if not trainer.is_global_zero:
            return

        # 避免 sanity checking 阶段产物污染
        if getattr(trainer, "sanity_checking", False):
            return

        global_step = int(trainer.global_step)
        if global_step == 0 or (global_step % self.every_k_steps != 0):
            return

        if self.batch_key not in batch:
            if self.verbose:
                print(f"[CubeVisualizationCallback] batch 缺少 key='{self.batch_key}'，跳过。")
            return

        cube = batch[self.batch_key]  # 期望 [B, C, T, H, W]
        if not isinstance(cube, torch.Tensor):
            if self.verbose:
                print(f"[CubeVisualizationCallback] batch['{self.batch_key}'] 不是 Tensor，跳过。")
            return
        if cube.ndim != 5:
            if self.verbose:
                print(f"[CubeVisualizationCallback] 期望 5D [B,C,T,H,W]，但得到 shape={tuple(cube.shape)}，跳过。")
            return

        if random.random() < 0.5:
            cube = cube.permute(0, 1, 4, 3, 2)  # [B, C, T, H, W]
        # 标准 Conv3d 不需要因果 padding

        # 只取前 N 个样本（默认1），减少可视化开销
        cube = cube[: self.max_items_per_batch]

        # 移动到模型设备
        device = pl_module.device
        cube = cube.to(device, non_blocking=True)

        # 进行重建：尽量不污染训练状态
        was_training = pl_module.training
        try:
            with torch.no_grad():
                pl_module.eval()

                # 安全 autocast：根据 precision + device 决定
                with self._autocast_context(trainer, device):
                    if not hasattr(pl_module, "vae"):
                        raise AttributeError("pl_module 没有 'vae' 属性，无法可视化重建结果。")

                    out = pl_module.vae(cube, deterministic=True)

                    # 兼容：vae 可能返回 (recon, *others) 或 直接 recon
                    recon = out[0] if isinstance(out, (tuple, list)) else out

        except Exception as e:
            # 可视化失败不应让训练崩
            if self.verbose:
                print(f"[CubeVisualizationCallback] 重建/保存前处理失败（step={global_step}）：{repr(e)}")
            return
        finally:
            # 恢复训练状态
            if was_training:
                pl_module.train()
            else:
                pl_module.eval()

        # 保存图像
        try:
            self._save_visualization(
                input_cube=cube,
                recon_cube=recon,
                global_step=global_step,
                epoch=int(trainer.current_epoch),
            )
        except Exception as e:
            if self.verbose:
                print(f"[CubeVisualizationCallback] 保存失败（step={global_step}）：{repr(e)}")

    # -------------------------
    # Helpers
    # -------------------------
    def _autocast_context(self, trainer: Trainer, device: torch.device):
        """根据 trainer.precision 与 device 决定 autocast 策略。"""
        if device.type != "cuda":
            return nullcontext()

        prec = str(getattr(trainer, "precision", "")).lower()
        # Lightning 常见：'16-mixed', 'bf16-mixed', 也可能是 16 / '32-true'
        if "bf16" in prec:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if "16" in prec:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _to_avg_np(self, cube: torch.Tensor) -> np.ndarray:
        """
        cube: [B,C,T,H,W] 或 [C,T,H,W]，返回 [T,H,W] 的 numpy float32
        """
        if cube.ndim == 5:
            cube = cube[0]  # [C,T,H,W]
        if cube.ndim != 4:
            raise ValueError(f"Expect [C,T,H,W], got {tuple(cube.shape)}")
        avg = cube.mean(dim=0)  # [T,H,W]
        avg = avg.detach().float().cpu().contiguous().numpy().astype(np.float32)
        # 防 NaN/Inf
        avg = np.nan_to_num(avg, nan=0.0, posinf=0.0, neginf=0.0)
        return avg

    def _compute_vmin_vmax(self, arrays: list) -> Tuple[float, float]:
        """
        arrays: list of 2D arrays
        """
        data = np.concatenate([a.reshape(-1) for a in arrays], axis=0)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return -1.0, 1.0

        if self.percentile_clip is not None:
            lo, hi = self.percentile_clip
            lo_v, hi_v = np.percentile(data, [lo, hi])
            if self.symmetric_color:
                vmax = float(max(abs(lo_v), abs(hi_v), 1e-6))
                return -vmax, vmax
            # 非对称
            if hi_v - lo_v < 1e-12:
                return float(lo_v - 1e-6), float(hi_v + 1e-6)
            return float(lo_v), float(hi_v)

        # 不用分位数裁剪：直接用 max(abs)
        if self.symmetric_color:
            vmax = float(max(np.max(np.abs(data)), 1e-6))
            return -vmax, vmax

        vmin, vmax = float(np.min(data)), float(np.max(data))
        if vmax - vmin < 1e-12:
            return vmin - 1e-6, vmax + 1e-6
        return vmin, vmax

    def _save_visualization(
        self,
        input_cube: torch.Tensor,
        recon_cube: torch.Tensor,
        global_step: int,
        epoch: int,
    ) -> None:
        """
        input_cube: [B,C,T,H,W]
        recon_cube: [B,C,T,H,W]（或至少可按此索引）
        """
        inp = self._to_avg_np(input_cube)     # [T,H,W]
        rec = self._to_avg_np(recon_cube)     # [T,H,W]

        T, H, W = inp.shape
        t0, h0, w0 = T // 2, H // 2, W // 2

        # 三个方向切片
        # D/T 方向：固定 t -> [H,W]
        # H 方向：固定 h -> [T,W]
        # W 方向：固定 w -> [T,H]
        inp_slices = [inp[t0, :, :], inp[:, h0, :], inp[:, :, w0]]
        rec_slices = [rec[t0, :, :], rec[:, h0, :], rec[:, :, w0]]

        vmin, vmax = self._compute_vmin_vmax(inp_slices + rec_slices)

        fig, axes = plt.subplots(2, 3, figsize=self.figsize, dpi=self.dpi)

        titles = ["T-mid slice (H×W)", "H-mid slice (T×W)", "W-mid slice (T×H)"]

        ims = []
        for j in range(3):
            im0 = axes[0, j].imshow(inp_slices[j], cmap=self.cmap, vmin=vmin, vmax=vmax)
            axes[0, j].set_title(titles[j], fontsize=12)
            axes[0, j].axis("off")
            ims.append(im0)

            im1 = axes[1, j].imshow(rec_slices[j], cmap=self.cmap, vmin=vmin, vmax=vmax)
            axes[1, j].axis("off")
            ims.append(im1)

        axes[0, 0].set_ylabel("Input", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Reconstructed", fontsize=12, fontweight="bold")

        # 更稳的布局（避免 tight_layout + bbox_inches="tight" 导致裁切异常）
        fig.suptitle(f"Epoch {epoch} | Step {global_step}  (key='{self.batch_key}')", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

        # 原子写入：先写临时文件，再 replace
        self.save_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.save_dir / f"cube_vis_e{epoch:04d}_s{global_step:08d}.png"
        tmp_path = self.save_dir / f".tmp_{final_path.stem}_{uuid.uuid4().hex}.png"

        fig.savefig(tmp_path)
        plt.close(fig)
        os.replace(tmp_path, final_path)

        if self.verbose:
            print(f"[CubeVisualizationCallback] Saved: {final_path}")
