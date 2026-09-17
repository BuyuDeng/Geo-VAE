"""CAS RGT-conditioned latent diffusion trainer; arrays use crossline,inline,time."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Reuse the exact Wan/GeoVAE implementation used by the existing CAS jobs.
from .own_conditional_diffusion3d import ConditionalDDPM, ConditionalUNet3D
from vae.wan_video_vae import WanVideoVAE


def vae_normalize(x, clip=3.2):
    flat=x.reshape(x.shape[0],-1)
    mean=flat.mean(1)[:,None,None,None]
    std=flat.std(1,correction=0)[:,None,None,None]
    return ((x-mean)/(std+1e-6)).clamp(-clip,clip)/clip


def to_vae(x):
    from vae.inference import causal_resize
    return causal_resize(x[:,None].repeat(1,3,1,1,1))


def from_vae(x,target_shape):
    y=x.mean(1,keepdim=True)
    if tuple(y.shape[-3:])!=tuple(target_shape):
        y=F.interpolate(y,size=target_shape,mode="trilinear",align_corners=False)
    return y[:,0]


class PairedSeismicFaultRGT(Dataset):
    """Strictly paired common crops from SX, FX and RGT source cubes."""

    def __init__(self, sx_root, fx_root, rgt_root, crop_txy, seed, train):
        roots = {"sx": Path(sx_root), "rgt": Path(rgt_root)}
        if fx_root: roots["fx"]=Path(fx_root)
        files = {kind: {p.name: p for p in root.glob("*.npy")} for kind, root in roots.items()}
        self.names = sorted(set.intersection(*(set(group) for group in files.values())))
        if not self.names:
            raise RuntimeError("no common SX/FX/RGT file names")
        self.files, self.crop, self.seed, self.train, self.epoch = files, tuple(crop_txy), int(seed), bool(train), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        import random

        name = self.names[index]
        rng = random.Random(self.seed + index + (1_000_003 * self.epoch if self.train else 0))
        sx = np.load(self.files["sx"][name], mmap_mode="r")
        fx = np.load(self.files["fx"][name], mmap_mode="r") if "fx" in self.files else None
        rgt = np.load(self.files["rgt"][name], mmap_mode="r")
        if sx.shape!=rgt.shape or (fx is not None and sx.shape!=fx.shape):
            raise RuntimeError(f"shape mismatch for {name}: {sx.shape}, {None if fx is None else fx.shape}, {rgt.shape}")
        if any(n<size for n,size in zip(sx.shape,self.crop)):
            raise ValueError(f"crop {self.crop} does not fit {sx.shape}")
        starts = [rng.randint(0, extent-size) for extent,size in zip(sx.shape,self.crop)]
        sl = tuple(slice(start, start + size) for start, size in zip(starts, self.crop))
        # FX=30 is the non-fault horizon/unconformity code and must not be a fault condition.
        fault = ((np.asarray(fx[sl]) > 0) & (np.asarray(fx[sl]) < 30)).astype(np.float32) if fx is not None else np.zeros(self.crop,dtype=np.float32)
        seismic = np.asarray(sx[sl], np.float32)
        relative_time = np.asarray(rgt[sl], np.float32)
        # Input files are already crossline, inline, time/depth.
        seismic,fault,relative_time=(np.array(v,copy=True,order="C") for v in (seismic,fault,relative_time))
        if self.train:
            import itertools
            perm=rng.choice(list(itertools.permutations(range(3))))
            seismic,fault,relative_time=(np.ascontiguousarray(value.transpose(perm))
                                       for value in (seismic,fault,relative_time))
        return torch.from_numpy(seismic), torch.from_numpy(fault), torch.from_numpy(relative_time), name


class CachedLatentTriplets(Dataset):
    """Pre-encoded, spatially aligned (SX, FX, RGT) latent crop triplets."""

    def __init__(self, root, seed, train):
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*.pt"))
        if not self.paths:
            raise RuntimeError(f"no cached latent triplets in {self.root}")
        self.seed, self.train, self.epoch = int(seed), bool(train), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        item = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        sx, fx, rgt = item["sx"], item["fx"], item["rgt"]
        # Cached values are already in GeoVAE latent space and spatially aligned.
        return sx, fx, rgt, item["name"]


def rgt_normalize(x):
    """Per-crop monotonic affine map to the GeoVAE input range [-1, 1]."""
    lo = x.amin(dim=(1, 2, 3), keepdim=True)
    hi = x.amax(dim=(1, 2, 3), keepdim=True)
    return 2 * (x - lo) / (hi - lo + 1e-6) - 1


def append_jsonl(path, item):
    with path.open("a") as handle:
        handle.write(json.dumps(item) + "\n")


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, source):
        for average, current in zip(self.model.parameters(), source.parameters()):
            average.lerp_(current, 1.0 - self.decay)


@torch.no_grad()
def ddim_sample(diffusion, shape, condition, steps, x0_clip):
    """Deterministic DDIM sampler with an explicit final x0 prediction."""
    times = torch.linspace(diffusion.timesteps - 1, 0, steps, device=condition.device).long().tolist() + [-1]
    image = torch.randn(shape, device=condition.device)
    for current, previous in zip(times[:-1], times[1:]):
        t = torch.full((shape[0],), current, device=image.device, dtype=torch.long)
        with torch.autocast(condition.device.type, dtype=torch.float16, enabled=condition.device.type=="cuda"):
            predicted = diffusion.model(image, t, condition)
        alpha = diffusion._extract(diffusion.alphas_cumprod, t, image.shape)
        sqrt_alpha, sqrt_sigma = torch.sqrt(alpha), torch.sqrt(1 - alpha)
        if diffusion.prediction_type == "v":
            x0 = (sqrt_alpha * image - sqrt_sigma * predicted).clamp(-x0_clip, x0_clip)
            noise = sqrt_sigma * image + sqrt_alpha * predicted
        else:
            noise = predicted
            x0 = ((image - sqrt_sigma * noise) / sqrt_alpha).clamp(-x0_clip, x0_clip)
        if previous < 0:
            return x0
        previous_t = torch.full_like(t, previous)
        previous_alpha = diffusion._extract(diffusion.alphas_cumprod, previous_t, image.shape)
        image = torch.sqrt(previous_alpha) * x0 + torch.sqrt(1 - previous_alpha) * noise
    raise RuntimeError("DDIM schedule did not reach x0")


def reduce_mean(value, device, distributed):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def latent_standardizer(path, device):
    stats = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key: (value["mean"].to(device)[None, :, None, None, None], value["std"].to(device)[None, :, None, None, None].clamp_min(1e-6))
        for key, value in stats.items()
    }


def standardize(value, key, stats):
    mean, std = stats[key]
    return (value - mean) / std


def destandardize(value, key, stats):
    mean, std = stats[key]
    return value * std + mean


def select_condition(z_fx, z_rgt, condition_source):
    """Return the requested frozen GeoVAE condition without leaking the other modality."""
    if condition_source == "fx":
        return z_fx
    if condition_source == "rgt":
        return z_rgt
    if condition_source == "fx_rgt":
        return torch.cat((z_fx, z_rgt), dim=1)
    raise ValueError(f"condition_source must be fx, rgt, or fx_rgt; got {condition_source!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    condition_source = config.get("condition_source", "fx_rgt")
    condition_channels = 16 if condition_source in {"fx", "rgt"} else 32

    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device, rank = torch.device("cuda", local_rank), dist.get_rank()
    else:
        device, rank = torch.device("cuda"), 0
    main_rank = rank == 0
    torch.manual_seed(int(config["seed"]) + rank)
    np.random.seed(int(config["seed"]) + rank)

    output = Path(config["output_root"]) / args.run_id
    if main_rank:
        # A failed pre-flight (for example an OOM before epoch 0) may leave an
        # empty run directory.  Preserve it for diagnostics and allow the
        # corrected launch to reuse the same explicit run identifier.
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if distributed:
        dist.barrier()

    cache_root = config.get("latent_cache_root")
    stats = latent_standardizer(config["latent_stats_path"], device) if config.get("latent_stats_path") else None
    if cache_root:
        datasets = {
            split: CachedLatentTriplets(Path(cache_root) / split, config["seed"], split == "train")
            for split in ("train", "valid")
        }
    else:
        datasets = {
            split: PairedSeismicFaultRGT(
                config[f"{split}_sx_root"], config.get(f"{split}_fx_root"), config[f"{split}_rgt_root"],
                config["raw_crop_txy"], config["seed"], split == "train",
            )
            for split in ("train", "valid")
        }
    for split,key in (("train","expected_train_crops"),("valid","expected_valid_crops")):
        if cache_root and len(datasets[split])!=config[key]:
            raise ValueError(f"paper protocol expects {config[key]} {split} crops, found {len(datasets[split])}")
    samplers = {
        split: DistributedSampler(datasets[split], shuffle=split == "train", seed=config["seed"]) if distributed else None
        for split in datasets
    }
    loaders = {
        split: DataLoader(
            datasets[split], batch_size=config["batch_size"], shuffle=split == "train" and not distributed,
            sampler=samplers[split], num_workers=config["num_workers"], pin_memory=True,
            persistent_workers=False,
        )
        for split in datasets
    }

    vae = WanVideoVAE().to(device).half()
    vae.load_state_dict(torch.load(config["vae_checkpoint"], map_location=device))
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    # Two latent reductions (34 -> 17 -> 9) retain meaningful 3-D detail;
    # no full attention is used on the 34^3 grid because it is quadratic in voxels.
    net = ConditionalUNet3D(
        width=config["width"], blocks_per_scale=config["res_blocks"],
        condition_channels=condition_channels, dropout=config.get("dropout", 0.0),
    ).to(device)
    diffusion = ConditionalDDPM(net, timesteps=config["timesteps"], prediction_type=config.get("prediction_type", "epsilon")).to(device)
    train_diffusion = DDP(diffusion, device_ids=[device.index], broadcast_buffers=False) if distributed else diffusion
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    # A large 3-D model with global batch size 2 benefits from a short linear
    # warm-up before the long cosine decay; LR is stepped per optimizer update.
    total_steps = config["epochs"] * len(loaders["train"])
    warmup_steps = min(int(config["warmup_steps"]), total_steps - 1)
    def lr_factor(update):
        if update < warmup_steps:
            return (update + 1) / warmup_steps
        progress = (update - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(diffusion, config["ema_decay"]) if main_rank else None
    start_epoch, step, best = 0, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        diffusion.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if main_rank and checkpoint.get("ema"):
            ema.model.load_state_dict(checkpoint["ema"])
        start_epoch, step, best = int(checkpoint["epoch"]) + 1, int(checkpoint.get("step", 0)), float(checkpoint.get("loss", best))
        # The checkpoint predates the scheduler state.  Reconstruct its
        # position explicitly and apply the (possibly lowered) config LR,
        # rather than retaining a stale optimizer LR from the old run.
        scheduler.last_epoch = step
        resumed_lr = config["lr"] * lr_factor(step)
        for group in optimizer.param_groups:
            group["lr"] = resumed_lr
        scheduler._last_lr = [resumed_lr for _ in optimizer.param_groups]

    parameters = sum(parameter.numel() for parameter in diffusion.parameters())
    if main_rank:
        print(f"paired_train={len(datasets['train'])} paired_valid={len(datasets['valid'])} parameters={parameters:,}", flush=True)

    for epoch in range(start_epoch, config["epochs"]):
        if distributed:
            samplers["train"].set_epoch(epoch)
        for split in ("train", "valid"):
            train = split == "train"
            datasets[split].set_epoch(epoch if train else 0)
            # Keep validation loss on the current DDP-synchronised model for
            # every rank.  EMA is reserved for periodic sampling/checkpoints.
            active = train_diffusion if train else diffusion
            active.train(train)
            losses, sample_case = [], None
            with torch.set_grad_enabled(train):
                for seismic, fault, rgt, names in loaders[split]:
                    if cache_root:
                        z_sx = seismic.to(device, non_blocking=True)
                        z_fx = fault.to(device, non_blocking=True)
                        z_rgt = rgt.to(device, non_blocking=True)
                        raw_shape = tuple(config["raw_crop_txy"])
                    else:
                        seismic = seismic.to(device, non_blocking=True)
                        fault = fault.to(device, non_blocking=True)
                        rgt = rgt.to(device, non_blocking=True)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                            z_sx = vae.encode(to_vae(vae_normalize(seismic, config["z_clip"])).half(), device=device, tiled=config["vae_tiled"])
                            z_fx = torch.zeros_like(z_sx)
                            z_rgt = vae.encode(to_vae(rgt_normalize(rgt)).half(), device=device, tiled=config["vae_tiled"])
                        raw_shape = tuple(seismic.shape[-3:])
                    # Preserve GeoVAE's native causal latent geometry exactly:
                    # a 256^3 source crop becomes 33 x 32 x 32.  The custom
                    # U-Net restores odd sizes by interpolating to skip shapes,
                    # so it needs no artificial 33 -> 36 edge padding.
                    if not cache_root:
                        z_sx,z_fx,z_rgt=(z/config["latent_scale"] for z in (z_sx,z_fx,z_rgt))
                    if stats:
                        z_sx = standardize(z_sx, "sx", stats)
                        if condition_source in {"fx", "fx_rgt"}:
                            z_fx = standardize(z_fx, "fx", stats)
                        z_rgt = standardize(z_rgt, "rgt", stats)
                    condition = select_condition(z_fx, z_rgt, condition_source)
                    if main_rank and epoch == start_epoch and step == 0:
                        print(f"latent_target={tuple(z_sx.shape)} latent_condition={tuple(condition.shape)}", flush=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = active(z_sx, condition)
                    if train:
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(diffusion.parameters(), config["grad_clip"])
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        step += 1
                        if main_rank:
                            ema.update(diffusion)
                            if step % config["log_every_steps"] == 0:
                                append_jsonl(output / "train_steps.jsonl", {"epoch": epoch, "step": step, "loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"], "time": time.time()})
                    elif main_rank and sample_case is None:
                        sample_case = (z_sx.detach(), condition.detach(), raw_shape, names[0])
                    losses.append(float(loss.detach()))
            mean_loss = reduce_mean(float(np.mean(losses)), device, distributed)
            if main_rank:
                append_jsonl(output / "history.jsonl", {"epoch": epoch, "split": split, "loss": mean_loss, "time": time.time()})
                print(f"{epoch:03d} {split} diffusion_loss={mean_loss:.6f}", flush=True)
                if not train and mean_loss < best:
                    best = mean_loss
                    torch.save({"model": diffusion.state_dict(), "ema": ema.model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step, "loss": best, "config": config}, output / "best.pt")
                if not train and sample_case is not None and epoch % config["sample_every_epochs"] == 0:
                    target, condition, raw_shape, name = sample_case
                    samples = []
                    for seed in range(config["samples_per_validation"]):
                        torch.manual_seed(config["seed"] + epoch * 10_000 + seed)
                        z_out = ddim_sample(ema.model, target.shape, condition, config["sample_steps"], config["sample_x0_clip"])
                        # Preserve the native causal latent length: 33 for a
                        # 256 crop and 65 for a full 512 cube.
                        z_decode = destandardize(z_out, "sx", stats) if stats else z_out
                        raw = from_vae(vae.decode((z_decode * config["latent_scale"])[:, :, :target.shape[2]].half(), device=device, tiled=config["vae_tiled"]), raw_shape)
                        samples.append(raw[0].float().cpu().numpy())
                    target_decode = destandardize(target, "sx", stats) if stats else target
                    reference = from_vae(vae.decode((target_decode * config["latent_scale"])[:, :, :target.shape[2]].half(), device=device, tiled=config["vae_tiled"]), raw_shape)
                    archive = {
                        "samples": np.stack(samples), "reference": reference[0].float().cpu().numpy(),
                        "condition_source": condition_source, "name": name,
                    }
                    if condition_source in {"fx", "fx_rgt"}:
                        fx_decode = destandardize(z_fx[:1], "fx", stats) if stats else z_fx[:1]
                        fault_condition = from_vae(vae.decode((fx_decode * config["latent_scale"])[:, :, :target.shape[2]].half(), device=device, tiled=config["vae_tiled"]), raw_shape)
                        archive["fault"] = fault_condition[0].float().cpu().numpy()
                    if condition_source in {"rgt", "fx_rgt"}:
                        rgt_decode = destandardize(z_rgt[:1], "rgt", stats) if stats else z_rgt[:1]
                        rgt_condition = from_vae(vae.decode((rgt_decode * config["latent_scale"])[:, :, :target.shape[2]].half(), device=device, tiled=config["vae_tiled"]), raw_shape)
                        archive["rgt"] = rgt_condition[0].float().cpu().numpy()
                    np.savez_compressed(output / f"samples_epoch_{epoch:03d}.npz", **archive)
        if main_rank:
            torch.save({"model": diffusion.state_dict(), "ema": ema.model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step, "loss": best, "config": config}, output / "last.pt")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
