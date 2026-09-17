"""One-time GeoVAE encoding cache for conditional seismic diffusion.

Stores several deterministic, aligned 256^3 crops per training cube.  Every
stored pair uses crossline, inline, time/depth order and the same
modality-specific normalization as the online trainer.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

HERE = Path(__file__).resolve().parent
from .train_fx_rgt_to_sx_ldm import PairedSeismicFaultRGT, rgt_normalize, to_vae, vae_normalize

from vae.wan_video_vae import WanVideoVAE


class CropViews(Dataset):
    def __init__(self, base, views):
        self.base, self.views = base, int(views)

    def __len__(self):
        return len(self.base) * self.views

    def __getitem__(self, index):
        source_index, view = divmod(index, self.views)
        # PairedSeismicFaultRGT derives crop coordinates from seed + epoch.
        # A view-specific virtual epoch gives deterministic distinct crops.
        old = self.base.epoch
        self.base.epoch = 10_000 + view
        value = self.base[source_index]
        self.base.epoch = old
        seismic, fault, rgt, name = value
        return seismic, fault, rgt, f"{Path(name).stem}__view{view:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
        local_rank, rank = int(os.environ["LOCAL_RANK"]), dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, device = 0, torch.device("cuda")

    base = PairedSeismicFaultRGT(
        config[f"{args.split}_sx_root"], config.get(f"{args.split}_fx_root"), config[f"{args.split}_rgt_root"],
        # Training views must enable the dataset's epoch-dependent RNG; each
        # virtual epoch below then creates a distinct crop/flip.  Validation
        # remains a fixed, reproducible crop.
        config["raw_crop_txy"], config["seed"], train=args.split == "train",
    )
    dataset = CropViews(base, config[f"cache_{args.split}_views"])
    sampler = DistributedSampler(dataset, shuffle=False) if world > 1 else None
    loader = DataLoader(dataset, batch_size=config["cache_batch_size"], sampler=sampler,
                        shuffle=False, num_workers=config["num_workers"], pin_memory=True,
                        persistent_workers=config["num_workers"] > 0)
    root = Path(config["latent_cache_root"]) / args.split
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    vae = WanVideoVAE().to(device).half().eval()
    vae.load_state_dict(torch.load(config["vae_checkpoint"], map_location=device))
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    for batch_index, (sx, fx, rgt, names) in enumerate(loader):
        sx, fx, rgt = sx.to(device, non_blocking=True), fx.to(device, non_blocking=True), rgt.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z_sx = vae.encode(to_vae(vae_normalize(sx, config["z_clip"])).half(), device=device, tiled=config["vae_tiled"])
            z_fx = torch.zeros_like(z_sx)
            z_rgt = vae.encode(to_vae(rgt_normalize(rgt)).half(), device=device, tiled=config["vae_tiled"])
        for i, name in enumerate(names):
            torch.save({"sx": (z_sx[i] / config["latent_scale"]).cpu().contiguous(),
                        "fx": (z_fx[i] / config["latent_scale"]).cpu().contiguous(),
                        "rgt": (z_rgt[i] / config["latent_scale"]).cpu().contiguous(),
                        "name": name}, root / f"{name}.pt")
        if batch_index % 5 == 0:
            print(f"rank={rank} split={args.split} cached={min((batch_index + 1) * config['cache_batch_size'], len(dataset))}/{len(dataset)}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
