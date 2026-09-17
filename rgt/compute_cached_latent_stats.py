"""Compute per-channel train-set moments for cached GeoVAE latent triplets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
        local_rank, rank = int(os.environ["LOCAL_RANK"]), dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, device = 0, torch.device("cuda")
    paths = sorted((Path(args.cache) / "train").glob("*.pt"))[rank::world]
    if not paths:
        raise RuntimeError("no cache shards assigned")
    sums = {key: torch.zeros(16, device=device, dtype=torch.float64) for key in ("sx", "fx", "rgt")}
    squares = {key: torch.zeros(16, device=device, dtype=torch.float64) for key in sums}
    count = torch.zeros(1, device=device, dtype=torch.float64)
    for index, path in enumerate(paths):
        item = torch.load(path, map_location="cpu", weights_only=True)
        for key in sums:
            value = item[key].to(device=device, dtype=torch.float32)
            sums[key] += value.sum(dim=(1, 2, 3), dtype=torch.float64)
            squares[key] += value.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        count += item["sx"].numel() // item["sx"].shape[0]
        if index % 250 == 0:
            print(f"rank={rank} files={index + 1}/{len(paths)}", flush=True)
    if world > 1:
        for value in [*sums.values(), *squares.values(), count]:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if rank == 0:
        output = {}
        for key in sums:
            mean = sums[key] / count
            std = (squares[key] / count - mean.square()).clamp_min(1e-12).sqrt()
            output[key] = {"mean": mean.float().cpu(), "std": std.float().cpu()}
            print(f"{key}: mean={mean.mean():.5f} avg_std={std.mean():.5f}", flush=True)
        torch.save(output, args.output)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
