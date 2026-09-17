"""Train the latent mapper through a frozen Geo-VAE with decoded-domain L1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import CubeDataset
from .ema import ModelEMA
from .model import build_mapper
from .noise import add_noise
from .pipeline import checkpoint_state, forward_denoise, load_vae
from .vae_io import normalize_amplitude, robust_normalize


def save_jsonl(path, record):
    with path.open('a') as handle:
        handle.write(json.dumps(record) + '\n')


def corrupt(clean, config, epoch, ids):
    """Replayable noise/identity examples with a distinct seed for each crop."""
    inputs = []
    corruption = config['corruption']
    for index, source_id in enumerate(ids):
        offset = int(hashlib.sha256(str(source_id).encode()).hexdigest()[:8], 16)
        seed = (int(config['seed']) + epoch * 100003 + index * 7919 + offset) % (2**32)
        chooser = random.Random(seed)
        value = clean[index].numpy()
        if chooser.random() < float(corruption.get('identity_probability', 0.0)):
            noisy = value.copy()
        else:
            family = chooser.choice(corruption['train_families'])
            snr = float(chooser.choice(corruption['input_snr_db']))
            noisy, _ = add_noise(value, family, snr, seed)
        inputs.append(noisy)
    return torch.from_numpy(np.stack(inputs))


def run(config, run_id, resume=None, init=None, reset_best=False, device='cuda'):
    if resume and init:
        raise ValueError('resume and init are mutually exclusive')
    if config.get('loss', {}).get('mode', 'raw_l1') != 'raw_l1':
        raise ValueError('latent denoising uses decoded-domain raw_l1 loss')
    if config.get('task', 'denoise') != 'denoise':
        raise ValueError('this trainer only supports denoising')
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        dist.init_process_group('nccl')
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank) if distributed else torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    is_main = rank == 0
    torch.manual_seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    random.seed(config['seed'] + rank)

    output = Path(config['output_root']) / run_id
    if is_main:
        output.mkdir(parents=True, exist_ok=bool(resume))
        (output / 'config.yaml').write_text(yaml.safe_dump(config))
    if distributed:
        dist.barrier()

    datasets = {
        split: CubeDataset(
            config['manifest'], split, config['crop_shape'], config['seed'],
            samples_per_volume=config.get('samples_per_volume', 1) if split == 'train' else 1,
        )
        for split in ('train', 'val')
    }
    samplers = {
        split: DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=split == 'train')
        if distributed else None
        for split, dataset in datasets.items()
    }
    loaders = {
        split: DataLoader(
            dataset, batch_size=config['batch_size'],
            shuffle=split == 'train' and not distributed,
            sampler=samplers[split], num_workers=config['num_workers'],
            pin_memory=device.type == 'cuda',
        )
        for split, dataset in datasets.items()
    }
    vae = load_vae(config['vae_checkpoint'], device)
    model = build_mapper(config).to(device)
    ema_config = config.get('ema', {})
    ema = ModelEMA(model, ema_config.get('decay', .999)) if ema_config.get('enabled', True) else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=float(config.get('weight_decay', 0.0)))
    amp_config = config.get('amp', {})
    amp_enabled = bool(amp_config.get('enabled', False)) and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_config.get('dtype', 'bfloat16') == 'bfloat16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled and amp_dtype == torch.float16)
    best, start_epoch, global_step = float('inf'), 0, 0

    if resume or init:
        state = torch.load(resume or init, map_location=device, weights_only=False)
        model.load_state_dict(state['model'] if resume else checkpoint_state(state), strict=True)
        if ema is not None:
            if state.get('ema'):
                ema.load_state_dict(state['ema'])
            else:
                ema.model.load_state_dict(model.state_dict())
        if resume:
            optimizer.load_state_dict(state['opt'])
            start_epoch = int(state['epoch']) + 1
            global_step = int(state.get('global_step', 0))
            best = float('inf') if reset_best else float(state.get('val_raw_fidelity_loss', float('inf')))
            if 'scaler' in state:
                scaler.load_state_dict(state['scaler'])

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    def checkpoint(loss):
        source = model.module if distributed else model
        return dict(model=source.state_dict(), ema=None if ema is None else ema.state_dict(),
                    opt=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                    global_step=global_step, val_raw_fidelity_loss=loss, config=config)

    accumulation = max(1, int(config.get('grad_accum_steps', 1)))
    loss_weight = float(config.get('loss', {}).get('waveform', 1.0))
    for epoch in range(start_epoch, config['epochs']):
        for split in ('train', 'val'):
            training = split == 'train'
            datasets[split].set_epoch(epoch if training else 0)
            if samplers[split] is not None:
                samplers[split].set_epoch(epoch if training else 0)
            model.train(training)
            active = model if training or ema is None else ema.model.eval()
            loss_sum, sample_count = 0.0, 0
            with torch.set_grad_enabled(training):
                for batch_index, (clean, ids) in enumerate(loaders[split]):
                    noisy = corrupt(clean.float(), config, epoch if training else 0, ids)
                    clean, noisy = clean.to(device), noisy.to(device)
                    normalized, centre, scale = robust_normalize(noisy)
                    target = normalize_amplitude(clean, centre, scale)
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                        prediction = forward_denoise(
                            vae, active, normalized, device,
                            encode_config=config.get('vae_encode'),
                            decode_config=config.get('vae_decode'),
                        )
                    loss = loss_weight * F.l1_loss(prediction, target.float())
                    if training:
                        if batch_index % accumulation == 0:
                            optimizer.zero_grad(set_to_none=True)
                        # The final, shorter accumulation group keeps its full weight.
                        group_start = batch_index // accumulation * accumulation
                        group_size = min(accumulation, len(loaders[split]) - group_start)
                        scaler.scale(loss / group_size).backward()
                        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loaders[split]):
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            global_step += 1
                            if ema is not None:
                                ema.update(model.module if distributed else model)
                            if is_main and global_step % config.get('log_every_steps', 20) == 0:
                                save_jsonl(output / 'train_steps.jsonl', dict(
                                    epoch=epoch, global_step=global_step, loss=float(loss.detach()),
                                    grad_norm=float(grad_norm), lr=optimizer.param_groups[0]['lr'],
                                    effective_batch_size=config['batch_size'] * group_size * world,
                                    time=time.time(),
                                ))
                    loss_sum += float(loss.detach()) * clean.shape[0]
                    sample_count += clean.shape[0]
            totals = torch.tensor([loss_sum, sample_count], device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(totals)
            mean_loss = float((totals[0] / totals[1]).item())
            if is_main:
                print(f'{epoch:03d} {split} decoded_l1={mean_loss:.6f}', flush=True)
                save_jsonl(output / 'history.jsonl', dict(epoch=epoch, split=split, loss=mean_loss))
                if not training and mean_loss < best:
                    best = mean_loss
                    torch.save(checkpoint(best), output / 'best.pt')
        if is_main:
            torch.save(checkpoint(best), output / 'last.pt')
    if distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--resume')
    source.add_argument('--init')
    parser.add_argument('--reset-best', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    run(config, args.run_id, args.resume, args.init, args.reset_best, args.device)


if __name__ == '__main__':
    main()
