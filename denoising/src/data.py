"""Native seismic crops for latent denoising training."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_cube(record):
    path = Path(record['path'])
    suffix = path.suffix.lower()
    if suffix == '.npy':
        value = np.load(path, allow_pickle=False)
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            value = archive['seis'] if 'seis' in archive else archive[archive.files[0]]
    elif suffix == '.dat':
        # Stored crossline,inline,time -> internal time,crossline,inline.
        value = np.fromfile(path, np.float32).reshape(record['shape']).transpose(2, 0, 1)
    elif suffix in {'.sgy', '.segy'}:
        from cigsegy import SegyNP
        value = SegyNP(str(path)).to_numpy().transpose(2, 0, 1)
    else:
        raise ValueError(f'unsupported volume: {path}')
    value = np.asarray(value, dtype=np.float32).squeeze()
    if value.ndim != 3:
        raise ValueError(f'{path}: expected 3D cube, got {value.shape}')
    return value


def crop(value, shape, rng):
    if any(n < size for n, size in zip(value.shape, shape)):
        raise ValueError(f'native crop {shape} does not fit source shape {value.shape}')
    starts = [rng.randint(0, n - size) for n, size in zip(value.shape, shape)]
    return value[tuple(slice(start, start + size) for start, size in zip(starts, shape))]


class CubeDataset(Dataset):
    def __init__(self, manifest, split, crop_shape, seed, samples_per_volume=1):
        rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        groups = {}
        for row in rows:
            groups.setdefault(row['source_id'], set()).add(row['split'])
        if any(len(splits) > 1 for splits in groups.values()):
            raise ValueError('source_id crosses data splits')
        self.rows = [row for row in rows if row['split'] == split]
        if not self.rows:
            raise RuntimeError(f'no {split} data in {manifest}')
        self.shape = tuple(crop_shape)
        self.seed = int(seed)
        self.epoch = 0
        self.training = split == 'train'
        self.samples_per_volume = int(samples_per_volume)

    def set_epoch(self, epoch):
        self.epoch = int(epoch) if self.training else 0

    def __len__(self):
        return len(self.rows) * self.samples_per_volume

    def __getitem__(self, index):
        row = self.rows[index % len(self.rows)]
        seed = self.seed + index + self.epoch * 1_000_003
        value = crop(load_cube(row), self.shape, random.Random(seed))
        origin = ','.join(str(v) for v in row.get('origin_txy', ()))
        sample_id = (f"{row['source_id']}|{Path(row['path']).name}|{origin}"
                     f"|repeat={index // len(self.rows)}")
        return torch.from_numpy(np.ascontiguousarray(value)), sample_id
