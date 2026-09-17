"""Prepare source-disjoint training and validation manifests for latent denoising."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def source_id(path):
    return path.stem.split('_')[0].split('-')[0]


def files(root):
    return sorted(path for path in Path(root).rglob('*')
                  if path.is_file() and path.suffix.lower() in {'.npy', '.npz', '.dat'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument('--root')
    roots.add_argument('--train-root')
    parser.add_argument('--val-root')
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20260710)
    parser.add_argument('--dat-shape', nargs=3, type=int)
    args = parser.parse_args()

    if args.root:
        if args.val_root:
            parser.error('--val-root is used with --train-root')
        if not 0 < args.val_fraction < 1:
            parser.error('--val-fraction must lie between 0 and 1')
        paths = files(args.root)
        groups = sorted({source_id(path) for path in paths})
        if len(groups) < 2:
            parser.error('at least two source groups are needed for a training/validation split')
        np.random.default_rng(args.seed).shuffle(groups)
        count = min(len(groups) - 1, max(1, round(len(groups) * args.val_fraction)))
        validation = set(groups[:count])
        selections = [(path, 'val' if source_id(path) in validation else 'train') for path in paths]
    else:
        if not args.val_root:
            parser.error('--train-root requires --val-root')
        training, validation = files(args.train_root), files(args.val_root)
        if not training or not validation:
            parser.error('training and validation directories must both contain volumes')
        if {source_id(p) for p in training} & {source_id(p) for p in validation}:
            parser.error('source IDs overlap between training and validation')
        selections = [(path, 'train') for path in training] + [(path, 'val') for path in validation]

    records = []
    for path, split in selections:
        if path.suffix.lower() == '.dat':
            if not args.dat_shape:
                parser.error('--dat-shape is required for DAT inputs')
            shape = args.dat_shape
        elif path.suffix.lower() == '.npy':
            shape = np.load(path, mmap_mode='r', allow_pickle=False).shape
        else:
            with np.load(path, allow_pickle=False) as archive:
                shape = archive['seis' if 'seis' in archive else archive.files[0]].shape
        if len(shape) != 3:
            raise ValueError(f'{path}: expected a 3D volume')
        records.append(dict(path=str(path.resolve()), source_id=source_id(path),
                            split=split, shape=list(shape)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row) + '\n' for row in records))
    print(f'wrote {len(records)} source-level records to {output}')


if __name__ == '__main__':
    main()
