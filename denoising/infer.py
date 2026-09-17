"""Denoise a seismic volume with VAE encoding, one latent mapper call, and decoding."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .src.pipeline import load_models, predict_normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='NPY: crossline,inline,time/depth')
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/denoise_inference.yaml')
    parser.add_argument('--model-config', default='configs/denoise_latent.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/denoise_latent.pt')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    model_config = yaml.safe_load(Path(args.model_config).read_text())
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(config['memory_budget_gib'] * 2**30 / total, 1.0), device)

    physical = np.load(args.input, mmap_mode='r', allow_pickle=False)
    if physical.ndim != 3 or any(n <= 0 or n % 8 for n in physical.shape):
        raise ValueError('expected a nonempty 3D volume with dimensions divisible by 8')
    if not np.isfinite(physical).all():
        raise ValueError('nonfinite input')
    # Public crossline,inline,time -> internal time,crossline,inline.
    value = np.ascontiguousarray(physical.transpose(2, 0, 1), dtype=np.float32)
    centre = float(value.mean())
    scale = (float(value.std()) + 1e-6) * 3.2
    normalized = np.clip((value - centre) / scale, -1, 1).astype(np.float32)
    vae, mapper, epoch = load_models(model_config, args.checkpoint, device)
    prediction = predict_normalized(normalized, vae, mapper, device, config)
    restored = (prediction * scale + centre).transpose(1, 2, 0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, restored)
    output.with_suffix('.json').write_text(json.dumps(dict(
        axes='crossline_inline_time', shape=list(physical.shape), checkpoint_epoch=epoch,
        normalization=dict(centre=centre, scale=scale),
    ), indent=2))


if __name__ == '__main__':
    main()
