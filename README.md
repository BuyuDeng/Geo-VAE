# Geo-VAE

Training and inference code for **Geo-VAE: A Unified Variational Autoencoder for 3D Geophysical Data Compression and Latent Processing**.

The repository provides three workflows:

- Geo-VAE training and reconstruction of 3D geophysical data.
- RGT-conditioned latent diffusion for seismic generation.
- Seismic denoising through a frozen Geo-VAE encoder, a latent mapper, and the frozen Geo-VAE decoder.

## Data availability

The original synthetic datasets used in the manuscript are not publicly available. The field data originate from public repositories such as NLOG and USGS. CIG-Bench is an alternative public training resource, not the original training dataset. See [data availability and preparation](data/README.md) for links, scope, and input requirements.

## Install

Use Python 3.10 or newer. Full-size training and inference require a CUDA-enabled PyTorch installation.

```bash
pip install -e .
```

Run the commands below from the repository root. SEG-Y input additionally requires `pip install cigsegy`; NumPy input does not. See [data preparation](data/README.md) for array layouts and manifests.

## Model weights

This code release does not include the pretrained weights used for the manuscript results. Inference examples below expect the following files in `checkpoints/`; train the corresponding models and export their weights before running those examples.

| File in `checkpoints/` | Model |
| --- | --- |
| `geovae.ckpt` | Causal Geo-VAE, 8×8×8 downsampling, 16 latent channels |
| `rgt_diffusion.pt` | RGT-conditioned diffusion, 147.834M parameters, tensor-only EMA state |
| `rgt_latent_stats.pt` | Per-channel SX and RGT means and standard deviations required for inference |
| `denoise_latent.pt` | Latent residual mapper, 2.076M parameters, tensor-only EMA weights |

Training checkpoints are saved separately by the trainers and are required for `--resume`. The exported inference files need the model weights, buffers, and normalization statistics from a matching training run.

## Geo-VAE training

The VAE shares one encoder and decoder across `synthetic_seismic`, `field_seismic`, `impedance`, `rgt`, and `fault`. Training samples these five categories with equal probability. Paired 512³ synthetic volumes receive random 256³ or 384³ crops; additional channel-containing 256³ seismic volumes and prebuilt 256³/384³ field cubes retain their native sizes. Shape buckets allow these sizes to coexist with batch size 5. Training uniformly samples all six axis permutations.

Use two explicit stages and supply the number of updates for each:

```bash
python -m vae.train --manifest data/vae_manifest.jsonl --stage warmup \
  --init /path/to/Wan2.1_VAE.pth --max-steps "$WARMUP_STEPS" \
  --output-dir outputs/vae_warmup

python -m vae.train --manifest data/vae_manifest.jsonl --stage full \
  --init outputs/vae_warmup/checkpoints/last.ckpt --max-steps "$FINETUNE_STEPS" \
  --output-dir outputs/vae_full
```

Warm-up optimizes only the added encoder/decoder temporal operators. Full fine-tuning optimizes the complete VAE. Defaults: AdamW, fixed learning rate `1e-5`, batch 5, EMA, L1 weight 1, slice LPIPS weight 1, and KL weight `1e-4`. The relativistic least-squares 3D PatchGAN starts after 100,000 VAE updates in the full stage and multiplies its adaptive gradient-norm weight by 0.5. There is no learning-rate scheduler. `--resume` restores a checkpoint from the same stage.

Export the EMA VAE weights for inference and downstream training:

```bash
python scripts/export_vae.py --input outputs/vae_full/checkpoints/last.ckpt \
  --output outputs/geovae_ema.ckpt
```

## Geo-VAE inference

Public NumPy inputs use **(crossline, inline, time/depth)**. Single-channel input is copied to RGB; decoded channels are averaged. Crossline is the default pseudo-temporal axis. Its length is resized to `1+8n` before encoding, and the reconstruction is resized back to the input shape. A 256³ cube yields a `16×33×32×32` latent; a 768³ cube yields `16×97×96×96`.

```bash
python -m vae.inference --checkpoint checkpoints/geovae.ckpt \
  --input data/seismic.npy --output outputs/reconstruction.npy \
  --category synthetic_seismic
```

Seismic normalization is `clip((x-mean)/(std+1e-6), -3.2, 3.2)/3.2`. Impedance, RGT, and fault volumes use `2*(x-min)/(max-min+1e-6)-1`. Reconstruction output is in this normalized domain.

## RGT-conditioned seismic training

Supply paired seismic/RGT files with matching names. The RGT-only configuration does not require fault inputs. Four crops per training source give 3,800 training crops from 950 sources; 50 validation sources give 50 fixed crops. The trainer requires these configured totals.

```bash
python -m rgt.cache_fx_rgt_sx_latents --config configs/rgt.yaml --split train
python -m rgt.cache_fx_rgt_sx_latents --config configs/rgt.yaml --split valid
python -m rgt.compute_cached_latent_stats \
  --cache data/rgt_latents --output data/rgt_latents/stats.pt
```

For a new training cache, set `latent_stats_path` in `configs/rgt.yaml` to `data/rgt_latents/stats.pt`, then train:

```bash
python -m rgt.train_fx_rgt_to_sx_ldm --config configs/rgt.yaml --run-id rgt
```

Defaults: width 192, four residual blocks per scale, batch 16, AdamW at `5e-5`, cosine decay, EMA, and velocity prediction. Training saves `best.pt` and `last.pt` using validation loss. The original training module names are retained.

## RGT-conditioned seismic inference

```bash
python -m rgt.infer --config configs/rgt.yaml \
  --checkpoint checkpoints/rgt_diffusion.pt \
  --input data/rgt.npy --output outputs/rgt_samples.npz --seeds 0 1 2 3
```

Use the VAE, diffusion weights, and latent statistics from the same training run. Set the latent-statistics path in `configs/rgt.yaml` to the file produced by that run. Inference uses EMA weights and 50-step DDIM. `samples` in the NPZ has shape `(realization, crossline, inline, time/depth)` in the normalized seismic domain.

## Latent denoising training

The denoising path is:

```text
noisy seismic -> frozen Geo-VAE encoder -> latent mapper -> frozen Geo-VAE decoder
```

The mapper receives the complete 16-channel latent volume. Training minimizes L1 between decoded seismic and the clean target. Encoder weights are frozen; decoder weights are frozen while gradients pass through decoding to the mapper. Only noisy input is encoded.

```bash
python -m denoising.src.train --config configs/denoise_latent.yaml --run-id latent
```

The training configuration uses 256³ crops, AdamW at `1e-5`, 60 epochs, BF16, gradient checkpointing, and EMA. White, band-limited, coherent-linear, and mixed noise use 0, 3, 6, 9, and 12 dB, with 30% clean identity examples. Normalization statistics come from the noisy input and are reused for the clean target. `--resume` restores training state; `--init` initializes mapper weights for a new run. Validation L1 selects `best.pt`; each epoch also saves `last.pt`.

Both `best.pt` and `last.pt` contain model and EMA weights, optimizer and AMP scaler state, configuration, training progress, and the best validation loss. Training does not automatically export a separate inference-only file. To use a training checkpoint directly for inference, pass `--checkpoint outputs/denoising/<run-id>/best.pt`; the loader selects its EMA weights when available.

## Latent denoising inference

```bash
python -m denoising.infer --input data/field_768.npy \
  --output outputs/field_denoised.npy
```

The entry point loads `checkpoints/geovae.ckpt` and `checkpoints/denoise_latent.pt`. Override the mapper with `--checkpoint` and its architecture/VAE configuration with `--model-config`.

An exported `denoise_latent.pt` containing EMA parameter tensors can be used for inference and training with `--init`; use a full training checkpoint for `--resume`.

Public input and output use **(crossline, inline, time/depth)**; dimensions must be divisible by 8. The causal VAE uses spatial tiling and pseudo-temporal caching, and the latent mapper runs once on the complete encoded volume. `configs/denoise_inference.yaml` controls VAE tiling and the CUDA allocator budget, defaulting to 30 GiB. Output preserves the input axis order and restores amplitude units; normalization clipping remains lossy. A companion JSON records shape, checkpoint epoch, and normalization statistics. Small-volume inference can use `--device cpu`.

License: Apache-2.0. See [third-party notices](THIRD_PARTY_NOTICES.md).
