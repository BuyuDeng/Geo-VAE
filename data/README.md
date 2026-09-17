# Data layout

No research volumes are bundled with this code release. Prepare source-level training and validation splits so that each parent survey belongs to only one split.

## Data availability and provenance

The original synthetic datasets used in the manuscript are not publicly available and are not included in this repository. The original training corpus must not be described as CIG-Bench.

The field seismic data used in the study came from publicly accessible online repositories, including [NLOG](https://www.nlog.nl/en/data) and [USGS](https://www.usgs.gov/). These are source portals; they do not identify the exact surveys, files, or train/validation/test partitions used in the manuscript. Obtain field data from their original providers and retain the applicable attribution and use conditions.

[CIG-Bench](https://arxiv.org/abs/2606.09094) is a public alternative for independent training, with synthetic seismic volumes and associated labels including acoustic impedance. It was not used to obtain the results reported in the manuscript. Training with alternative data does not reproduce the original training corpus or ensure the reported numerical results. The linked project paper identifies the public dataset.

CIG-Bench integration has not been validated in this release. Before training, inspect the downloaded arrays and adapt their keys, axis order, dimensions, normalization, and label semantics to the interfaces below. In particular, verify that fault labels meet the instance-label requirement. Do not assume that downloading the alternative dataset produces a ready-to-run manuscript reproduction.

## VAE

All VAE arrays use `(crossline, inline, time/depth)`:

- Paired synthetic: `.npz`, four arrays (`seis`, `imp`, `rgt`, `fault`) of shape 512³. Fault arrays contain integer instance labels.
- Additional channel seismic: `.npy`, shape 256³.
- Field seismic: `.npy`, non-overlapping 256³ or 384³ cubes. Keep the parent survey ID for every cube.

Prepare a field survey without filtering or resampling:

```bash
python scripts/prepare_data.py field --input /path/to/survey.npy \
  --source-id survey_A --edge 256 --split train --output data/field/survey_A
```

The script excludes incomplete boundary cubes, nonfinite cubes, constant cubes and cubes containing empty traces. It records source ID and cube origin. It does not interpolate or normalize at this stage.

```bash
python scripts/prepare_data.py manifest --paired-root /path/to/paired512 \
  --channel-root /path/to/channel256 --field-manifests data/field/survey_A/manifest.jsonl \
  --output data/vae_manifest.jsonl
```

Key names can be supplied using `--seismic-key`, `--impedance-key`, `--rgt-key`, `--fault-key`. Each manifest row includes `path`, `category`, `source`, `shape`, `split`, `source_id`, and `key` for NPZ data. Paths are relative to the manifest. The loader refuses source IDs that occur in multiple splits and requires all five categories during training.

## RGT diffusion

```text
rgt/
  train/seismic/*.npy
  train/rgt/*.npy
  valid/seismic/*.npy
  valid/rgt/*.npy
```

Use matching file names and shapes, with axes `(crossline, inline, time/depth)`. Training uses 950 sources × 4 paired crops; validation uses 50 sources × 1 fixed crop. Source volumes must be at least 256³ along every axis. No raw-resolution condition is given to the diffusion U-Net.

## Denoising

`denoise_manifest.jsonl` rows contain source_id, path, shape, and split (train or val). The latent mapper trainer uses this manifest. NPY arrays in this data-loader interface use **(time/depth, crossline, inline)**; `.dat` arrays are float32 stored as `(crossline, inline, time/depth)` and are transposed by the loader. Paths are relative to the repository root. Inference's public NPY entry point instead accepts the VAE's `(crossline, inline, time/depth)` order and handles the conversion.

Training crops must fit the source volume without padding. Keep all cubes from a parent survey in the same split. Replace the example manifest paths with your local data paths.

Generate a denoising manifest from separate training and validation directories:

```bash
python -m denoising.src.make_manifest --train-root /path/to/train \
  --val-root /path/to/val --output data/denoise_manifest.jsonl
```

Alternatively, use `--root /path/to/volumes --val-fraction 0.15` for a seeded source-level split. Source IDs are inferred from the filename prefix before the first underscore or hyphen; use filenames with correct parent-survey prefixes or edit the resulting IDs. DAT inputs additionally require `--dat-shape CROSSLINE INLINE TIME`.
