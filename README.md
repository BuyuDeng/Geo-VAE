# Geo-VAE: A Unified Variational Autoencoder for 3D Geophysical Data Compression and Latent Processing

[![Status: WIP](https://img.shields.io/badge/Status-Work_in_Progress-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **🚧 Work In Progress:** We are currently cleaning up the source code and organizing the pre-trained weights for public release. Please ⭐ **Star** this repository to stay updated! 

This repository contains the official implementation of **Geo-VAE**, a unified variational autoencoder framework designed to compress diverse 3D geophysical modalities (e.g., seismic data, relative geological time cubes, property models) into a shared, highly compact latent space.

## 🌟 Highlights

* **Unified Representation:** Encodes multiple 3D geophysical data types into a shared continuous latent space.
* **Extreme Compression:** Achieves an **8×8×8** downsampling rate via pseudo-temporal downsampling and 3D causal convolutions, drastically reducing computational and memory costs.
* **Memory-Efficient Inference:** Leverages a feature cache mechanism to process massive 3D seismic volumes chunk-by-chunk without boundary artifacts.
* **Actionable Latent Space:** Directly supports high-efficiency downstream tasks in the compressed domain, such as missing trace interpolation and seismic denoising.



