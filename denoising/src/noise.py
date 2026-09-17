"""Controlled noise families.  Each corruption is stored and replayable by seed."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

def _scale(noise: np.ndarray, clean: np.ndarray, snr_db: float) -> np.ndarray:
    rms = np.sqrt(np.mean(clean**2)) + 1e-8
    return noise / (np.sqrt(np.mean(noise**2)) + 1e-8) * rms / (10 ** (snr_db / 20))

def add_noise(clean: np.ndarray, family: str, snr_db: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if family == 'white': noise = rng.normal(size=clean.shape)
    elif family == 'band_limited': noise = gaussian_filter(rng.normal(size=clean.shape), sigma=(1.0, .4, .4))
    elif family == 'coherent_linear':
        d,h,w = clean.shape; t=np.arange(d)[:,None,None]; x=np.arange(w)[None,None,:]
        noise = np.sin(2*np.pi*(t/17 + x/43) + rng.uniform(0,2*np.pi)) * rng.normal(1, .15, (1,h,1))
    elif family == 'mixed':
        _,a=add_noise(clean, 'white', 0, seed)
        _,b=add_noise(clean, 'coherent_linear', 0, seed+1)
        noise=.6*a+.4*b
    else: raise ValueError(f'unknown noise family {family}')
    noise = _scale(noise.astype(np.float32), clean, snr_db)
    return (clean + noise).astype(np.float32), noise.astype(np.float32)
