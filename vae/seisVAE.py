"""Compatibility imports for the restored causal model."""
from .model import SeismicVAE3D
from .inference import SeismicVAEInference
SeisVAE = SeismicVAEInference

def load_seisvae(checkpoint,device='cuda',**kwargs):
    return SeismicVAEInference(checkpoint,device=device,**kwargs)
