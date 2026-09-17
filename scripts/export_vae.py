"""Export EMA causal VAE weights from a Lightning checkpoint for inference."""
import argparse
from pathlib import Path
import torch
from vae.model import SeismicVAE3D
from vae.inference import load_backbone_weights

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();m=SeismicVAE3D();load_backbone_weights(m,a.input,prefer_ema=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model.'+k:v for k,v in m.state_dict().items()},a.output)
if __name__=='__main__':main()
