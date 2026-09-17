"""Generate seismic realizations from one RGT cube using EMA DDIM weights."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import yaml
from .own_conditional_diffusion3d import ConditionalDDPM,ConditionalUNet3D
from .train_fx_rgt_to_sx_ldm import rgt_normalize,to_vae,from_vae,latent_standardizer,standardize,destandardize,ddim_sample
from vae.wan_video_vae import WanVideoVAE

@torch.inference_mode()
def generate(config,checkpoint,rgt,seeds,device):
    device=torch.device(device);dtype=torch.float16 if device.type=='cuda' else torch.float32
    if rgt.ndim!=3 or not np.isfinite(rgt).all(): raise ValueError('RGT must be a finite 3D array')
    vae=WanVideoVAE().to(device,dtype).eval()
    vae.load_state_dict(torch.load(config['vae_checkpoint'],map_location='cpu',weights_only=True),strict=True)
    net=ConditionalUNet3D(width=config['width'],blocks_per_scale=config['res_blocks'],condition_channels=16,dropout=config.get('dropout',0))
    diffusion=ConditionalDDPM(net,timesteps=config['timesteps'],prediction_type=config['prediction_type']).to(device).eval()
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state=weights['ema'] if weights.get('ema') else weights.get('model',weights)
    diffusion.load_state_dict(state,strict=True);del state,weights
    stats=latent_standardizer(config['latent_stats_path'],device)
    x=torch.from_numpy(np.ascontiguousarray(rgt,dtype=np.float32))[None].to(device)
    z=vae.encode(to_vae(rgt_normalize(x)).to(dtype),device=device,tiled=config['vae_tiled']).to(device).float()
    condition=standardize(z/config['latent_scale'],'rgt',stats)
    samples=[]
    for seed in seeds:
        torch.manual_seed(seed)
        prediction=ddim_sample(diffusion,z.shape,condition,config['sample_steps'],config['sample_x0_clip'])
        latent=destandardize(prediction,'sx',stats)*config['latent_scale']
        raw=from_vae(vae.decode(latent.to(dtype),device=device,tiled=config['vae_tiled']),rgt.shape)
        samples.append(raw[0].float().cpu().numpy())
    return np.stack(samples)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/rgt.yaml');p.add_argument('--checkpoint',default='checkpoints/rgt_diffusion.pt')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3]);p.add_argument('--device',default='cuda')
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text())
    result=generate(cfg,a.checkpoint,np.load(a.input,allow_pickle=False),a.seeds,a.device)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,samples=result,seeds=np.asarray(a.seeds),steps=cfg['sample_steps'])
if __name__=='__main__':main()
