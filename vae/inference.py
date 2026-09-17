"""Causal Geo-VAE weight loading and paper reconstruction interface."""
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from .model import SeismicVAE3D
from .dataset.vae_dataset import normalize_cube,CATEGORIES


def causal_resize(x,factor=8):
    """Interpolate pseudo-time to the smallest valid 1+n*r length."""
    t,h,w=x.shape[-3:]
    if h%8 or w%8: raise ValueError('inline and time/depth lengths must be divisible by 8')
    target=1+((t-1+factor-1)//factor)*factor
    if target==t: return x
    return F.interpolate(x,size=(target,h,w),mode='trilinear',align_corners=False)


def backbone_state(checkpoint,prefer_ema=True):
    sd=checkpoint
    if prefer_ema and isinstance(sd,dict) and 'ema_state_dict' in sd: sd=sd['ema_state_dict']
    elif isinstance(sd,dict):
        for key in ('state_dict','model_state','model'):
            if key in sd and isinstance(sd[key],dict): sd=sd[key]; break
    result={}
    for key,value in sd.items():
        if not isinstance(value,torch.Tensor): continue
        for prefix in ('module.','vae.vae.','vae.','model.'):
            if key.startswith(prefix): key=key[len(prefix):]
        if key.startswith(('encoder.','decoder.','conv1.','conv2.')): result[key]=value
    if not result: raise ValueError('no recognizable causal VAE backbone weights')
    return result


def load_backbone_weights(model,path,allow_wan=False,prefer_ema=True):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    state=backbone_state(ckpt,prefer_ema)
    expected=model.state_dict()
    wrong={k:(tuple(v.shape),tuple(expected[k].shape)) for k,v in state.items()
           if k in expected and v.shape!=expected[k].shape}
    missing=set(expected)-set(state); extra=set(state)-set(expected)
    allowed=set(model.added_temporal_parameters()) if allow_wan else set()
    if wrong or extra or missing-allowed:
        raise ValueError(f'incompatible checkpoint: shapes={wrong}, missing={sorted(missing)}, unexpected={sorted(extra)}. Noncausal weights cannot be folded into this model.')
    model.load_state_dict(state,strict=not allow_wan)
    return {'loaded':len(state),'new_temporal_parameters':sorted(missing)}


class SeismicVAEInference:
    def __init__(self,checkpoint,device='cuda',dtype=torch.float32,**model_kwargs):
        self.device=torch.device(device); self.dtype=dtype
        self.model=SeismicVAE3D(**model_kwargs)
        self.load_report=load_backbone_weights(self.model,checkpoint)
        self.model.to(self.device,dtype).eval().requires_grad_(False)
    @torch.no_grad()
    def encode(self,cube,category='synthetic_seismic'):
        value=normalize_cube(cube,category)
        x=torch.from_numpy(np.ascontiguousarray(value)).to(self.device,self.dtype)[None,None].repeat(1,3,1,1,1)
        mu,_=self.model.encode(causal_resize(x,self.model.temporal_factor))
        return mu
    @torch.no_grad()
    def decode(self,z,shape):
        value=self.model.decode(z.to(self.device,self.dtype)).mean(1,keepdim=True)
        if tuple(value.shape[-3:])!=tuple(shape):
            value=F.interpolate(value,size=shape,mode='trilinear',align_corners=False)
        return value[0,0].float().cpu().numpy()
    @torch.no_grad()
    def reconstruct(self,cube,category='synthetic_seismic'):
        return self.decode(self.encode(cube,category),cube.shape)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--input',required=True)
    p.add_argument('--output',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--category',choices=CATEGORIES,default='synthetic_seismic')
    a=p.parse_args(); x=np.load(a.input,allow_pickle=False)
    api=SeismicVAEInference(a.checkpoint,a.device)
    result=api.reconstruct(x,a.category)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); np.save(a.output,result)

if __name__=='__main__': main()
