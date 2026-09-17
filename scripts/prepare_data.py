"""Prepare non-overlapping field cubes and the VAE's five-category manifest."""
import argparse,json
from pathlib import Path
import numpy as np

def field(a):
    x=np.load(a.input,mmap_mode='r',allow_pickle=False)
    if x.ndim!=3:raise ValueError('input must use crossline,inline,time axes')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for i in range(0,x.shape[0]-a.edge+1,a.edge):
        for j in range(0,x.shape[1]-a.edge+1,a.edge):
            for k in range(0,x.shape[2]-a.edge+1,a.edge):
                cube=np.asarray(x[i:i+a.edge,j:j+a.edge,k:k+a.edge])
                if not np.isfinite(cube).all() or np.ptp(cube)==0 or np.any(np.all(cube==0,axis=-1)):continue
                name=f'{a.source_id}__{i}_{j}_{k}.npy';np.save(out/name,cube)
                records.append(dict(path=name,category='field_seismic',source='field',shape=[a.edge]*3,
                    source_id=a.source_id,split=a.split,origin=[i,j,k]))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(f'Wrote {len(records)} valid non-overlapping cubes; incomplete edge regions excluded.')

def manifest(a):
    rows=[];output=Path(a.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    import os
    def entry(path,category,source,shape,source_id,key=None):
        r=dict(path=os.path.relpath(path.resolve(),output.parent),category=category,source=source,
            shape=list(shape),source_id=source_id,split=a.split)
        if key:r['key']=key
        rows.append(r)
    for p in sorted(Path(a.paired_root).glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            for category,key in [('synthetic_seismic',a.seismic_key),('impedance',a.impedance_key),('rgt',a.rgt_key),('fault',a.fault_key)]:
                if z[key].shape!=(512,)*3:raise ValueError(f'{p}:{key} must be 512^3')
                entry(p,category,'paired_synthetic',z[key].shape,f'synthetic/{p.stem}',key)
    if a.channel_root:
        for p in sorted(Path(a.channel_root).glob('*.npy')):
            shape=np.load(p,mmap_mode='r',allow_pickle=False).shape
            if shape!=(256,)*3:raise ValueError(f'{p} must be 256^3')
            entry(p,'synthetic_seismic','channel_synthetic',shape,f'channel/{p.stem}')
    for f in a.field_manifests:
        f=Path(f)
        for line in f.read_text().splitlines():
            r=json.loads(line);r['path']=os.path.relpath((f.parent/r['path']).resolve(),output.parent)
            rows.append(r)
    if a.append and output.exists():rows=[json.loads(line) for line in output.read_text().splitlines() if line]+rows
    groups={}
    for r in rows:
        groups.setdefault(r['source_id'],set()).add(r['split'])
    if any(len(s)>1 for s in groups.values()):raise ValueError('a source or survey crosses data splits')
    output.write_text(''.join(json.dumps(r)+'\n' for r in rows));print(f'Wrote {len(rows)} records')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('field');f.add_argument('--input',required=True);f.add_argument('--output',required=True)
    f.add_argument('--source-id',required=True);f.add_argument('--edge',type=int,choices=[256,384],default=256)
    f.add_argument('--split',choices=['train','val','test'],default='train');f.set_defaults(func=field)
    m=sub.add_parser('manifest');m.add_argument('--paired-root',required=True);m.add_argument('--channel-root')
    m.add_argument('--field-manifests',nargs='*',default=[]);m.add_argument('--output',required=True);m.add_argument('--append',action='store_true')
    m.add_argument('--split',choices=['train','val','test'],default='train')
    for key,default in [('seismic','seis'),('impedance','imp'),('rgt','rgt'),('fault','fault')]:m.add_argument(f'--{key}-key',default=default)
    m.set_defaults(func=manifest);a=p.parse_args();a.func(a)
if __name__=='__main__':main()
