"""Paper data protocol: equal-category draws and source-dependent crop sizes.

JSONL rows: path, category, source, shape, split, source_id; optional key (.npz).
Arrays use (crossline, inline, time/depth). No on-the-fly geology synthesis.
"""
from pathlib import Path
import json
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

CATEGORIES = ('synthetic_seismic','impedance','rgt','fault','field_seismic')
PERMUTATIONS = tuple(itertools.permutations(range(3)))


def normalize_cube(data, category):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError('expected a finite 3D cube')
    if category in ('synthetic_seismic','field_seismic'):
        return np.clip((data-data.mean())/(data.std()+1e-6),-3.2,3.2)/3.2
    if category not in CATEGORIES:
        raise ValueError(f'unknown category: {category}')
    low,high = data.min(),data.max()
    return 2*(data-low)/(high-low+1e-6)-1


def load_volume(row):
    path=Path(row['path'])
    if path.suffix=='.npy':
        data=np.load(path,mmap_mode='r',allow_pickle=False)
    elif path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as pack:
            data=pack[row['key']]
    elif path.suffix=='.dat':
        data=np.memmap(path,dtype=row.get('dtype','float32'),mode='r',shape=tuple(row['shape']))
    else:
        raise ValueError(f'unsupported array format: {path}')
    if tuple(data.shape)!=tuple(row['shape']):
        raise ValueError(f'shape mismatch: {path}: {data.shape} != {row["shape"]}')
    return data


class GeoDiffusionDataset(Dataset):
    def __init__(self, manifest, split='train', augment=True):
        manifest=Path(manifest).resolve()
        rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
        # Group identity, not filename, is the leakage boundary.
        groups={}
        for row in rows:
            key=row['source_id']; group=groups.setdefault(key,set()); group.add(row['split'])
            if len(group)>1: raise ValueError(f'source_id crosses splits: {key}')
        self.rows=[r for r in rows if r['split']==split]
        self.augment=augment and split=='train'
        self.groups={k:[] for k in CATEGORIES}
        for i,row in enumerate(self.rows):
            if row['category'] not in CATEGORIES: raise ValueError('invalid category')
            source=row['source']; shape=tuple(row['shape'])
            if source=='paired_synthetic':
                if shape!=(512,512,512) or row['category']=='field_seismic':
                    raise ValueError('paired synthetic volumes must be 512^3')
            elif source=='channel_synthetic':
                if shape!=(256,256,256) or row['category']!='synthetic_seismic':
                    raise ValueError('channel samples must be 256^3 synthetic seismic')
            elif source=='field':
                if shape not in ((256,)*3,(384,)*3) or row['category']!='field_seismic':
                    raise ValueError('field training inputs must be prebuilt 256^3 or 384^3 cubes')
            else: raise ValueError(f'unknown source type: {source}')
            if not Path(row['path']).is_absolute(): row['path']=str(manifest.parent/row['path'])
            self.groups[row['category']].append(i)
        if split=='train' and any(not x for x in self.groups.values()):
            raise ValueError('all five categories are required for equal-probability training')

    def __len__(self): return len(self.rows)

    def __getitem__(self, selection):
        if isinstance(selection,int):
            row=self.rows[selection]; selection=(selection,256 if row['source']=='paired_synthetic' else row['shape'][0],selection)
        index,edge,seed=selection
        row=self.rows[index]; data=load_volume(row); rng=np.random.default_rng(seed)
        if row['source']=='paired_synthetic':
            starts=[int(rng.integers(0,n-edge+1)) for n in data.shape]
            data=data[tuple(slice(s,s+edge) for s in starts)]
        elif tuple(data.shape)!=(edge,)*3:
            raise ValueError('field/channel cubes must not be recropped or resized')
        data=normalize_cube(data,row['category'])
        permutation=PERMUTATIONS[int(rng.integers(6))] if self.augment else (0,1,2)
        data=np.ascontiguousarray(data.transpose(permutation))
        return {'img':torch.from_numpy(data).unsqueeze(0).repeat(3,1,1,1),
                'data_type':row['category'],'source_id':row['source_id']}


class EqualCategoryBatchSampler(Sampler):
    """Draw categories uniformly, then bucket by native/crop size for stacking.

    Each rank consumes disjoint batches from the same seeded global stream.
    Bucketing changes order only; it never crops or pads a field cube.
    """
    def __init__(self,dataset,batch_size=5,batches_per_epoch=1000,seed=0,rank=0,world_size=1):
        self.dataset=dataset; self.batch_size=batch_size; self.batches=batches_per_epoch
        self.seed=seed; self.epoch=0; self.rank=rank; self.world_size=world_size
    def __len__(self): return self.batches
    def set_epoch(self,epoch): self.epoch=epoch
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch*1000003)
        queues={256:[],384:[]}; produced=0
        while produced < self.batches*self.world_size:
            category=CATEGORIES[int(rng.integers(5))]
            index=int(rng.choice(self.dataset.groups[category])); row=self.dataset.rows[index]
            edge=int(rng.choice((256,384))) if row['source']=='paired_synthetic' else row['shape'][0]
            queue=queues[edge]; queue.append((index,edge,int(rng.integers(2**32))))
            if len(queue)==self.batch_size:
                if produced % self.world_size==self.rank: yield queue.copy()
                queue.clear(); produced+=1
