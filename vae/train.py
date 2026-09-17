"""Train the restored causal Geo-VAE using the existing Lightning module."""
import argparse
from pathlib import Path
import torch
from pytorch_lightning import Trainer, seed_everything, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from .lightning import SeismicVAELightning, LossPrintCallback
from .dataset.vae_dataset import GeoDiffusionDataset, EqualCategoryBatchSampler
from .inference import load_backbone_weights


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--stage',choices=('warmup','full'),required=True)
    init=p.add_mutually_exclusive_group(required=True)
    init.add_argument('--init',help='Wan weights (warmup), Geo-VAE weights (full)')
    init.add_argument('--resume',help='Resume a Lightning checkpoint from the SAME stage')
    p.add_argument('--output-dir',default='outputs/vae')
    p.add_argument('--max-steps',type=int,required=True,help='Explicit stage budget; durations are not specified in the paper')
    p.add_argument('--batch-size',type=int,default=5)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--batches-per-epoch',type=int,default=1000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--devices',type=int,default=1)
    p.add_argument('--precision',default='bf16-mixed')
    p.add_argument('--grad-checkpoint',choices=('none','shallow','full'),default='full')
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--ema-decay',type=float,default=.9999)
    p.add_argument('--save-every',type=int,default=1000)
    return p.parse_args()


class PaperDataModule(LightningDataModule):
    def __init__(self,args):
        super().__init__(); self.args=args
    def train_dataloader(self):
        a=self.args; ds=GeoDiffusionDataset(a.manifest)
        sampler=EqualCategoryBatchSampler(ds,a.batch_size,a.batches_per_epoch,a.seed,
                                         self.trainer.global_rank,self.trainer.world_size)
        return torch.utils.data.DataLoader(ds,batch_sampler=sampler,num_workers=a.num_workers)

def remap_vae_keys(sd):
    """把各种来源的 checkpoint key 映射到本模型的 `vae.vae.*` 命名。

    支持的输入格式：
      1. 本代码库 Lightning ckpt : ``vae.vae.encoder.…``  (原样保留)
      2. Wan 原生权重            : ``model.encoder.…``    -> ``vae.vae.encoder.…``
      3. 裸 VAE state_dict       : ``encoder.…``          -> ``vae.vae.encoder.…``

    Returns:
        (remapped_dict, stats) —— stats 记录各来源命中的数量。
    """
    bare_prefixes = ("encoder.", "decoder.", "conv1.", "conv2.")
    out = {}
    stats = {"vae_prefix": 0, "model_prefix": 0, "bare": 0, "skipped": 0}

    for key, value in sd.items():
        if key.startswith("vae.vae."):
            out[key] = value
            stats["vae_prefix"] += 1
        elif key.startswith("vae.") and any(key[4:].startswith(p) for p in bare_prefixes):
            # ``vae.encoder.…``（TrainableSeismicVAE 的 state_dict 形态）
            out[f"vae.{key}"] = value
            stats["vae_prefix"] += 1
        elif key.startswith("model.") and any(key[6:].startswith(p) for p in bare_prefixes):
            out[f"vae.vae.{key[6:]}"] = value
            stats["model_prefix"] += 1
        elif any(key.startswith(p) for p in bare_prefixes):
            out[f"vae.vae.{key}"] = value
            stats["bare"] += 1
        else:
            stats["skipped"] += 1

    return out, stats



def main():
    a=parse_args(); seed_everything(a.seed,workers=True)
    if a.max_steps<=0: raise ValueError('--max-steps must be positive')
    model=SeismicVAELightning(stage=a.stage,learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,ema_decay=a.ema_decay,grad_checkpoint=a.grad_checkpoint)
    if a.init:
        load_backbone_weights(model.vae.vae,a.init,allow_wan=(a.stage=='warmup'))
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if saved.get('hyper_parameters',{}).get('stage')!=a.stage:
            raise ValueError('resume requires the same stage; use --init for the stage transition')
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    callbacks=[ModelCheckpoint(dirpath=out/'checkpoints',save_last=True,
        save_top_k=-1,every_n_train_steps=a.save_every,filename='vae-{step:08d}'),
        LearningRateMonitor(logging_interval='step'),LossPrintCallback(every_k_steps=20)]
    trainer=Trainer(default_root_dir=out,max_steps=a.max_steps,accelerator='auto',
        devices=a.devices,precision=a.precision,
        strategy=DDPStrategy(find_unused_parameters=True) if a.devices>1 else 'auto',
        use_distributed_sampler=False,callbacks=callbacks,
        logger=TensorBoardLogger(str(out),name='logs'),log_every_n_steps=20)
    trainer.fit(model,datamodule=PaperDataModule(a),ckpt_path=a.resume)

if __name__=='__main__': main()
