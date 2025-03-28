from datasets.scan3r import Scan3RDataset
from datasets.lamar import LamarDataset
from utils import torch_util

def get_train_val_data_loader(cfg):
    train_dataset = LamarDataset(cfg, split='train')
    train_dataloader = torch_util.build_dataloader(train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.num_workers, shuffle=False,
                                                   collate_fn=train_dataset.collate_fn, pin_memory=True, drop_last=True)
    val_dataset = LamarDataset(cfg, split='val')
    val_dataloader = torch_util.build_dataloader(val_dataset, batch_size=cfg.val.batch_size, num_workers=cfg.num_workers, shuffle=False,
                                                   collate_fn=val_dataset.collate_fn, pin_memory=True, drop_last=True)

    return train_dataloader, val_dataloader

def get_val_dataloader(cfg, batch_size=None):
    val_dataset = LamarDataset(cfg, split='test')
    batch_s = cfg.val.batch_size
    if batch_size is not None:
        batch_s = batch_size
    val_dataloader = torch_util.build_dataloader(val_dataset, batch_size=batch_s, num_workers=cfg.num_workers, shuffle=False,
                                                   collate_fn=val_dataset.collate_fn, pin_memory=True, drop_last=True)
    return val_dataset, val_dataloader
    
