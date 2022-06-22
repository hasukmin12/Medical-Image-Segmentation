import os, glob
import torch
from torch import optim
from monai.data import *

def call_fold_dataset(list_, target_fold, total_folds=5):
    train, valid = [],[]
    count = 0
    for i in list_:
        count += 1
        if count == total_folds: count = 1
        if count == target_fold:
            valid.append(i)
        else:
            train.append(i)
    return train, valid

def call_dataloader(info, config, data_list, transforms, progress=False, mode='train'):
    if mode == 'train' : 
        batch_size = config["BATCH_SIZE"]
        shuffle = True
    else:
        batch_size = 1
        shuffle = False

    if info["MEM_CACHE"]>0:
        ds = CacheDataset(
            data=data_list,
            transform=transforms,
            cache_rate=info["MEM_CACHE"], num_workers=info["WORKERS"],
            progress=progress
        )
    else:
        ds = Dataset(
            data=data_list,
            transform=transforms,
        )

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=info["WORKERS"],
        pin_memory=True, shuffle=shuffle, 
        prefetch_factor=10, persistent_workers=False,
    )

    # if config["FAST"]:
    #     loader = ThreadDataLoader(
    #         ds, batch_size=batch_size, num_workers=0, shuffle=shuffle,
    #     )
    # else:
    #     loader = DataLoader(
    #         ds, batch_size=batch_size, num_workers=info["WORKERS"],
    #         pin_memory=True, shuffle=shuffle, 
    #     )
    return loader

