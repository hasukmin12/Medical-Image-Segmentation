import sys
import importlib
sys.path.append('../')
def call_trans_function(config):
    mod = importlib.import_module(f'transforms.trans_v{config["TRANSFORM"]}')
    train_transforms, val_transforms = mod.call_transforms(config)
    return train_transforms, val_transforms