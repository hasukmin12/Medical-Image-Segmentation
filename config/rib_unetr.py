import os
from ray import tune
## model information
info = {
    "TARGET_NAME"   : "rib",
    "VERSION"       : 1,
    "FOLD"          : 1,
    "FOLDS"         : 5, 
    "ROOT"          : "/nas3/jepark/nnUNet_raw_data_base/nnUNet_raw_data/Task003",
    "CHANNEL_IN"    : 1,
    "CHANNEL_OUT"   : 2,
    "CLASS_NAMES"   : {1:'rib'},   
    #### wandb
    "ENTITY"        : "hutom_miai",
    "PROJ_NAME"     : "rib",
    "VISUAL_AXIS"   : 1, # 1 or 2 or 3
    #### ray
    "TUNE"          : False,
    #### hardware
    "GPUS"          : "1,2",
    "MEM_CACHE"     : 1.0,
    "VALID_GPU"     : False,
}
## training configuration
config = {
    "SPACING"       : None,
    "CONTRAST"      : [-100,150],
    "TRANSFORM"     : 1,
    "FAST"          : False,
    "BATCH_SIZE"    : 8,
    "MAX_ITERATIONS": 50000,
    "EVAL_NUM"      : 500,
    "SAMPLES"       : 2,
    "SEEDS"         : 12321,
    "INPUT_SHAPE"   : [96,96,96],
    "MODEL_NAME"    : "unetr_pretrained",
    "LOAD_MODEL"    : False,
    "OPTIM_NAME"    : "AdamW",
    "LOSS_NAME"     : "DiceCE",
    "LR_INIT"       : 5e-04,
    "LR_DECAY"      : 1e-05,
    "PATCH_SIZE"    : 32,
    "EMBED_DIM"     : 768,
    "MLP_DIM"       : 3072,
    "NUM_HEADS"     : 12,
    "DROPOUT"       : 0.0,
    "POS_EMBED"     : "conv", # "perceptron"
    "NORM_NAME"     : "INSTANCE",
    #### DoNotChange! by JEpark
    "CHANNEL_IN"    : info["CHANNEL_IN"],   
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],  
    "FOLD"          : info["FOLD"],         
}

info["NUM_GPUS"] = len(info["GPUS"].split(','))
info["WORKERS"] = 4*info["NUM_GPUS"] if info["MEM_CACHE"]>0 else 8*info["NUM_GPUS"]
info["LOGDIR"] = f'/nas3/jepark/train_results/{info["TARGET_NAME"]}_{info["VERSION"]}/{config["MODEL_NAME"]}/fold{config["FOLD"]}'
if os.path.isdir(info["LOGDIR"]):
    os.system(f'rm -rf {info["LOGDIR"]}')
os.makedirs(info["LOGDIR"])

## ray optimization - configuration
search = {
    "MAX_ITERATIONS": config["MAX_ITERATIONS"],
    "EVAL_NUM"      : config["EVAL_NUM"],
    "CHANNEL_IN"    : info["CHANNEL_IN"],   
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],  
    "FOLD"          : info["FOLD"],   
    "MODEL_NAME"    : config["MODEL_NAME"],
    "LOAD_MODEL"    : False,
    "FAST"          : False,
    "LR_INIT"       : 5e-04,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    "ISOTROPIC"     : True,
    #######
    "CONTRAST_L"    : tune.randint(-1000,500),
    "CONTRAST_U"    : tune.randint(50,1000),
    "BATCH_SIZE"    : tune.qrandint(4,64,2),
    "SAMPLES"       : tune.qrandint(2,20,2),
    "SEEDS"         : tune.choice([1,12321]),
    "INPUT_SHAPE_XY": tune.choice([32,64,96,128,196]),
    "SPACING_XY"    : tune.choice([None, 0.5, 1, 1.5]),
    "DROPOUT"       : tune.quniform(0.0,0.5,0.1),
    "OPTIM_NAME"    : tune.choice(["SGD","AdamW"]),
    "LOSS_NAME"     : tune.choice(["DiceCE","Dice","DiceFocal"]),
}
if not search["ISOTROPIC"]: 
    search["INPUT_SHAPE_Z"] = tune.choice([32,64,96,128,196])
    search["SPACING_Z"] = tune.choice([None, 0.5, 1, 1.5])
