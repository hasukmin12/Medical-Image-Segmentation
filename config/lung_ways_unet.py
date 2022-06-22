import os
from ray import tune
## model information
info = {
    "TARGET_NAME"   : "lung_ways",
    "VERSION"       : 0,
    "FOLD"          : 1,
    "FOLDS"         : 5, 
    "ROOT"          : "/home/jepark/Downloads/hutom_lung",
    "CHANNEL_IN"    : 1,
    "CHANNEL_OUT"   : 4,
    "CLASS_NAMES"   : {1:"Artery",2:"Vein",3:"Bronchus"},   
    #### wandb
    "ENTITY"        : "jeune",
    "PROJ_NAME"     : "lung",
    "VISUAL_AXIS"   : 1, # 1 or 2 or 3
    #### ray
    "TUNE"          : False,
    #### hardware
    "GPUS"          : "0",
    "MEM_CACHE"     : 0.05,
    "VALID_GPU"     : False,
}
## training configuration
config = {
    "SPACING"       : None,
    "CONTRAST"      : [-150,1000],
    "TRANSFORM"     : "Lung",
    "FAST"          : False,
    "BATCH_SIZE"    : 1,
    "MAX_ITERATIONS": 50000,
    "EVAL_NUM"      : 500,
    "SAMPLES"       : 2,
    "SEEDS"         : 12321,
    "INPUT_SHAPE"   : [128,128,128],
    #########
    "MODEL_NAME"    : "DynUnet",
    "DynUnet_kernel": [3,3,3,3,3],
    "DynUnet_strides": [1,2,2,2,2],
    "DynUnet_upsample": [2,2,2,2,2],
    "DynUnet_filters": [64, 96, 128, 192, 256, 384, 512, 768, 1024],
    "DynUnet_residual": True,
    "LOAD_MODEL"    : False,
    "DROPOUT"       : 0.25,
    #########
    "OPTIM_NAME"    : "AdamW",
    "LOSS_NAME"     : "DiceFocal",
    "LR_INIT"       : 5e-04,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    #### DoNotChange! by JEpark
    "CHANNEL_IN"    : info["CHANNEL_IN"],   
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],  
    "FOLD"          : info["FOLD"],         
}

info["NUM_GPUS"] = len(info["GPUS"].split(','))
info["WORKERS"] = 4*info["NUM_GPUS"] if info["MEM_CACHE"]>0 else 8*info["NUM_GPUS"]
info["LOGDIR"] = f'/home/jepark/Downloads/code/MIAI_Segmentation/train_results/{info["TARGET_NAME"]}/{info["VERSION"]}/{config["MODEL_NAME"]}/fold{config["FOLD"]}'
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
