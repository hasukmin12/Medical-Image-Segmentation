import os
from ray import tune


save_name = 'cadd_DFP' # 'unter_DFP_pretrained'

info = {
    "TARGET_NAME"   : "kipa_model",
    "VERSION"       : 1,
    "FOLD"          : 4,
    "FOLDS"         : 5, 
    "ROOT"          : "/nas3/sukmin/datasets/Task302_KiPA",
    "CHANNEL_IN"    : 1,
    "CHANNEL_OUT"   : 5,
    "CLASS_NAMES"   : {1:'Renal vein',2:'Kidney', 3:'Renal atery', 4:'Tumor'},   
    #### wandb
    "ENTITY"        : "has_in_tom",
    "PROJ_NAME"     : "kipa2022",
    "VISUAL_AXIS"   : 3, # 1 or 2 or 3
    #### ray
    "TUNE"          : False,
    "GPUS"          : "0,1",
    "MEM_CACHE"     : 0.5,
    "VALID_GPU"     : True,
}

info["LOGDIR"] = os.path.join(f'/nas3/sukmin/{info["TARGET_NAME"]}', save_name)
if os.path.isdir(info["LOGDIR"])==False:
    os.makedirs(info["LOGDIR"])
info["NUM_GPUS"] = len(info["GPUS"].split(','))
info["WORKERS"] = 4*info["NUM_GPUS"] if info["MEM_CACHE"]>0 else 8*info["NUM_GPUS"]

config = {
    "save_name"     : save_name,
    "LOSS_NAME"     : "DiceFocal_Portion",
    "BATCH_SIZE"    : 2,
    "TRANSFORM"     : 1,
    "SPACING"       : False,
    "INPUT_SHAPE"   : [96,96,64],
    "DROPOUT"       : 0.1,
    "CONTRAST"      : [0,2000], # [-150,300],
    "FAST"          : False,
    "MAX_ITERATIONS": 100000,
    "EVAL_NUM"      : 500,
    "SAMPLES"       : 2,
    "SEEDS"         : 12321,
    "MODEL_NAME"    : "caddunet",# "unetr_pretrained",
    "LOAD_MODEL"    : False,
    "OPTIM_NAME"    : "AdamW",
    "LR_INIT"       : 5e-04,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    #### DoNotChange! by JEpark
    "CHANNEL_IN"    : info["CHANNEL_IN"], 
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],
    "FOLD"          : info["FOLD"],   

    # for UNETR
    "PATCH_SIZE"    : 32,
    "EMBED_DIM"     : 768,
    "MLP_DIM"       : 3072,
    "NUM_HEADS"     : 12,
    "DROPOUT"       : 0.0,
    "POS_EMBED"     : "conv", # "perceptron"
    "NORM_NAME"     : "INSTANCE",   
}


search = {
    "CONTRAST_L"    : tune.randint(0,500),
    "CONTRAST_U"    : tune.randint(50,1000),
    "BATCH_SIZE"    : tune.qrandint(4,8,2),
    "SAMPLES"       : tune.qrandint(2,20,2),
    "SEEDS"         : tune.choice([1,12321]),
    "INPUT_SHAPE_XY": tune.choice([32,64,96,128]),
    "INPUT_SHAPE_Z" : tune.choice([32,64,96,128]),
    "MODEL_NAME"    : tune.choice(["unet","vnet"]),
    "DROPOUT"       : tune.quniform(0.0,0.5,0.1),
    "OPTIM_NAME"    : tune.choice(["SGD","AdamW"]),
    "LOSS_NAME"     : tune.choice(["DiceCE","Dice","DiceFocal, DiceFocal_Portion"]),
}
