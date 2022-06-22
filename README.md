# MIAI Segmentation Framework 
1. config 폴더에 [target]_[version].py 를 생성해주세요.
2개의 dictionary info (experiment infomation)와 config (experiment configuration)를 생성해주세요. 
##### "info" includes 
> TARGET_NAME, VERSION(optional), FOLDS, ROOT, CHANNEL_IN, CHANNEL_OUT, CLASS_NAMES
##### "config" includes 
> FOLD, FAST, MAX_ITERATIONS, EVAL_NUM, BATCH_SIZE, SEEDS, 
> SAMPLES, SPACING, CONTRAST, 
###### For Model
> CHANNEL_IN, CHANNEL_OUT, INPUT_SHAPE, MODEL_NAME, LOAD_MODEL, DROPOUT, OPTIM_NAME, 
###### For Transformer-based Model
> PATCH_SIZE(optional), EMBED_DIM(optional), MLP_DIM(optional), NUM_HEADS(optional), 
> KERNEL(optional), STRIDES(optional), UP_KERNEL(optional), DEEP_SUPER(optional), ACTIVATION_NAME(optional), 
###### For Loss
> LOSS_NAME, LR_INIT, LR_DECAY, MOMENTUM(optional)
2. python train.py [target]_[version] 으로 학습해주세요. 
3. python inference.py [target]_[version] -i 'input_path' -o 'output_path' 로 inference 해주세요. 
