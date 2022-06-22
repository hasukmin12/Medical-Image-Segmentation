import os, sys, glob
sys.path.append('../')
import torch
import wandb
import numpy as np
from torch import nn
from tqdm import tqdm

from monai.data import *
from monai.metrics import *
from monai.transforms import Activations
from monai.inferers import sliding_window_inference

from core.utils import *   
from core.call_loss import *
from core.call_model import *

def validation(info, config, valid_loader, model, logging=False, threshold=0.5):  
    activation = Activations(sigmoid=True) # softmax : odd result! ToDO : check!  
    dice_metric = DiceMetric(include_background=False, reduction='none')
    confusion_matrix = ConfusionMatrixMetric(include_background=False, reduction='none')

    epoch_iterator_val = tqdm(
        valid_loader, desc="Validate (X / X Steps)", dynamic_ncols=True
    )
    dice_class, mr_class, fo_class = [], [], []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            step += 1
            
            if info["VALID_GPU"]:
                val_inputs, val_labels = batch["image"], batch["label"].to("cuda", non_blocking=True).long()
                val_outputs = sliding_window_inference(val_inputs, config["INPUT_SHAPE"], 4, model,device='cuda',sw_device='cuda')
            else:
                val_inputs, val_labels = batch["image"], batch["label"].long()
                val_outputs = sliding_window_inference(val_inputs, config["INPUT_SHAPE"], 4, model,device='cpu',sw_device='cuda')
            
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (step, len(epoch_iterator_val))
            )
            dice_class.append(dice_metric(val_outputs>=threshold, val_labels)[0])

            confusion = confusion_matrix(val_outputs>=threshold, val_labels)[0]
            mr_class.append([
                calc_confusion_metric('fnr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
            ])
            fo_class.append([
                calc_confusion_metric('fpr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
            ])
            torch.cuda.empty_cache()
        dice_dict, dice_val = calc_mean_class(info, dice_class, 'valid_dice')
        miss_dict, miss_val = calc_mean_class(info, mr_class, 'valid_miss rate')
        false_dict, false_val = calc_mean_class(info, fo_class, 'valid_false alarm')
        if logging:
            wandb.log({
                'valid_dice': dice_val,
                'valid_miss rate': miss_val,
                'valid_false alarm': false_val,
                'valid_image': log_image_table(info, val_inputs[0].cpu(),
                                                val_labels[0].cpu(),val_outputs[0].cpu()),
            })
            wandb.log(dice_dict)
            wandb.log(miss_dict)
            wandb.log(false_dict)        
    return dice_val

def train(info, config, global_step, dice_val_best, model, optimizer, train_loader, valid_loader, logging=False): 
    # print(model)
    loss_function = call_loss(loss_mode=config["LOSS_NAME"], sigmoid=True, config=config)
    dice_loss = call_loss(loss_mode='dice', sigmoid=True)
 
    model.train()

    step = 0
    epoch_loss, epoch_dice = 0., 0.
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1

        x, y = batch["image"].to("cuda"), batch["label"].to("cuda").long()
        # x, y = batch["image"].to("cuda", non_blocking=True), batch["label"].to("cuda", non_blocking=True).long()

        logit_map = model(x)

        loss, dice = 0, 0
        if logit_map.dim() > x.dim(): # deep supervision
            for ds in range(logit_map.shape[1]):
                loss += loss_function(logit_map[:,ds], y)
                dice += 1 - dice_loss(logit_map[:,ds], y)
            loss /= logit_map.shape[1]
            dice /= logit_map.shape[1]
        else:
            loss = loss_function(logit_map, y)
            dice = 1 - dice_loss(logit_map, y)

        epoch_loss += loss.item()
        epoch_dice += dice.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step+1, config["MAX_ITERATIONS"], loss)
        )
        if (
            global_step % config["EVAL_NUM"] == 0 and global_step != 0
        ) or global_step == config["MAX_ITERATIONS"]:
            
            dice_val = validation(info, config, valid_loader, model, logging)

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(info["LOGDIR"], f"model_best.pth"))
                print(
                    f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
                )
            else:
                print(
                    f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
                )

            # 10k마다 model save
            if global_step % 10000 == 0 and global_step != 0 :
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(info["LOGDIR"], "model_e{0:05d}.pth".format(global_step)))
                print(
                    f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
                )

        global_step += 1
    return global_step, dice_val_best, epoch_loss / step, epoch_dice / step
