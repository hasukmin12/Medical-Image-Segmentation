import glob
import os, tempfile
import random
from requests import post
import torch
import wandb
import argparse as ap
import numpy as np
import nibabel as nib
import yaml
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from monai.data import *
from monai.transforms import *
from monai.handlers.utils import from_engine

from core.utils import *
from core.call import *
from config.call import call_config

def main(info, config, data_dir, output_dir):

    test_images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    test_data = [{"image": image} for image in test_images]

    test_transforms = [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS")
        ]
    if config["SPACING"] is not None:
        test_transforms += [Spacingd(
            pixdim=config["SPACING"], image_only=True
        )]
    test_transforms += [
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config["CONTRAST"][0], a_max=config["CONTRAST"][1], 
            b_min=0, b_max=1, clip=True
        ),
        EnsureTyped(keys="image")
    ]
    test_transforms = Compose(test_transforms)

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=None),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir, output_postfix="seg", resample=False, separate_folder=False),
    ])

    test_org_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

    model = call_model(info, config)
    # Multi-GPU로 학습했다면 inference도 Multi-GPU로 해야함! 그렇지 않으면 아래 issue 발생
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(info["LOGDIR"],args.epoch_num))["model_state_dict"])
    model.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data['image'].to(device)
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(
                test_inputs, config["INPUT_SHAPE"], sw_batch_size, model)
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
      
if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-t', dest='trainer', default='organ_v1')
    parser.add_argument('-i', dest='input_path', default="/nas3/sukmin/datasets/Task001_Multi_Organ/imagesTs")
    parser.add_argument('-o', dest='output_path', default="/nas3/sukmin/inf_rst/multi_organ_model/ddunet_focal_portion_drop0.1_e4")
    parser.add_argument('-n', dest='epoch_num', default='model_e40000.pth') # model_e50000.pth # model_best.pth
    args = parser.parse_args()
    if os.path.isdir(args.output_path)== False:
        os.makedirs(args.output_path, exist_ok=True)
    info, config, search = call_config(args.trainer)
    os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    main(info, config, args.input_path, args.output_path)
