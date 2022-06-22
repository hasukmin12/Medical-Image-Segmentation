import os, glob
import torch
from torch import optim
from monai.data import *

def call_model(info, config):
    model = None
    if config["MODEL_NAME"] in ['unetr', 'UNETR']:
        from monai.networks.nets import UNETR
        model = UNETR(
            in_channels = config["CHANNEL_IN"],
            out_channels = config["CHANNEL_OUT"],
            img_size = config["INPUT_SHAPE"],
            feature_size = config["PATCH_SIZE"],
            hidden_size = config["EMBED_DIM"],
            mlp_dim = config["MLP_DIM"],
            num_heads = config["NUM_HEADS"],
            dropout_rate = config["DROPOUT"],
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
        )
    elif config["MODEL_NAME"] in ['unetr_pretrained', 'UNETR_PreTrained']:
        from core.model.UNETR import call_pretrained_unetr
        model = call_pretrained_unetr(info, config)
    elif config["MODEL_NAME"] in ['vnet', 'VNET', 'VNet', 'Vnet']:
        from monai.networks.nets import VNet
        model = VNet(
            spatial_dims=3,
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
        )
    elif config["MODEL_NAME"] in ['unet', 'UNET', 'UNet', 'Unet']:
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3,
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif config["MODEL_NAME"] in ['dynunet', 'DynUNet', 'DynUnet']:
        from monai.networks.nets import DynUNet
        assert config["DynUnet_strides"][0] == 1, "Strides should be start with 1"
        model = DynUNet(
            spatial_dims=3,
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            kernel_size=config["DynUnet_kernel"],#[3,3,3,3,3],
            strides=config["DynUnet_strides"],#[1,2,2,2,2],
            upsample_kernel_size=config["DynUnet_upsample"],#[2,2,2,2,2],
            filters=config["DynUnet_filters"],#[64, 96, 128, 192, 256, 384, 512, 768, 1024],
            dropout=config["DROPOUT"],
            deep_supervision=True,
            deep_supr_num=len(config["DynUnet_strides"])-2,
            norm_name='INSTANCE',
            act_name='leakyrelu',
            res_block=config["DynUnet_residual"], #False
            trans_bias=False,
        )
    elif config["MODEL_NAME"] in ['swinUNETR', 'sunetr', 'sUNETR']:
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=config["INPUT_SHAPE"],
            in_channels=config["CHANNEL_IN"],
            out_channels=config["CHANNEL_OUT"],
            feature_size=config["FEATURE_SIZE"],
            use_checkpoint=True,
        )

    elif config["MODEL_NAME"] in ['CADD_UNet', 'Caddunet', 'CADD_Unet', 'caddunet']:
        from core.model.CADD_UNet import CADD_UNet
        model = CADD_UNet(
            in_channel=config["CHANNEL_IN"],
            out_channel=config["CHANNEL_OUT"],
            channel_list=[32, 64, 128, 256, 512],
            kernel_size= (3, 3, 3),
            drop_rate = config["DROPOUT"],
        )
        # print(model)  # 필요없으면 지우셔도 됩니다!

    elif config["MODEL_NAME"] in ['DD_UNet', 'ddunet', 'DD_Unet', 'DDunet']:
        from core.model.DD_UNet import DD_UNet
        model = DD_UNet(
            in_channel=config["CHANNEL_IN"],
            out_channel=config["CHANNEL_OUT"],
            channel_list=[32, 64, 128, 256, 512],
            kernel_size= (3, 3, 3),
            drop_rate = config["DROPOUT"],
        )
        # print(model)  # 필요없으면 지우셔도 됩니다!


    assert model is not None, 'Model Error!'    
    return model


def call_optimizer(config, model):
    if config["OPTIM_NAME"] in ['SGD', 'sgd']:
        return optim.SGD(model.parameters(), lr=config["LR_INIT"], momentum=config["MOMENTUM"])
    elif config["OPTIM_NAME"] in ['ADAM', 'adam', 'Adam']:
        return optim.Adam(model.parameters(), lr=config["LR_INIT"])
    elif config["OPTIM_NAME"] in ['ADAMW', 'adamw', 'AdamW', 'Adamw']:
        return optim.AdamW(model.parameters(), lr=config["LR_INIT"], weight_decay=config["LR_DECAY"])
    elif config["OPTIM_NAME"] in ['ADAGRAD', 'adagrad', 'AdaGrad']:
        return optim.Adagrad(model.parameters(), lr=config["LR_INIT"], lr_decay=config["LR_DECAY"])
    else:
        return None
