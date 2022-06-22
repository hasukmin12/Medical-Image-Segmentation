import torch
from monai.networks.nets import UNETR
# Load ViT backbone weights into UNETR
def call_pretrained_unetr(info, config):     
    model = UNETR(
        in_channels = config["CHANNEL_IN"],
        out_channels = config["CHANNEL_OUT"],
        img_size = config["INPUT_SHAPE"],
        feature_size = config["PATCH_SIZE"],
        hidden_size = config["EMBED_DIM"],
        mlp_dim = config["MLP_DIM"],
        num_heads = config["NUM_HEADS"],
        dropout_rate = config["DROPOUT"],
        pos_embed= config["POS_EMBED"],
        norm_name= config["NORM_NAME"],
        res_block=True,        
    )
    try:
        vit_dict = torch.load(info["PRE_TRAINED"])
        vit_weights = vit_dict['state_dict']
    except:
        vit_dict = torch.load('/nas3/jepark/pretrained/vitautoenc_weights.pt')
        vit_weights = vit_dict['state_dict']

    # Remove items of vit_weights if they are not in the ViT backbone (this is used in UNETR).
    # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
    # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
    # while pretraining with ViTAutoEnc and are not a part of ViT backbone.
    model_dict = model.vit.state_dict()
    vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
    model_dict.update(vit_weights)
    model.vit.load_state_dict(model_dict)
    del model_dict, vit_weights, vit_dict
    print('Pretrained Weights Succesfully Loaded !')
    
    return model