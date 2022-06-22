# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from core.model.Conv_blocks import CoordAtt, DDenseDownLayer_2, DDenseDownBlock_2, DenseUpBlock, DenseUpLayer , DDenseDownLayer_first, DDenseDownBlock_first

__all__ = ["DD_Unet", "ddunet", "ddunet", "DD_Unet"]


class DDense_Encoder(nn.Module):
    def __init__(self, in_channel, channel_list, kernel_size, drop_rate):
        super(DDense_Encoder, self).__init__()
        self.stages = []
        initial_channel = channel_list[0]
        self.initial_conv = nn.Conv3d(in_channel, initial_channel, 3, padding=1)
        self.initial_norm = nn.BatchNorm3d(initial_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.initial_nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True) 

        current_stage = DDenseDownLayer_first(initial_channel, initial_channel, kernel_size, drop_rate)
        self.stages.append(current_stage)

        for stage in range(1, len(channel_list)):
            current_input_feature = channel_list[stage-1]
            current_output_feature = channel_list[stage]
            current_stage = DDenseDownLayer_2(current_input_feature, current_output_feature,  kernel_size, drop_rate)
            self.stages.append(current_stage)
        self.stages = nn.ModuleList(self.stages)

    def forward(self, x):
        skips = []
        x = self.initial_nonlin(self.initial_norm(self.initial_conv(x)))
        for s in self.stages:
            x, resi = s(x)
            skips.append(resi)
        return skips
        


class DDense_Decoder(nn.Module):
    def __init__(self, out_channel, channel_list, kernel_size, drop_rate):
        super(DDense_Decoder, self).__init__()
        self.CA = []
        self.trans_conv = []
        self.stages = []
        transpconv = nn.ConvTranspose3d

        for stage in range(len(channel_list)-1, 0, -1):
            current_input_feature = channel_list[stage]
            current_output_feature = channel_list[stage-1]

            self.trans_conv.append(transpconv(current_input_feature, current_output_feature, kernel_size=(2,2,2), stride=(2,2,2), bias=False))
            current_stage = DenseUpLayer(current_input_feature, current_output_feature, kernel_size, drop_rate)
            self.stages.append(current_stage)

        self.segmentation_output = nn.Conv3d(channel_list[0], out_channel, kernel_size=[1 for _ in kernel_size], stride=[1 for i in kernel_size], padding=[0 for i in kernel_size], bias=False)
        
        self.trans_conv = nn.ModuleList(self.trans_conv)
        self.stages = nn.ModuleList(self.stages)

    def forward(self, skips):
        skips = skips[::-1]
        seg_outputs = []
        x = skips[0]  

        for i in range(len(skips)-1):

            x = self.trans_conv[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)

            # if self.deep_supervision and (i != len(self.tus) - 1):
            #     seg_outputs.append(self.deep_supervision_outputs[i](x))

        segmentation = self.segmentation_output(x)
        return segmentation




class DD_UNet(nn.Module):

    def __init__(self, in_channel, out_channel, channel_list, kernel_size, drop_rate):
        super().__init__()

        self.encoder = DDense_Encoder(in_channel, channel_list, kernel_size, drop_rate)
        self.decoder = DDense_Decoder(out_channel, channel_list, kernel_size, drop_rate)

    def forward(self, x):
        skips = self.encoder(x)
        rst = self.decoder(skips)
        return rst
