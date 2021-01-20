import random
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x*1. / divisible_by)* divisible_by)


def conv_1x1_bn(input_channel, output_channel, bn_momentum=0.1, activation="relu"):
    if activation == "relu":
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel, momentum=bn_momentum),
            nn.ReLU6(inplace=True)
        )
    elif activation == "hswish":
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel, momentum=bn_momentum),
            HSwish()
        )

class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace=True) / 6
        return out


class ConvBNRelu(nn.Sequential):
    def __init__(self, 
                 input_channel,
                 output_channel,
                 kernel,
                 stride,
                 pad,
                 activation="relu",
                 bn=True,
                 group=1,
                 bn_momentum=0.1,
                 *args,
                 **kwargs):

        super(ConvBNRelu, self).__init__()

        assert activation in ["hswish", "relu", None]
        assert stride in [1, 2, 4]

        self.add_module("conv", nn.Conv2d(input_channel, output_channel, kernel, stride, pad, groups=group, bias=False))
        if bn:
            self.add_module("bn", nn.BatchNorm2d(output_channel, momentum=bn_momentum))

        if activation == "relu":
            self.add_module("relu", nn.ReLU6(inplace=True))
        elif activation == "hswish":
            self.add_module("hswish", HSwish())

            
class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out

            
class SEModule(nn.Module):
    def __init__(self,  in_channel,
                 reduction=4,
                 squeeze_act=nn.ReLU(inplace=True),
                 excite_act=HSigmoid(inplace=True)):
        super(SEModule, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(in_channels=in_channel,
                                      out_channels=in_channel // reduction,
                                      kernel_size=1,
                                      bias=True)
        self.squeeze_act = squeeze_act
        self.excite_conv = nn.Conv2d(in_channels=in_channel // reduction,
                                     out_channels=in_channel,
                                     kernel_size=1,
                                     bias=True)
        self.excite_act = excite_act

    def forward(self, inputs):
        feature_pooling = self.global_pooling(inputs)
        feature_squeeze_conv = self.squeeze_conv(feature_pooling)
        feature_squeeze_act = self.squeeze_act(feature_squeeze_conv)
        feature_excite_conv = self.excite_conv(feature_squeeze_act)
        feature_excite_act = self.excite_act(feature_excite_conv)
        return inputs * feature_excite_act


class MPDBlock(nn.Module):
    """Mixed path depthwise block"""
    def __init__(self,
                 input_channel,
                 output_channel,
                 stride,
                 split_block=4,
                 kernels=[3, 5, 7, 9],
                 axis=1,
                 activation="relu",
                 block_type="MB",
                 bn_momentum=0.1):
        super(MPDBlock, self).__init__()
        self.block_input_channel = input_channel//split_block
        self.block_output_channel = output_channel//split_block
        self.split_block = len(kernels)

        self.blocks = nn.ModuleList()
        for b in range(split_block):
            operation = nn.Conv2d(
                        self.block_input_channel, 
                        self.block_output_channel, 
                        kernels[b], 
                        stride, 
                        (kernels[b]//2), 
                        groups=self.block_output_channel, 
                        bias=False)
            self.blocks.append(operation)
            
        self.bn = nn.BatchNorm2d(output_channel, momentum=bn_momentum)

        if activation == "relu":
            self.activation= nn.ReLU6(inplace=True)
        elif activation == "hswish":
            self.activation = HSwish()

        self.axis = axis 

    def forward(self, x):
        split_x = torch.split(x, self.block_input_channel, dim=self.axis)

        output_list = []
        for x_i, conv_i in zip(split_x, self.blocks):
            output = conv_i(x_i)
            output_list.append(output)

        x = torch.cat(output_list, dim=self.axis)

        x = self.bn(x)
        x = self.activation(x)

        return x


class MBConv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 expansion,
                 kernels,
                 stride,
                 activation,
                 split_block=1,
                 group=1,
                 se=False,
                 bn_momentum=0.1,
                 *args,
                 **kwargs):
        super(MBConv, self).__init__()

        self.use_res_connect = True if (stride==1 and input_channel == output_channel) else False
        mid_channel = int(input_channel * expansion)

        self.group = group

        if input_channel == mid_channel:
            self.point_wise = nn.Sequential()
        else:
            self.point_wise = ConvBNRelu(input_channel,
                                         mid_channel,
                                         kernel=1,
                                         stride=1,
                                         pad=0,
                                         activation=activation,
                                         group=group,
                                         bn_momentum=bn_momentum
                                    )

        self.depthwise = MPDBlock(mid_channel,
                                  mid_channel,
                                  stride,
                                  split_block=expansion,
                                  kernels=kernels,
                                  activation=activation,
                                  bn_momentum=bn_momentum)

        self.point_wise_1 = ConvBNRelu(mid_channel,
                                       output_channel,
                                       kernel=1,
                                       stride=1,
                                       pad=0,
                                       activation=None,
                                       bn=True,
                                       group=group,
                                       bn_momentum=bn_momentum
                                    )
        self.se = SEModule(mid_channel) if se else None
        self.expansion = expansion

    def forward(self, x):
        y = self.point_wise(x)
        y = self.depthwise(y)

        y = self.se(y) if self.se is not None else y
        y = self.point_wise_1(y)

        y = y + x if self.use_res_connect else y
        return y

