import math

import torch
import torch.nn as nn

from utils.network_utils import ConvBNRelu, MBConv, conv_1x1_bn

SGNAS_A = [[6, 32, [5, 7, 3, 7, 3, 3], 2, 4, False], [4, 32, [7, 3, 3, 3], 1, 4, False], [3, 40, [3, 5, 3], 2, 4, False], [5, 40, [7, 7, 3, 7, 5], 1, 4, False], [4, 40, [7, 5, 3, 3], 1, 4, False], [5, 40, [7, 5, 5, 5, 7], 1, 4, False], [5, 80, [7, 3, 3, 7, 7], 2, 4, False], [5, 80, [7, 3, 3, 3, 3], 1, 4, False], [5, 80, [7, 5, 5, 7, 3], 1, 4, False], [4, 80, [7, 5, 5, 3], 1, 4, False], [4, 96, [5, 7, 3, 5], 1, 4, False], [4, 96, [7, 7, 3, 7], 1, 4, False], [4, 96, [7, 3, 5, 3], 1, 4, False], [4, 96, [7, 7, 3, 5], 1, 4, False], [4, 192, [3, 7, 3, 3], 2, 4, False], [5, 192, [3, 3, 3, 7, 3], 1, 4, False], [4, 192, [7, 3, 3, 7], 1, 4, False], [3, 192, [7, 7, 7], 1, 4, False], [5, 320, [3, 3, 7, 3, 7], 1, 4, False]]
SGNAS_B = [[4, 32, [5, 7, 3, 5], 2, 4, False], [3, 32, [5, 5, 3], 1, 4, False], [3, 40, [5, 5, 3], 2, 4, False], [5, 40, [5, 5, 3, 5, 5], 1, 4, False], [5, 40, [5, 5, 5, 3, 3], 1, 4, False], [5, 40, [3, 5, 5, 5, 5], 1, 4, False], [5, 80, [3, 5, 5, 7, 7], 2, 4, False], [4, 80, [5, 5, 3, 7], 1, 4, False], [4, 80, [5, 5, 5, 7], 1, 4, False], [4, 80, [5, 5, 5, 3], 1, 4, False], [2, 96, [5, 5], 1, 4, False], [3, 96, [5, 5, 5], 1, 4, False], [3, 96, [5, 3, 5], 1, 4, False], [4, 96, [3, 7, 5, 3], 1, 4, False], [4, 192, [3, 7, 3, 3], 2, 4, False], [4, 192, [3, 7, 7, 7], 1, 4, False], [4, 192, [7, 5, 5, 7], 1, 4, False], [3, 192, [5, 3, 5], 1, 4, False], [5, 320, [3, 3, 7, 3, 7], 1, 4, False]]
SGNAS_C = [[4, 32, [5, 5, 3, 5], 2, 4, False], [3, 32, [5, 5, 3], 1, 4, False], [3, 40, [5, 5, 3], 2, 4, False], [3, 40, [5, 5, 3], 1, 4, False], [4, 40, [5, 5, 5, 3], 1, 4, False], [3, 40, [3, 5, 5], 1, 4, False], [4, 80, [3, 5, 5, 7], 2, 4, False], [3, 80, [5, 5, 3], 1, 4, False], [3, 80, [5, 5, 5], 1, 4, False], [4, 80, [5, 5, 5, 3], 1, 4, False], [2, 96, [5, 5], 1, 4, False], [3, 96, [5, 5, 5], 1, 4, False], [3, 96, [5, 3, 5], 1, 4, False], [3, 96, [5, 5, 5], 1, 4, False], [3, 192, [3, 3, 3], 2, 4, False], [2, 192, [7, 5], 1, 4, False], [4, 192, [3, 5, 7, 7], 1, 4, False], [2, 192, [7, 3], 1, 4, False], [5, 320, [3, 3, 7, 3, 3], 1, 4, False]]


class Model(nn.Module):
    def __init__(self, se=False, activation="relu", bn_momentum=0.1, l_cfgs_name="SGNAS_A", input_size=224, classes=1000, seg_state=False):
        super(Model, self).__init__()

        if l_cfgs_name == "SGNAS_A":
            l_cfgs = SGNAS_A
        elif l_cfgs_name == "SGNAS_B":
            l_cfgs = SGNAS_B
        elif l_cfgs_name == "SGNAS_C":
            l_cfgs = SGNAS_C

        if input_size == 32:
            self.first = ConvBNRelu(input_channel=3, output_channel=32, kernel=3, stride=1,
                                    pad=3//2, activation=activation, bn_momentum=bn_momentum)
        elif input_size >= 224:
            self.first = ConvBNRelu(input_channel=3, output_channel=32, kernel=3, stride=2,
                                    pad=3//2, activation=activation, bn_momentum=bn_momentum)

        input_channel = 32
        output_channel = 16
        self.first_mb = MBConv(input_channel=input_channel,
                               output_channel=output_channel,
                               expansion=1,
                               kernels=[3],
                               stride=1,
                               activation=activation,
                               split_block=1,
                               se=se,
                               bn_momentum=bn_momentum)
               
        input_channel = output_channel
        self.stages = nn.ModuleList()
        for l_cfg in l_cfgs:
            expansion, output_channel, kernel, stride, split_block, _ = l_cfg
            self.stages.append(MBConv(input_channel=input_channel,
                                      output_channel=output_channel,
                                      expansion=expansion,
                                      kernels=kernel,
                                      stride=stride,
                                      activation=activation,
                                      split_block=split_block,
                                      se=se,
                                      bn_momentum=bn_momentum))
            input_channel = output_channel

        self.last_stage = conv_1x1_bn(input_channel, 1280, activation=activation, bn_momentum=bn_momentum)

        self.classifier_new = nn.Sequential(
                    nn.Sequential(),
                    nn.Linear(1280, classes))

        # ============== Segmentation ==================
        self.seg_state = seg_state
        if self.seg_state:
            self.segmentation_conv_1 = nn.Sequential(
                                        ConvBNRelu(input_channel=1280,
                                                   output_channel=320,
                                                   kernel=3,
                                                   stride=1,
                                                   pad=3//2),
                                        nn.Conv2d(320, 1, 1)
                                                 )
            self.segmentation_conv_2 = nn.Sequential(
                                        ConvBNRelu(input_channel=96,
                                                   output_channel=96,
                                                   kernel=3,
                                                   stride=1,
                                                   pad=3//2),
                                        nn.Conv2d(96, 1, 1)
                                                 )
            self.segmentation_conv_3 = nn.Sequential(
                                        ConvBNRelu(input_channel=40,
                                                   output_channel=40,
                                                   kernel=3,
                                                   stride=1,
                                                   pad=3//2),
                                        nn.Conv2d(40, 1, 1)
                                                 )
        # ==============================================

        self._initialize_weights()

    def forward(self, x):
        x = self.first(x)
        x = self.first_mb(x)
        for i, l in enumerate(self.stages):
            x = l(x)
            if i == 5 and self.seg_state:
                first_seg_out = self.segmentation_conv_3(x)
            if i == 13 and self.seg_state:
                second_seg_out = self.segmentation_conv_2(x)

        x = self.last_stage(x)

        if self.seg_state:
            # ============== Segmentation ====================
            third_seg_out = self.segmentation_conv_1(x)
            # ================================================

        x = x.mean(3).mean(2)
        x = self.classifier_new(x)

        if self.seg_state:
            return x, first_seg_out, second_seg_out, third_seg_out
        return x

    def set_state(self, seg_state):
        self.seg_state = seg_state

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
