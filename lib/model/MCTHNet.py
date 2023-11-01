#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 22-6-21
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # # 

import math
import numpy as np
from lib.model.msbn.layers import maxpool2D, conv2d, deconv2d, relu

from lib.model.msbn.msbn import ModalitySpecificBatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
import torch


# bilinear interpolation
class MyUpsample2(nn.Module):
    def __init__(self, ):
        super(MyUpsample2, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.up(x)

def normalization(planes, norm='gn', num_domains=None, momentum=0.1):
    if norm == 'msbn':
        m = ModalitySpecificBatchNorm2d(planes, num_domains=num_domains, momentum=momentum)
    elif norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        # origin 32
        m = nn.GroupNorm(32, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m


class MSSC(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, shuffle=False, num_domains=None, momentum=0.1):
        super(MSSC, self).__init__()
        self.shuffle = shuffle

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes//2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = normalization(planes//2, norm, num_domains, momentum=momentum)

        self.conv2_3x3 = nn.Conv2d(planes//2, planes//2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_3x3 = normalization(planes//2, norm, num_domains, momentum=momentum)

        self.conv2_5x5 = nn.Conv2d(planes//2, planes//2, kernel_size=5, stride=1, padding=2, bias=True)
        self.bn2_5x5 = normalization(planes//2, norm, num_domains, momentum=momentum)

        self.conv_fusion = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_fusion = normalization(planes, norm, num_domains, momentum=momentum)

        r = 2
        L = 8
        d = max(int(planes//2 / r), L)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(planes//2, d, bias=True)
        # self.bn_fc1 = normalization(d, norm, num_domains, momentum=momentum)
        self.fc2_1 = nn.Linear(d, planes//2, bias=True)
        self.fc2_2 = nn.Linear(d, planes//2, bias=True)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x, weights=None, layer_idx=None, domain_label=None):
        if not self.first:
            x = maxpool2D(x, kernel_size=2)

        if weights == None:
            weight_1, bias_1 = self.conv1.weight, self.conv1.bias
            weight_2_3x3, bias_2_3x3 = self.conv2_3x3.weight, self.conv2_3x3.bias
            weight_2_5x5, bias_2_5x5 = self.conv2_5x5.weight, self.conv2_5x5.bias
            weight_fusion, bias_fusion = self.conv_fusion.weight, self.conv_fusion.bias

        else:
            assert False
        #
        # layer 1 conv, bn
        input_feature = conv2d(x,  weight_1, bias_1, kernel_size=3, stride=1, padding=1)
        if domain_label is not None:
            input_feature, _ = self.bn1(input_feature, domain_label)
        else:
            input_feature = self.bn1(input_feature)
        input_feature = relu(input_feature)

        identity = input_feature

        d1 = conv2d(input_feature,  weight_2_3x3, bias_2_3x3, kernel_size=3, stride=1, padding=1)
        d1, _ = self.bn2_3x3(d1, domain_label)
        d1 = relu(d1)

        d2 = conv2d(input_feature,  weight_2_5x5, bias_2_5x5, kernel_size=5, stride=1, padding=2)
        d2, _ = self.bn2_5x5(d2, domain_label)
        d2 = relu(d2)

        # # # # # # # # # # # #
        # scale attention
        # # # # # # # # # # # #
        fea_U = d1 + d2

        fea_s = self.gap(fea_U).squeeze_(dim=-1).squeeze_(dim=-1)
        feat_z = self.fc1(fea_s)

        vector_1 = self.fc2_1(feat_z)
        vector_2 = self.fc2_2(feat_z)

        scale_attention_vectors = torch.cat([vector_1.unsqueeze_(dim=1), vector_2.unsqueeze_(dim=1)], dim=1)
        # (2,2,32)
        scale_attention_vectors = F.softmax(scale_attention_vectors, dim=1)
        # (2,2,32,1,1)
        scale_attention_vectors = scale_attention_vectors.unsqueeze(-1).unsqueeze(-1)

        # (2,32,1,1)
        scale_attention_vector1 = scale_attention_vectors[:, 0, :, :, :]
        scale_attention_vector2 = scale_attention_vectors[:, 1, :, :, :]

        d1 = d1 * scale_attention_vector1
        d2 = d2 * scale_attention_vector2

        # # # # # # # # # # # #
        # # # # # # # # # # # #
        fusion_feature = torch.cat([d1,d2], dim=1)
        fusion_feature = self.channel_shuffle(fusion_feature, 2)
        fusion_feature = conv2d(fusion_feature,  weight_fusion, bias_fusion, kernel_size=3, stride=1, padding=1)
        fusion_feature, _ = self.bn_fusion(fusion_feature, domain_label)
        fusion_feature = relu(fusion_feature)

        return fusion_feature

class ConvU(nn.Module):
    def __init__(self, planes, norm='bn', first=False, num_domains=None, momentum=0.1):
        super(ConvU, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2 * planes, planes, 3, 1, 1, bias=True)
            self.bn1 = normalization(planes, norm, num_domains, momentum=momentum)

        self.pool = MyUpsample2()
        self.conv2 = nn.Conv2d(planes, planes // 2, 1, 1, 0, bias=True)
        self.bn2 = normalization(planes // 2, norm, num_domains, momentum=momentum)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3 = normalization(planes, norm, num_domains, momentum=momentum)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev, weights=None, layer_idx=None, domain_label=None):

        if weights == None:
            if not self.first:
                weight_1, bias_1 = self.conv1.weight, self.conv1.bias
            weight_2, bias_2 = self.conv2.weight, self.conv2.bias
            weight_3, bias_3 = self.conv3.weight, self.conv3.bias

        else:
            if not self.first:
                weight_1, bias_1 = weights[layer_idx + '.conv1.weight'], weights[layer_idx + '.conv1.bias']
            weight_2, bias_2 = weights[layer_idx + '.conv2.weight'], weights[layer_idx + '.conv2.bias']
            weight_3, bias_3 = weights[layer_idx + '.conv3.weight'], weights[layer_idx + '.conv3.bias']

        # layer 1 conv, bn, relu
        if not self.first:
            x = conv2d(x, weight_1, bias_1, )
            if domain_label is not None:
                x, _ = self.bn1(x, domain_label)
            else:
                x = self.bn1(x)
            x = relu(x)

        # upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = conv2d(y, weight_2, bias_2, kernel_size=1, stride=1, padding=0)
        if domain_label is not None:
            y, _ = self.bn2(y, domain_label)
        else:
            y = self.bn2(y)
        y = relu(y)

        # concatenation of two layers
        y = torch.cat([prev, y], 1)

        # layer 3 conv, bn
        y = conv2d(y, weight_3, bias_3)
        if domain_label is not None:
            y, _ = self.bn3(y, domain_label)
        else:
            y = self.bn3(y)
        y = relu(y)

        return y

from lib.model.layers.MIViTblock import MIViTBlock

class MCTHNet(nn.Module):
    def __init__(self, in_chns=1, base_filter=16, norm='msbn', class_num=2, momentum=0.1):
        super(MCTHNet, self).__init__()
        # CT AND MR
        num_domains = 2
        c = in_chns
        n = base_filter
        num_classes = class_num

        self.MSSC1 = MSSC(c, n, norm, first=True, num_domains=num_domains, momentum=momentum)
        self.MSSC2 = MSSC(n, 2 * n, norm, num_domains=num_domains, momentum=momentum)
        self.MSSC3 = MSSC(2 * n, 4 * n, norm, num_domains=num_domains, momentum=momentum)
        self.MSSC4 = MSSC(4 * n, 8 * n, norm, num_domains=num_domains, momentum=momentum)
        self.MSSC5 = MSSC(8 * n, 16 * n, norm, num_domains=num_domains, momentum=momentum)

        self.trans_block_invariant = MIViTBlock(16 * n, depth=4)

        self.convu4 = ConvU(16 * n, norm, first=True, num_domains=num_domains, momentum=momentum)
        self.convu3 = ConvU(8 * n, norm, num_domains=num_domains, momentum=momentum)
        self.convu2 = ConvU(4 * n, norm, num_domains=num_domains, momentum=momentum)
        self.convu1 = ConvU(2 * n, norm, num_domains=num_domains, momentum=momentum)

        self.seg1 = nn.Conv2d(2 * n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, modality='ct'):
        if modality == 'ct':
            domain_label = 0
        elif modality == 'mr':
            domain_label = 1
        else:
            assert False

        x1 = self.MSSC1(x, domain_label=domain_label)
        x2 = self.MSSC2(x1, domain_label=domain_label)
        x3 = self.MSSC3(x2, domain_label=domain_label)
        x4 = self.MSSC4(x3, domain_label=domain_label)
        x5 = self.MSSC5(x4, domain_label=domain_label)

        modal_spec_feat = x5
        modal_invar_feat = self.trans_block_invariant(x5)
        # print('modal_invar_feat.shape', modal_invar_feat.shape)
        enc_feat = modal_spec_feat + modal_invar_feat

        # print('x5', x5.shape)
        y4 = self.convu4(enc_feat, x4, domain_label=domain_label)
        # print('y4.shape', y4.shape)
        y3 = self.convu3(y4, x3, domain_label=domain_label)
        # print('y3.shape', y3.shape)
        y2 = self.convu2(y3, x2, domain_label=domain_label)
        # print('y2.shape', y2.shape)
        y1 = self.convu1(y2, x1, domain_label=domain_label)
        # print('y1.shape', y1.shape)

        y1_pred = conv2d(y1, self.seg1.weight, self.seg1.bias, kernel_size=None, stride=1, padding=0)

        return y1_pred


from lib.model.layers.utils import count_param

if __name__ == '__main__':
    n_modal = 1
    n_classes = 5

    net = MCTHNet(in_chns=n_modal, class_num=n_classes, base_filter=32, norm='msbn')
    # print(net)
    param = count_param(net)
    print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))

    input_tensor = torch.rand(1, n_modal, 256, 256)
    #
    out_tensor= net(input_tensor, modality='mr')
    print(out_tensor.shape)