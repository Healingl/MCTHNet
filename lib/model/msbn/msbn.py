#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: MCTHNet
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2023/11/1
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from torch import nn


class _ModalitySpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ModalitySpecificBatchNorm, self).__init__()
        self.bns = nn.ModuleList(
            # BN
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_domains)]

        )

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label]

        return bn(x), domain_label


class ModalitySpecificBatchNorm2d(_ModalitySpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))