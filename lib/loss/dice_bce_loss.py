# !/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: MSCGNet
# @IDE: PyCharm
# @File: dice_loss.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2020/10/2
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from lib.loss.loss_utils import *


class DiceLoss(nn.Module):
	def __init__(self, eps=1e-6, ignore_index=None):
		super(DiceLoss, self).__init__()
		self.ignore_index = ignore_index
		self.eps = eps

	def forward(self, prediction, target):
		"""
		pred :    shape (N, C, H, W)
		target :  shape (N, H, W) ground truth
		return:  dice loss
		"""
		target = target.long()
		pred = F.softmax(prediction, 1)
		one_hot_label, mask = self.encode_one_hot_label(pred, target)

		assert pred.shape == one_hot_label.shape

		dice_loss = self.cal_dice_loss_by_array(pred, one_hot_label, mask)

		return dice_loss

	def encode_one_hot_label(self, pred, target):
		one_hot_label = pred.detach() * 0
		if self.ignore_index is not None:
			mask = (target == self.ignore_index)
			target = target.clone()
			target[mask] = 0
			one_hot_label.scatter_(1, target.unsqueeze(1), 1)
			mask = mask.unsqueeze(1).expand_as(one_hot_label)
			one_hot_label[mask] = 0
			return one_hot_label, mask
		else:
			# print(type(target))
			# target.unsqueeze(1)在第一维度上扩展
			one_hot_label.scatter_(1, target.unsqueeze(1), 1)  #scatter_(dim: _int, index: Tensor, value:Number), fill input by value
			return one_hot_label, None

	def cal_dice_loss_by_array(self,pred,one_hot_label,mask=None, value_pow=2):
		intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
		union = pred.pow(value_pow) + one_hot_label.pow(value_pow)

		if mask is not None:
			union[mask] = 0

		union = torch.sum(union, dim=(0, 2, 3)) + self.eps

		dice = intersection / union

		dice_loss = 1 - dice.mean()
		return dice_loss



class BCEDiceLoss(DiceLoss):
	def __init__(self, alpha=1, beta=1):
		super(BCEDiceLoss, self).__init__()
		self.alpha = alpha
		self.beta = beta
	def forward(self, pred, target, ):
		"""
		pred :    shape (N, C, H, W)
		target :  shape (N, H, W) ground truth
		return:  dice loss
		"""
		one_hot_label, _ = self.encode_one_hot_label(pred.long(), target.long())
		bce = F.binary_cross_entropy_with_logits(pred.float(), one_hot_label.float())
		input = F.softmax(pred,dim=1)
		dice = self.cal_dice_loss_by_array(pred=input,one_hot_label=one_hot_label)
		return self.alpha * bce + self.beta * dice