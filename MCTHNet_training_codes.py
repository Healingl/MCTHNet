#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
from tqdm import tqdm


import time

current_time = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())))



def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return model_yaml_config['consistency'] * sigmoid_rampup(epoch, model_yaml_config['consistency_rampup'])


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


if __name__ == "__main__":
    model_yaml_config = {
        'model_name': 'MCTHNet',
        'cate_to_label_dict': {'bg': 0, "Spleen": 1, 'RightKidney': 2, 'LeftKidney': 3, 'Liver': 4},
        'label_to_cate_dict': {'0': 'bg', '1': 'Spleen', '2': 'RightKidney', '3': 'LeftKidney', '4': 'Liver'},
        'cal_class_list': [1, 2, 3, 4],

        'input_image_size': [256, 256],

        'input_channels': 1,
        'num_classes': 5,

        'base_filter': 32,
        'use_aug': True,

        'train_labeled_batch_size': 2,
        'train_unlabeled_batch_size': 2,

        'num_epochs': 300,
        'eval_metric_epoch': 1,
        'step_epoch_size': 50,
        'step_gamma': 0.5,
        'num_workers': 0,
        'seed': 2022,
        'cudnn_enabled': True,
        'cudnn_benchmark': False,
        'cudnn_deterministic': True,

        # optimizer:
        'opt_name': 'AdamW',
        'lr_schedule': True,
        'lr_seg': 0.0002,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'betas': [0.9, 0.99],

        # consistency param
        'ema_decay': 0.99,
        'consistency': 0.1,
        'consistency_rampup': 200
    }


    # unpaired CT and MR dataset
    # ct
    labeled_ct_slices_dataset = ...
    unlabeled_ct_slices_dataset = ...

    # mr
    labeled_mr_slices_dataset = ...
    unlabeled_mr_slices_dataset = ...

    softmax = lambda x: F.softmax(x, dim=1)

    labeled_ct_slices_loader = DataLoader(labeled_ct_slices_dataset,
                                          shuffle=True,
                                          batch_size=model_yaml_config['train_labeled_batch_size'],
                                          num_workers=0,drop_last=True)
    unlabeled_ct_slices_loader = DataLoader(unlabeled_ct_slices_dataset,
                                            shuffle=True,
                                            batch_size=model_yaml_config['train_unlabeled_batch_size'],
                                            num_workers=0,drop_last=True)

    labeled_mr_slices_loader = DataLoader(labeled_mr_slices_dataset,
                                          shuffle=True,
                                          batch_size=model_yaml_config['train_labeled_batch_size'],
                                          num_workers=0,drop_last=True)
    unlabeled_mr_slices_loader = DataLoader(unlabeled_mr_slices_dataset,
                                            shuffle=True,
                                            batch_size=model_yaml_config['train_unlabeled_batch_size'],
                                            num_workers=0,drop_last=True)

    MODEL_DIR = os.path.join(model_yaml_config['workdir'], 'model')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Setup Model
    print('building models ...')
    assert model_yaml_config['num_classes'] == len(model_yaml_config['cate_to_label_dict']), 'Error Class Num: %s' % (
        model_yaml_config['num_classes'])


    def kaiming_normal_init_weight(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return model


    def xavier_normal_init_weight(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return model


    from lib.model.MCTHNet import MCTHNet
    # student
    seg_unified_model = MCTHNet(in_chns=model_yaml_config['input_channels'],
                                         class_num=model_yaml_config['num_classes'],
                                         base_filter=model_yaml_config['base_filter']).cuda()
    # teacher
    seg_teacher_model = MCTHNet(in_chns=model_yaml_config['input_channels'],
                                         class_num=model_yaml_config['num_classes'],
                                         base_filter=model_yaml_config['base_filter']).cuda()

    # different initialization
    seg_unified_model = kaiming_normal_init_weight(seg_unified_model)
    seg_teacher_model = xavier_normal_init_weight(seg_teacher_model)

    # Load or Save Model
    model_dict = {}
    model_dict['seg_unified_model'] = seg_unified_model
    model_dict['seg_teacher_model'] = seg_teacher_model

    seg_unified_model_opt = torch.optim.AdamW(seg_unified_model.parameters(),
                                              lr=model_yaml_config['lr_seg'],
                                              weight_decay=model_yaml_config['weight_decay'],
                                              betas=model_yaml_config['betas']
                                              )

    seg_teacher_model_opt = torch.optim.AdamW(seg_teacher_model.parameters(),
                                              lr=model_yaml_config['lr_seg'],
                                              weight_decay=model_yaml_config['weight_decay'],
                                              betas=model_yaml_config['betas']
                                              )

    lr_unified_scheduler_seg = torch.optim.lr_scheduler.StepLR(seg_unified_model_opt,
                                                               step_size=model_yaml_config['step_epoch_size'],
                                                               gamma=model_yaml_config['step_gamma'])
    lr_teacher_scheduler_seg = torch.optim.lr_scheduler.StepLR(seg_teacher_model_opt,
                                                               step_size=model_yaml_config['step_epoch_size'],
                                                               gamma=model_yaml_config['step_gamma'])

    from torch.nn.modules.loss import CrossEntropyLoss
    from lib.loss.dice_bce_loss import DiceLoss

    # softmax and pow 2 dice
    dice_loss = DiceLoss()
    # ce
    ce_loss = CrossEntropyLoss()

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # Training
    # # # # # # # # # # # # # # # # # # # # # # # # #
    best_dice = 0
    best_ct_dice = 0
    best_mr_dice = 0
    iter_num = 0

    for current_epoch in range(model_yaml_config['num_epochs']):
        # ct
        train_labeled_ct_slices_loader_iter = iter(labeled_ct_slices_loader)
        train_unlabeled_ct_slices_loader_iter = iter(unlabeled_ct_slices_loader)

        # mr
        train_labeled_mr_slices_loader_iter = iter(labeled_mr_slices_loader)
        train_unlabeled_mr_slices_loader_iter = iter(unlabeled_mr_slices_loader)


        epoch_all_iter_num = min(len(train_labeled_ct_slices_loader_iter),
                                 len(train_unlabeled_ct_slices_loader_iter),
                                 len(train_labeled_mr_slices_loader_iter),
                                 len(train_unlabeled_mr_slices_loader_iter),
                                 )
        modality_list = ['ct', 'mr']

        print('[Epoch %s/%s] [Modality: %s, Num: %s]' % (
            current_epoch, model_yaml_config['num_epochs'], str(modality_list), epoch_all_iter_num))
        max_iterations = epoch_all_iter_num * model_yaml_config['num_epochs']

        for current_epoch_iter in tqdm(range(epoch_all_iter_num), total=epoch_all_iter_num):
            iter_num += 1
            seg_unified_model.train()
            seg_teacher_model.train()

            # # # # # # # # # # # # # # # # # # # # # # # # #
            # # Supervised Seg Loss
            # # # # # # # # # # # # # # # # # # # # # # # # #
            total_loss = None
            for current_modality in modality_list:
                # zero grade
                seg_unified_model_opt.zero_grad()
                seg_teacher_model_opt.zero_grad()

                if current_modality == 'ct':
                    current_label = 1
                    labeled_ct_feature_tensor, labeled_ct_seg_gt_tensor = next(train_labeled_ct_slices_loader_iter)
                    unlabeled_ct_feature_tensor, _ = next(train_unlabeled_ct_slices_loader_iter)

                    labeled_ct_feature_tensor, labeled_ct_seg_gt_tensor, unlabeled_ct_feature_tensor = labeled_ct_feature_tensor.cuda(
                        non_blocking=True), \
                                                                                                       labeled_ct_seg_gt_tensor.cuda(
                                                                                                           non_blocking=True), \
                                                                                                       unlabeled_ct_feature_tensor.cuda(
                                                                                                           non_blocking=True)

                    labeled_input_feature_tensor, labeled_input_seg_gt_tensor, unlabeled_input_feature_tensor = labeled_ct_feature_tensor, labeled_ct_seg_gt_tensor, unlabeled_ct_feature_tensor

                elif current_modality == 'mr':
                    current_label = 0
                    # mr
                    labeled_mr_feature_tensor, labeled_mr_seg_gt_tensor = next(train_labeled_mr_slices_loader_iter)
                    unlabeled_mr_feature_tensor, _ = next(train_unlabeled_mr_slices_loader_iter)

                    labeled_mr_feature_tensor, labeled_mr_seg_gt_tensor, unlabeled_mr_feature_tensor = labeled_mr_feature_tensor.cuda(
                        non_blocking=True), \
                                                                                                       labeled_mr_seg_gt_tensor.cuda(
                                                                                                           non_blocking=True), \
                                                                                                       unlabeled_mr_feature_tensor.cuda(
                                                                                                           non_blocking=True)

                    labeled_input_feature_tensor, labeled_input_seg_gt_tensor, unlabeled_input_feature_tensor = labeled_mr_feature_tensor, labeled_mr_seg_gt_tensor, unlabeled_mr_feature_tensor
                else:
                    assert False

                # supervised segmentation
                labeled_student_pred_outputs = seg_unified_model(labeled_input_feature_tensor,
                                                                                           modality=current_modality)
                loss_student_sup_seg = ce_loss(labeled_student_pred_outputs, labeled_input_seg_gt_tensor) + dice_loss(
                    labeled_student_pred_outputs, labeled_input_seg_gt_tensor)

                labeled_teacher_pred_outputs = seg_teacher_model(labeled_input_feature_tensor,
                                                                                           modality=current_modality)

                loss_teacher_sup_seg = ce_loss(labeled_teacher_pred_outputs, labeled_input_seg_gt_tensor) + dice_loss(
                    labeled_teacher_pred_outputs, labeled_input_seg_gt_tensor)

                #
                loss_sup_seg = 0.5 * loss_student_sup_seg + 0.5 * loss_teacher_sup_seg

                # semi-supervised unlabeled scans

                # student outpus
                unlabeled_student_outputs = seg_unified_model(
                    unlabeled_input_feature_tensor, modality=current_modality)
                unlabeled_student_soft_outputs = torch.softmax(unlabeled_student_outputs, dim=1)

                unlabeled_teacher_outputs = seg_teacher_model(
                    unlabeled_input_feature_tensor, modality=current_modality)
                unlabeled_teacher_soft_outputs = torch.softmax(unlabeled_teacher_outputs, dim=1)

                pseudo_outputs_student = torch.argmax(unlabeled_student_soft_outputs.detach(), dim=1, keepdim=False)
                pseudo_outputs_teacher = torch.argmax(unlabeled_teacher_soft_outputs.detach(), dim=1, keepdim=False)

                consistency_weight = get_current_consistency_weight(current_epoch)

                pseudo_supervision1 = ce_loss(unlabeled_student_outputs, pseudo_outputs_teacher)
                pseudo_supervision2 = ce_loss(unlabeled_teacher_outputs, pseudo_outputs_student)
                loss_semi_seg = consistency_weight * (pseudo_supervision1 + pseudo_supervision2)
                total_loss = loss_sup_seg + loss_semi_seg

                total_loss.backward()

                seg_unified_model_opt.step()
                seg_teacher_model_opt.step()

                seg_lr = seg_unified_model_opt.param_groups[0]['lr']
                print('[Epoch %d / %d], [iter %d / %d], [consistency weight and loss: %.4f %.4f], [seg_lr: %.7f], [loss all: %.4f]' % (
                        current_epoch, model_yaml_config['num_epochs'], current_epoch_iter, current_epoch_iter,
                        consistency_weight, loss_semi_seg.item(), seg_lr,
                        total_loss.item()))

        lr_unified_scheduler_seg.step()
        lr_teacher_scheduler_seg.step()