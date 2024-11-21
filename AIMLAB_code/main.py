# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import numpy as np
import pandas as pd
import os
import time
import random
import timm
from pathlib import Path
import wandb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import set_protocols_for_oversampling, MyDataset, MyDataset_2Aug

import models_patchvit as models_patchvit
from engine import test_every_epoch
from SupCon_loss import LivePatchSpoofSampleLoss_ViT

import util.misc as misc
import util.lr_sched as lr_sched

from PIL import Image
import albumentations as A

from margin_loss import CombinedMarginLoss, ArcFace, CosFace
from models_patchvit import NormalizedLinear


class AlbumentationTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img) 
        augmented = self.transform(image=img) 
        img = augmented['image']
        img = Image.fromarray(img)
        return img

albumentations_transform = A.Compose([
    A.Rotate(limit=(-180, 180), interpolation=1, border_mode=2, value=None, mask_value=None, always_apply=False, p=1),
])

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--dropout_ratio', type=float, default=0.)

    parser.add_argument('--margin_loss',type=str, default='arcface', help='combined, arcface, cosface')
    parser.add_argument('--margin_ce_loss_weight',type=float, default=0.3)
    parser.add_argument('--supcon_loss_weight',type=float, default=0.7)

    parser.add_argument('--pretrained',type=str, default='imagenet')
    parser.add_argument('--is_debug',  action='store_true')
    parser.add_argument('--protocol', type=str, default="Partial_Eye", help='Partial_Eye, Partial_FunnyeyeGlasses, Partial_Mouth, Partial_PaperGlasses')
    parser.add_argument('--batch_size', default=128, type=int,help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_false',help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=None, metavar='LR',help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',help='epochs to warmup LR')
    parser.add_argument('--lr_sched',type=str, default='adjust')

    parser.add_argument('--output_dir', default='./output_dir',help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',help='url used to set up distributed training')

    return parser

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def get_test_dataloader(attack_name, transform_test):
    print(f'Test Attack Type: {attack_name}')
    test_csv = pd.read_csv(f'./csv/siw/filtered/filtered_test_{attack_name}.csv')
    dataset_test = MyDataset(test_csv, transform=transform_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=2048,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    return data_loader_test

def get_dataloader(src1_live_df, src1_spoof_df, transform, transform2, args):
    src1_live_dataset = MyDataset_2Aug(src1_live_df, transform=transform, transform2=transform2)
    src1_spoof_dataset = MyDataset_2Aug(src1_spoof_df, transform=transform, transform2=transform2)

    src1_live_loader = torch.utils.data.DataLoader(src1_live_dataset, batch_size = args.batch_size//2, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    src1_spoof_loader = torch.utils.data.DataLoader(src1_spoof_dataset, batch_size = args.batch_size//2, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    return src1_spoof_loader, src1_live_loader

unseen_attack_types_dict = {
    '2D':     ['Partial_FunnyeyeGlasses', 'Partial_PaperGlasses', 'Partial_Mouth', 'Partial_Eye',
                'Makeup_Obfuscation', 'Makeup_Cosmetic', 'Makeup_Impersonation',
                'Mask_PaperMask', 'Mask_HalfMask', 'Mannequin', 'Silicone', 'Mask_TransparentMask'],

    'Makeup': ['Partial_FunnyeyeGlasses', 'Partial_PaperGlasses', 'Partial_Mouth', 'Partial_Eye',
                'Mask_PaperMask', 'Mask_HalfMask', 'Mannequin', 'Silicone', 'Mask_TransparentMask',
                'Replay', 'Paper'],

    'Mask':   ['Partial_FunnyeyeGlasses', 'Partial_PaperGlasses', 'Partial_Mouth', 'Partial_Eye',
                'Makeup_Obfuscation', 'Makeup_Cosmetic', 'Makeup_Impersonation',
                'Replay', 'Paper'],

    'Partial': ['Makeup_Obfuscation', 'Makeup_Cosmetic', 'Makeup_Impersonation',
                'Mask_PaperMask', 'Mask_HalfMask', 'Mannequin', 'Silicone', 'Mask_TransparentMask',
                'Replay', 'Paper']}

def main(args):
    wandb.run.name = f'[SiW_HighLevel_1to3]ViT_trial6_{args.protocol}'
    
    args.output_dir = os.path.join(args.output_dir, wandb.run.name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_train_v2 = transforms.Compose([
            AlbumentationTransform(albumentations_transform),
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
     
    transform_test = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_csv = pd.read_csv(f'./csv/siw/high_level/train_{args.protocol}_All.csv')
    src1_live_df = train_csv[train_csv['label']==1]
    src1_spoof_df = train_csv[train_csv['label']==0]

    src1_train_dataloader_fake, src1_train_dataloader_real = get_dataloader(src1_live_df, src1_spoof_df, \
                                                                            transform = transform_train, \
                                                                            transform2 = transform_train_v2, \
                                                                            args=args)

    print(f'\n** Num of Unseen Attack Type of {args.protocol} ==> {len(unseen_attack_types_dict[args.protocol])}')
    unseen_attack_types = unseen_attack_types_dict[args.protocol]
    test_dataloader_list = [get_test_dataloader(attack_type, transform_test) for attack_type in unseen_attack_types]

    if not args.pretrained:
        model = models_patchvit.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        model.to(device)
    elif args.pretrained == 'imagenet':
        patchvit_model = models_patchvit.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, dropout_ratio=args.dropout_ratio)
        pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)

        pretrained_model_dict = pretrained_model.state_dict()
        patch_vit_model_dict = patchvit_model.state_dict()

        pretrained_weights = {k: v for k, v in pretrained_model_dict.items() if k in patch_vit_model_dict and 'head' not in k}
        patch_vit_model_dict.update(pretrained_weights)

        patchvit_model.load_state_dict(patch_vit_model_dict)

        patchvit_model.head = NormalizedLinear(patchvit_model.head.in_features, 2)
        # patchvit_model.head = NormalizedLinear(patchvit_model.head.in_features, 1)
        model = patchvit_model.to(device)
    else:
        patchvit_model = models_patchvit.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        pretrained_model = timm.create_model('vit_base_patch16_clip_224', pretrained=True, global_pool='avg', num_classes=2)

        timm_state_dict = pretrained_model.state_dict()
        timm_state_dict_modified = {k: v for k, v in timm_state_dict.items() if 'norm_pre' not in k}

        patchvit_state_dict = patchvit_model.state_dict()

        for name, param in patchvit_state_dict.items():
            if name in timm_state_dict_modified and 'head' not in name:
                patchvit_state_dict[name].copy_(timm_state_dict_modified[name])

        patchvit_model.load_state_dict(patchvit_state_dict, strict=False)

        patchvit_model.head = NormalizedLinear(patchvit_model.head.in_features, 2)
        # patchvit_model.head = NormalizedLinear(patchvit_model.head.in_features, 1)
        model = patchvit_model.to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    param_groups = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler()

    if args.margin_loss == 'combined':
        margin_loss = CombinedMarginLoss(s=64., m1=1.0, m2=0.5, m3=0.0)
    elif args.margin_loss == 'arcface':
        margin_loss = ArcFace()
    elif args.margin_loss == 'cosface':
        margin_loss = CosFace()

    ce_loss = nn.CrossEntropyLoss()
    asym_supcon_criterion = LivePatchSpoofSampleLoss_ViT(spoof_average_pool=False)

    iter_per_epoch = 10

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)

    min_hter = 10000
    max_iter = 4000
    epoch = 1

    for iter_num in range(max_iter+1):
        if (iter_num % src1_iter_per_epoch_real == 0):
            src1_train_iter_real = iter(src1_train_dataloader_real)
        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1

        model.train()
        optimizer.zero_grad()
        
        # if args.adjust_lr:
        #     print(f'Adjust LR')
        #     lr_sched.adjust_learning_rate(optimizer, iter_num, max_iter, args) 

        src1_img_real, src1_img_real_v2, src1_label_real = next(src1_train_iter_real)
        src1_img_real, src1_img_real_v2 = src1_img_real.cuda(), src1_img_real_v2.cuda()
        src1_label_real = src1_label_real.cuda()

        src1_img_fake, src1_img_fake_v2, src1_label_fake = next(src1_train_iter_fake)
        src1_img_fake, src1_img_fake_v2 = src1_img_fake.cuda(), src1_img_fake_v2.cuda()
        src1_label_fake = src1_label_fake.cuda()

        samples_v1 = torch.cat([src1_img_real, src1_img_fake], dim=0)
        samples_v2 = torch.cat([src1_img_real_v2, src1_img_fake_v2], dim=0)
        targets = torch.cat([src1_label_real, src1_label_fake], dim=0)

        samples_combined = torch.cat((samples_v1, samples_v2), dim=0)
        targets_combined = torch.cat((targets, targets), dim=0)

        samples_combined = samples_combined.to(device, non_blocking=True)
        targets_combined = targets_combined.to(device)

        with torch.cuda.amp.autocast():
            feature, logits = model(samples_combined, is_train=True)

        margin_logits = margin_loss(logits, targets_combined) 
        margin_ce_loss = ce_loss(margin_logits, targets_combined)
        asym_supcon_loss = asym_supcon_criterion(feature[:, 1:, :], targets_combined)

        loss = (args.margin_ce_loss_weight*margin_ce_loss + args.supcon_loss_weight*asym_supcon_loss) / args.accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)

        # if (iter_num != 0 and iter_num % 40 == 0):
        if (iter_num != 0 and iter_num % 80 == 0):

            avg_hter = 0.
            for idx, test_data_loader in enumerate(test_dataloader_list):
                attack_name = unseen_attack_types[idx]
                print(f'TEST -- Attack Name: {attack_name}')

                hter, optimal_test_apcer, optimal_test_npcer, auc, eer = test_every_epoch(model, test_data_loader)
                avg_hter += hter

                print(f'EPOCH: {epoch}, HTER: {hter}, APCER: {optimal_test_apcer}, NPCER: {optimal_test_npcer}, AUC: {auc}, EER: {eer} \n')
                
                wandb.log({f'HTER_{attack_name}': hter, f'AUC_{attack_name}': auc, f'EER_{attack_name}': eer, \
                           f'APCER_{attack_name}': optimal_test_apcer, f'NPCER_{attack_name}': optimal_test_npcer}, step=epoch)

            avg_hter /= float(len(unseen_attack_types))
            print(f'{epoch}EPOCH -- Averaged HTER: {avg_hter}\n')
            wandb.log({'AVG_HTER': avg_hter}, step=epoch)

            if min_hter >= avg_hter or epoch+1==args.epochs:
                min_hter = avg_hter
                misc.save_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        if (iter_num + 1) % args.accum_iter == 0:
            optimizer.zero_grad()
            torch.cuda.synchronize()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    wandb.init(project='TRIAL', entity='jimin2')     
    
    main(args)
