# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

def calculate_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER

def test_every_epoch(model, test_data_loader):
    optimal_test_hter = 0
    test_auc_score = 0

    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for step, (samples, label) in enumerate(test_data_loader):
            if step%20 == 0:
                print(f'TEST ... {step} / {len(test_data_loader)}')
            samples, label = samples.cuda(), label.cuda()

            with torch.cuda.amp.autocast():
                _, logits = model(samples, is_train=False)  # logits.shape: [128, 2]
                test_preds += torch.softmax(logits, axis=1).detach().cpu().numpy().tolist()
                test_labels += label.detach().cpu().numpy().tolist()

        test_auc_score = roc_auc_score(test_labels, [i[1] for i in test_preds])
        eer = calculate_eer(test_labels, [i[1] for i in test_preds])
        _, _, thresholds = roc_curve(test_labels, [i[1] for i in test_preds])

        optimal_err_cand = []
        optimal_apcer_cand = []
        optimal_npcer_cand = []

        print(f'len(thresholds): {len(thresholds)}')
        for thresh in thresholds:
            _conf_mat = confusion_matrix(test_labels, [1 if i[1] >= thresh else 0 for i in test_preds])
            _tn, _fp, _fn, _tp = _conf_mat.ravel()
            _test_apcer = _fp / (_fp + _tn)
            _test_npcer = _fn / (_fn + _tp)
            _test_acer = (_test_apcer + _test_npcer) / 2
            optimal_err_cand.append(_test_acer)
            optimal_apcer_cand.append(_test_apcer)
            optimal_npcer_cand.append(_test_npcer)
            
        optimal_idx = np.argmin(optimal_err_cand)
        optimal_test_hter = optimal_err_cand[optimal_idx]
        optimal_apcer = optimal_apcer_cand[optimal_idx]
        optimal_npcer = optimal_npcer_cand[optimal_idx]

    return optimal_test_hter, optimal_apcer, optimal_npcer, test_auc_score, eer
