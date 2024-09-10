# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

class LR_Scheduler:
    def __init__(self, optimizer, warmup_epochs, min_lr, lr, epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.lr = lr
        self.epochs = epochs


    def adjust_learning_rate(self, epoch, lr_decay=0.1):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs 
        else:
            # step decay after warmup
            # lr = self.lr * pow(lr_decay, math.floor(epoch - self.warmup_epochs))
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
