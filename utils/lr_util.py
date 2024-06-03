"""
@author supermantx
@date 2024/4/18 10:43
"""
import warnings

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler



class StepScaleLR(_LRScheduler):

    def __init__(self, optimizer, stages, gamma=0.1, last_epoch=-1, verbose=False):
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
        self.gamma = gamma
        self.stages = stages
        self.last_stage = min(stages)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0 or self.last_stage == -1:
            return [group["lr"] for group in self.optimizer.param_groups]
        if self.last_epoch == self.last_stage:
            stages = [stage for stage in self.stages if stage > self.last_stage]
            if len(stages) > 0:
                self.last_stage = min(stages)
            else:
                self.last_stage = -1
            return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
        else:
            return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        # 到了第几个stage
        steps = np.sum(self.last_epoch >= np.asarray(self.last_stage))
        return [base_lr * (self.gamma ** steps) for base_lr in self.base_lrs]
