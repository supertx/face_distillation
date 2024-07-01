"""
@author supermantx
@date 2024/5/20 10:44
"""
from torch import nn
import torch
import torch.nn.functional as F

from ._base import Distiller


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def fcd_loss(logits_student, logits_teacher):
    # 将特征归一化
    logits_student_norm = normalize(logits_student)
    logits_teacher_norm = normalize(logits_teacher)
    # 计算l2距离
    loss_fcd = F.mse_loss(logits_student_norm, logits_teacher_norm, reduction="mean")
    return loss_fcd


class Fcd(Distiller):

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher, cfg)
        self.fcd_loss_weight = cfg.KD.LOSS.FCD_WEIGHT

    def forward_train(self, index, image, flip_flag, **kwargs):
        logits_student = self.student(image)
        if self.preload:
            logits_teacher = torch.stack(
                [self.teacher_logit[i] if not flag else self.teacher_flip_logit[i] for i, flag in
                 zip(index, flip_flag)]).cuda()
        else:
            with torch.no_grad():
                logits_teacher = self.teacher(image)

        # losses
        loss_fcd = self.fcd_loss_weight * fcd_loss(logits_student, logits_teacher)
        return logits_teacher, logits_student, loss_fcd.unsqueeze(0)
