"""
@author supermantx
@date 2024/5/20 10:44
"""
from torch import nn
import torch
import torch.nn.functional as F

from ._base import Distiller


class Fcd(Distiller):

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher, cfg)
        self.fcd_loss_weight = cfg.KD.LOSS.FCD_WEIGHT

    def fcd_loss(self, logits_student, logits_teacher):
        # 将特征归一化
        logits_student_norm = logits_student / torch.norm(logits_student, dim=1, keepdim=True)
        logits_teacher_norm = logits_teacher / torch.norm(logits_teacher, dim=1, keepdim=True)
        # 计算l2距离
        loss_fcd = F.mse_loss(logits_student_norm, logits_teacher_norm, reduction="mean")
        return loss_fcd

    def forward_train(self, index, image, flip_flag, target, **kwargs):
        logits_student = self.student(image)
        if self.preload:
            logits_teacher = torch.stack(
                [self.teacher_logit[i] if not flag else self.teacher_flip_logit[i] for i, flag in
                 zip(index, flip_flag)]).cuda()
        else:
            with torch.no_grad():
                logits_teacher = self.teacher(image)

        # losses
        loss_fcd = self.fcd_loss_weight * self.fcd_loss(logits_student, logits_teacher)
        return logits_teacher, logits_student, loss_fcd.unsqueeze(0)
