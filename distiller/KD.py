import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher, cfg)
        self.temperature = cfg.KD.TEMPERATURE
        # 人脸识别模型只是一个特征提取器，所以用不到softmax的loss，后续可以考虑再增加进行数据的加强
        # self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 

    def forward_train(self, index, image, flip_flag, target, **kwargs):
        logits_student = self.student(image)
        if self.preload:
            logits_teacher = torch.stack([self.teacher_logit[i] if not flag else self.teacher_flip_logit[i] for i, flag in zip(index, flip_flag)]).cuda()
        else:
            with torch.no_grad():
                logits_teacher = self.teacher(image)

        # losses
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, self.logit_stand
        )

        return logits_teacher, logits_student, loss_kd.unsqueeze(0)
