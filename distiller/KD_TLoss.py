"""
@author supermantx
@date 2024/4/23 11:21
受论文的启发，知识蒸馏中，学生网络不仅需要学到教师网络特征的分布，还要学到特征之间的分布，所以可以考虑使用教师提取出的特征作为w，
并且使用一个queue维护一个负例的队列，增加类间负类的比例。然后间KL loss和设计的margin loss加起来
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import Queue
from ._base import Distiller
import threading


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").mean()
    loss_kd *= temperature ** 2
    return loss_kd


class MarginLoss(nn.Module):
    def __init__(self, cfg):
        super(MarginLoss, self).__init__()
        self.center_queue = Queue(maxsize=cfg.KD.LOSS.QUEUE_SIZE - cfg.DATASET.BATCH_SIZE / 2)
        self.toss_num = 0

    def reset(self):
        self.toss_num = 0

    # def forward(self, logits, centers):
    #     labels = []
    #     # 将新的batch加入队列中，并且保存下标
    #     # todo 想办法提升速度
    #     # queue不太好用，该方法是被多线程调用的，很难获取插入队列时的下标
    #     for center in centers:
    #         if self.center_queue.full():
    #             self.center_queue.get()
    #             self.toss_num += 1
    #         self.center_queue.put(center.cpu())
    #         labels.append(len(self.center_queue.queue))
    #     centers = torch.stack(list(self.center_queue.queue)).cuda()
    #     # 线性层
    #     X = F.linear(F.normalize(logits, eps=1e-7), F.normalize(centers, eps=1e-7), bias=None).cuda()
    #     labels = torch.tensor(labels) - self.toss_num - 1
    #     labels = labels.to(X).long()
    #     return F.cross_entropy(X, labels)

    def forward(self, logits, centers):
        i = len(self.center_queue.queue)
        # 将新的batch加入队列中，并且保存下标
        last_batch = torch.stack(list(self.center_queue.queue)).cuda()
        labels = torch.arange(0, len(centers)) + i
        labels.cuda().long()
        centers = torch.cat((last_batch, centers))
        # 线性层
        X = F.linear(F.normalize(logits, eps=1e-7), F.normalize(centers, eps=1e-7), bias=None).cuda()
        for center in centers:
            if self.center_queue.full():
                self.center_queue.get()
            self.center_queue.put(center.cpu())
        return F.cross_entropy(X, labels)

class KD_TLoss(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD_TLoss, self).__init__(student, teacher, cfg)
        self.temperature = cfg.KD.TEMPERATURE
        # 人脸识别模型只是一个特征提取器，所以用不到softmax的loss，后续可以考虑再增加进行数据的加强
        self.t_loss_weight = cfg.KD.LOSS.T_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.margin_loss = MarginLoss(cfg)

    def forward_train(self, index, image, target, **kwargs):
        logits_student = self.student(image)
        if self.preload:
            logits_teacher = torch.stack([self.teacher_logit[i] for i in index]).cuda()
        else:
            with torch.no_grad():
                logits_teacher = self.teacher(image)

        # losses
        loss_t = self.t_loss_weight * self.margin_loss(logits_student, logits_teacher)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, self.logit_stand
        )
        losses_dict = {
            "loss_t": loss_t,
            "loss_kd": loss_kd.unsqueeze(0),
        }

        self.margin_loss.reset()
        return logits_student, losses_dict
