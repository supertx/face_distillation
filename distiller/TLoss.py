"""
@author: supermantx
@time: 2024/4/24 20:32
实现TLoss的想法
"""
import torch
from torch import nn
from queue import Queue
from torch.nn import functional as F
from torch.nn import DataParallel
import torch.distributed as dist


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, logits, centers, labels):
        # print(labels)
        # print(labels.shape)
        # print(centers.shape)
        # print(logits.shape)
        X = F.linear(F.normalize(logits, eps=1e-7), F.normalize(centers, eps=1e-7), bias=None).cuda()
        res = F.cross_entropy(X, labels)
        return res


class TLoss(nn.Module):
    def __init__(self, cfg):
        super(TLoss, self).__init__()
        self.cfg = cfg
        self.center_queue = Queue(maxsize=cfg.KD.LOSS.QUEUE_SIZE)
        self.t_weight = cfg.KD.LOSS.T_WEIGHT
        self.MarginLoss = DataParallel(MarginLoss().cuda())

    def forward(self, logits, centers):
        l = len(self.center_queue.queue)
        for center in centers:
            if self.center_queue.full():
                self.center_queue.get()
            self.center_queue.put(center.cpu())
            if l < self.cfg.KD.LOSS.QUEUE_SIZE - self.cfg.DATASET.BATCH_SIZE:
                labels = torch.arange(0, len(logits)) + l
            else:
                labels = torch.arange(self.cfg.KD.LOSS.QUEUE_SIZE - len(logits),
                                      self.cfg.KD.LOSS.QUEUE_SIZE)
        labels = labels.cuda().long()
        centers = torch.stack(list(self.center_queue.queue)).cuda()
        return self.t_weight * self.MarginLoss(logits, torch.cat((centers, centers)), labels)


class TLoss_new(nn.Module):
    """重新写，使用DDP"""

    def __init__(self, cfg):
        super(TLoss_new, self).__init__()
        self.cfg = cfg
        self.center_queue = Queue(maxsize=cfg.KD.LOSS.QUEUE_SIZE)

    def forward(self, logits, centers):
        gathered_centers = [torch.zeros_like(centers) for _ in range(self.cfg.DDP.WORLD_SIZE)]
        dist.all_gather(gathered_centers, centers)
        l = len(self.center_queue.queue)
        for center in gathered_centers:
            if self.center_queue.full():
                self.center_queue.get()
            self.center_queue.put(center.cpu())
            if l < self.cfg.KD.LOSS.QUEUE_SIZE - self.cfg.DATASET.BATCH_SIZE:
                labels = torch.arange(0, len(logits)) + l
            else:
                labels = torch.arange(self.cfg.KD.LOSS.QUEUE_SIZE - len(logits),
                                      self.cfg.KD.LOSS.QUEUE_SIZE)
        centers = torch.stack(list(self.center_queue.queue)).cuda()
        labels = labels.cuda().long()
        labels = labels.chunks(self.cfg.DDP.WORLD_SIZE)[dist.get_rank()]
        return self.MarginLoss(logits, centers, labels)
