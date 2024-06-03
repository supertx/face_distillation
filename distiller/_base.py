import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as F
from tqdm import tqdm
from torch.nn import DataParallel
import threading
import queue
import time
from torch.utils.data import DataLoader
from dataset import MixFlipFaceDataset


class BackgroundLoader(threading.Thread):
    def __init__(self, dataset, batch_size, max_prefetch=4096):
        super(BackgroundLoader, self).__init__()
        self.queue = queue.Queue(max_prefetch)
        self.dataset = dataset
        self.batch_size = batch_size
        self.daemon = True
        self.start()

    def run(self):
        for i in range(len(self.dataset)):
            self.queue.put(self.dataset[i][1])
        self.queue.put(None)

    def __next__(self):
        # 一次性给一个batch
        items = []
        for _ in range(self.batch_size):
            item = self.queue.get()
            if item is None:
                break
            items.append(item)
        return torch.stack(items).cuda()

    def __iter__(self):
        return self


class Distiller(nn.Module):

    def __init__(self, student, teacher, cfg):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.teacher_logit = []
        self.teacher_flip_logit = []
        self.preload = cfg.PRELOAD_TEACHER_LOGIT
        preload_batch = cfg.PRELOAD_BATCH
        # loader = BackgroundLoader(dataset, preload_batch, max_prefetch=preload_batch * 4)
        mix_flip_face_dataset = MixFlipFaceDataset(cfg.DATASET.DATA_DIR)
        loader = DataLoader(mix_flip_face_dataset, batch_size=preload_batch, shuffle=False, num_workers=4)
        self.teacher.eval()
        self.teacher.cuda()
        model = DataParallel(self.teacher)
        time.sleep(0.5)
        # 初始化时将所有的教师的logit都保存下来，防止之后每一个epoch都重复计算
        # TODO 将教师的logit保存到文件中，下次直接读取
        if cfg.PRELOAD_TEACHER_LOGIT:
            t = tqdm(total=len(loader), desc="Preload teacher logit", ncols=100)
            with torch.no_grad():
                for i, (img, flip_img) in enumerate(loader):
                    # 将512个数据拼成一个tensor，然后一次性计算
                    # batch = torch.stack([dataset[i + j][1] for j in
                    #                      range(preload_batch)]).cuda() if self.teacher.cuda() else torch.tensor(
                    #     [dataset[i + j][1] for j in range(preload_batch)])
                    self.teacher_logit.extend(
                        [item.squeeze(0) for item in model(img.cuda()).cpu().chunk(preload_batch)])
                    del img
                    self.teacher_flip_logit.extend(
                        [item.squeeze(0) for item in model(flip_img.cuda()).cpu().chunk(preload_batch)])
                    del flip_img
                    t.update(1)
            t.close()

        if self.preload:
            del model
            self.teacher.cpu()
            gc.collect()
            torch.cuda.empty_cache()

    # 设置模型的训练状态
    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Distiller_Origin(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller_Origin, self).__init__()
        self.student = student
        self.teacher = teacher

    # 设置模型的训练状态
    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]
