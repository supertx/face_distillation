"""
@author supermantx
@date 2024/4/16 14:27
"""
import numbers

import mxnet as mx
import torch
import numpy as np
from PIL import ImageOps, ImageEnhance, Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
import os
import random


class MixFlipFaceDataset(Dataset):
    def __init__(self, root_dir):
        super(MixFlipFaceDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(root_dir, 'train.idx'),
                                                    os.path.join(root_dir, 'train.rec'),
                                                    'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        _, img = mx.recordio.unpack(s)
        image = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            image = self.transform(image)
        # 水平旋转
        flip_image = F.hflip(image)
        return image, flip_image

    def __len__(self):
        return 10000
        # return len(self.imgidx)


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(MXFaceDataset, self).__init__()
        if not transform:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self.flag = False
        else:
            self.transform = transform
            self.flag = True
        self.imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(root_dir, 'train.idx'),
                                                    os.path.join(root_dir, 'train.rec'),
                                                    'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        # 水平旋转
        if not self.flag:
            flip_flag = False
            if torch.rand(1) < 0.5:
                flip_flag = True
                sample = F.hflip(sample)
            return index, sample, flip_flag, label
        else:
            return index, sample[0], sample[1], label

    def __len__(self):
        return 10000
        # return len(self.imgidx)


def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return ImageOps.equalize(img)


def Invert(img, _):
    return ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    # assert -30 <= v <= 30
    # if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert v >= 0.0
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert 0 <= v
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        return img


class MultipleApply:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return [t(image) for t in self.transforms]


def get_dataloader_mlkd(root_dir, batch_size, num_workers=4, shuffle=True):
    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')

    train_transform_weak = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    train_transform_strong = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            RandAugment(2, 10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    aug_transform = MultipleApply([train_transform_weak, train_transform_strong])
    if os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, transform=aug_transform)
    else:
        train_set = ImageFolder(root_dir, aug_transform)
    return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_dataloader(root_dir, batch_size, num_workers=4, shuffle=True):
    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')

    if os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir)
    # Image Folder
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_set = ImageFolder(root_dir, transform)
    return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_distribute_dataloader(root_dir, rank, batch_size, num_workers=2):
    """
    使用ddp方案时，需要使用分布式的dataloader
    每个dataloader获取同一个batch的不同的部分
    """
    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')

    if os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir)
    # Image Folder
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_set = ImageFolder(root_dir, transform)
    #  arcface手动定义了DistributedSampler和torch写的没有什么区别，就是torch的可以drop_last
    sampler = DistributedSampler(train_set, shuffle=True, rank=rank, drop_last=True)
    # 每个epoch都需要重置采样器的种子，否则每个epoch采样器返回的indice采样都是一样的顺序
    # worker_init_fn产生随机数种子，dataloader会调用dataset进行数据增强，在这区间可能会需要随机数种子
    return sampler, DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
