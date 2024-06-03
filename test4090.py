"""
@author supermantx
@date 2024/4/16 17:46
测试不同模型的特征表达
4090的数据集路径不一样
"""


from tqdm import tqdm
import time
from models.mobilefacenet import get_mbf
from dataset import get_dataloader
from models.iresnet import iresnet18, iresnet100
import eval.verification as verification
from torch.nn import DataParallel
import torch
import os
import pandas as pd

# mbf = get_mbf(fp16=True, num_features=512, blocks=(1, 4, 6, 2), scale=2)
res18 = iresnet18()
res100 = iresnet100()
dataloader = get_dataloader("/home/power/tx/data/MS1MV2", 512, 4)
features_res18 = []
features_res100 = []
res18.load_state_dict(torch.load("./models/data/res18_backbone.pth"))
res100.load_state_dict(torch.load("./models/data/res100_backbone.pth"))
res18.eval()
res100.eval()
t = tqdm(range(5))
for i, (_, img, _) in enumerate(dataloader):
    img = img.cuda()
    with torch.no_grad():
        feature_res18 = res18(img)
        feature_res100 = res100(img)
    features_res18.append(feature_res18)
    features_res100.append(feature_res100)
    t.update(1)
    if i % 5 == 0:
        break

# 分析feature的L2
features_res18 = torch.cat(features_res18, dim=0)
features_res100 = torch.cat(features_res100, dim=0)
print(features_res18.shape, features_res100.shape)
# 计算L2
l2_res18 = torch.norm(features_res18, dim=1)
l2_res100 = torch.norm(features_res100, dim=1)
print(l2_res18.mean(), l2_res100.mean())

