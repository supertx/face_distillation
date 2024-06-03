"""
@author supermantx
@date 2024/4/22 14:10
计算不同的骨干网络对于数据集集模长分布
"""

import torch
from torch.nn import functional as F
from torch.nn import DataParallel
import numpy as np
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import os

from models import get_model
from distiller._base import Distiller
from dataset import get_dataloader
from utils.config_util import get_config

models = ["ResNet18", "ResNet34", "ResNet50", "ResNet100"]
PRETRAIN_PTH = {"ResNet18": "res18_backbone.pth",
                "ResNet34": "res32_backbone.pth",
                "ResNet50": "res50_backbone.pth",
                "ResNet100": "res100_backbone.pth"}
MODEL_DIR = "./models/data/"
dataloader = get_dataloader("/data/tx/MS1MV2", batch_size=128, num_workers=4)


def test():
    norms_np_ars = []
    for model_name in models:
        model = get_model(model_name)
        model = DataParallel(model.cuda(), device_ids=[0, 1])
        model.module.load_state_dict(torch.load(os.path.join(MODEL_DIR, PRETRAIN_PTH[model_name])))
        model.eval()
        t = tqdm(range(10), desc=f"model: {model_name} -> inferring...")
        norms = []
        for i, (_, img, _) in enumerate(dataloader):
            if i == 10:
                break
            out = model(img)
            norm = torch.norm(out.detach().cpu(), dim=1)
            norms.append(norm)
            t.update(1)
            del out
        t.close()
        norms = torch.cat(norms, dim=0)
        norms_np_ar = np.around(norms.view(-1).numpy(), 2)
        plt.figure(figsize=(10, 28))
        sns.displot(data=norms_np_ar, kde=True)
        plt.xlim(0, 45)
        plt.xlabel("logits norm")
        plt.title(f"logits norm count for {model_name}")
        plt.ylabel("count")
        plt.savefig(f"./test/{model_name}_norm.png")
        norms_np_ars.append(norms_np_ar.tolist())
    plt.figure(figsize=(10, 28))
    sns.displot(data=norms_np_ars, kde=True, multiple="layer")
    plt.legend(models)
    plt.savefig("./test/general_norms.png")


if __name__ == '__main__':
    # test()
    teacher_model = get_model("ResNet100")
    teacher_model.load_state_dict(torch.load("./models/data/res100_backbone.pth"))
    student_model = get_model("MobileNetV2")
    Distiller(student_model, teacher_model, get_config("./configs/KD+logit_stand,res100,mv2.yaml"))