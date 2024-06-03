"""
@author supermantx
@date 2024/4/16 17:46
"""

from tqdm import tqdm
from models import get_model
import eval.verification as verification
from torch.nn import DataParallel
import torch
import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import re
from utils.config_util import get_config


def get_available_devices():
    return [torch.cuda.device(i) for i in range(torch.cuda.device_count())]


def eval(cfg):
    # 加载模型
    model_names = os.listdir(cfg.EVAL.MODEL_PATH)
    if cfg.EVAL.EVAL_EPOCH:
        model_names = [model_name for model_name in model_names if
                       model_name.endswith("epoch.pt") and model_name.startswith("student")]
    else:
        pattern = r'^student_\d+\*\d+\.\d+\.pt$'
        model_names = [model_name for model_name in model_names if re.match(pattern, model_name)]
    mbf = get_model("MobileNetV2")
    mbf = DataParallel(mbf.cuda(), get_available_devices())
    mbf.eval()
    # 加载测试集
    bin_set = []
    datasets = list(cfg.EVAL.EVAL_DATASET)
    for bin_name in datasets:
        print("loading bin: ", bin_name)
        bin = verification.load_bin(os.path.join(cfg.EVAL.DATASET_PATH, bin_name + ".bin"), (112, 112))

        bin_set.append(bin)
        # 答应数据集的信息
    evaluates = []
    for bin_name, bin in zip(datasets, bin_set):
        print("evaluating: ", bin_name)
        t = tqdm(range(len(model_names)))
        evaluate = []
        for i, model_name in enumerate(model_names):
            mbf.module.load_state_dict(torch.load(os.path.join(cfg.EVAL.MODEL_PATH, model_name)))
            test = verification.test(bin, mbf, batch_size=512, nfolds=10)
            if cfg.EVAL.EVAL_EPOCH:
                e = [i + 1]
            else:
                e = [i * cfg.LOG.FREQUNCY * i]
            e.extend(test[2: 5])
            evaluate.append(e)
            t.update(1)
        t.close()
        evaluates.append(evaluate)
    # 保存数据
    for bin_name, evaluate in zip(datasets, evaluates):
        frame = pd.DataFrame(evaluate)
        frame.columns = ["train_time", "accuracy", "std", "norm"]
        frame.to_csv(os.path.join(str(cfg.EXPERIMENT.LOG_DIR),
                                  cfg.EXPERIMENT.NAME +"_"+ bin_name + ".csv"))
    # 绘制图形
    frame = []
    for i in range(len(evaluates[0])):
        f = [evaluates[0][i][0]]
        f.extend([evaluates[j][i][1] for j in range(len(evaluates))])
        frame.append(f)
    columns = ["epoch"]
    columns.extend(datasets)
    frame = pd.DataFrame(frame, columns=columns)
    frame.index = frame["epoch"]
    frame = frame.iloc[:, 1:]
    sns.lineplot(data=frame, dashes=False)
    plt.ylabel("accuracy")
    plt.savefig(os.path.join(str(cfg.EXPERIMENT.LOG_DIR), cfg.EXPERIMENT.NAME + ".png"))
    plt.show()


eval(get_config("./configs/KD+logit_stand,res100,mv2.yaml"))
