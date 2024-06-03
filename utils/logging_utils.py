"""
@author supermantx
@date 2024/4/16 15:10
"""
import logging
import sys
import os
import time
import torch
from datetime import datetime


class AverageMeter(object):
    """
    Computes and stores the average and current value
    日志中的输出是指step个batch的loss值的平均值
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunTimeLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0, writer=None):
        self.frequent: int = frequent
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    # 调用时会使用该方法
    def __call__(self,
                 step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        if step > 0 and step % self.frequent == 0:
            if self.init:
                try:
                    # 计算样本的处理速度
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                except ZeroDivisionError:
                    speed = float('inf')
                # 计算还需要花费的时间
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - step - 1)
                time_for_end = eta_sec / 3600
                # tensorboard
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, step)
                    self.writer.add_scalar('learning_rate', learning_rate, step)
                    self.writer.add_scalar('loss', loss.avg, step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed, loss.avg, learning_rate, epoch, step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed, loss.avg, learning_rate, epoch, step, time_for_end
                          )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


def init_logger(log_dir, expri_name, to_file=True):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    std_out_formatter = StdOutFormatter()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(std_out_formatter)
    logger.addHandler(handler)
    if to_file:
        now = datetime.now()
        file_formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler = logging.FileHandler(os.path.join(log_dir,
                                                        now.strftime("%m-%d_%H:%M_") + expri_name + "_training.log"))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)


color_map = {
    "INFO": 36,
    "TRAIN": 32,
    "EVAL": 33
}


def set_map(map, record=logging.LogRecord):
    if map in color_map.keys():
        record.map = map
    else:
        raise ValueError("Invalid map value. Please use one of the following: INFO, TRAIN, EVAL")


class StdOutFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super(StdOutFormatter, self).__init__(fmt, datefmt, style)
        self.color_map = {
            "INFO": 36,
            "TRAIN": 31,
            "EVAL": 33
        }
        self.map = "INFO"

    def format(self, record):
        asctime = self.formatTime(record, self.datefmt)
        return f"\033[{self.color_map[record.map]}m[{record.map}] {asctime} - {record.msg} \033[0m"
