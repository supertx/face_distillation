"""
@author supermantx
@date 2024/4/16 15:08
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# 使用ddp
from torch.nn import DataParallel
# import torch.distributed as dist
# import torch.multiprocessing as mp
import os

from models.iresnet import iresnet100, iresnet50, iresnet18
from dataset import get_dataloader, get_dataloader_mlkd
from utils.logging_utils import init_logger, set_map, RunTimeLogging, AverageMeter
from utils.config_util import get_config, print_config
from utils.lr_util import StepScaleLR
from models.mobilefacenet import get_mbf
from distiller import get_distiller, TLoss
from eval.verification import Verification


def train(distiller, dataloader, cfg, amp, t_loss=None):
    distiller = DataParallel(distiller.cuda())
    # 定义optimizer
    optimizer = optim.SGD(distiller.module.get_learnable_parameters(),
                          lr=cfg.SOLVER.BASE_LR,
                          momentum=cfg.SOLVER.MOMENTUM,
                          weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    schedule_lr = StepScaleLR(optimizer, cfg.SOLVER.LR_DECAY_STAGES, cfg.SOLVER.LR_DECAY_RATE)
    # todo 增加随机数种子的设置，使得实验结果能够复现
    cfg.total_step = cfg.DATASET.NUM_IMAGES // cfg.DATASET.BATCH_SIZE * cfg.SOLVER.EPOCHS
    # amp
    set_map("TRAIN")
    start_epoch = 0
    step = 0
    if cfg.RESUME.IS_RESUME:
        # 学生网络从断点继续训练
        check_point = torch.load(os.path.join(cfg.RESUME.RESUME_PATH, f"checkpoint_{cfg.RESUME.RESUME_EPOCH}_epoch.pt"))
        start_epoch = cfg.RESUME.RESUME_EPOCH + 1
        distiller.load_state_dict(check_point['distiller_state_dict'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        schedule_lr.load_state_dict(check_point['scheduler_state_dict'])
        step = cfg.DATASET.NUM_IMAGES // cfg.DATASET.BATCH_SIZE * start_epoch
        if len(schedule_lr.stages) < len(cfg.SOLVER.LR_DECAY_STAGES):
            schedule_lr.last_stage = cfg.SOLVER.LR_DECAY_STAGES[len(schedule_lr.stages)]
            schedule_lr.stages = cfg.SOLVER.LR_DECAY_STAGES
        del check_point
    loss_am = AverageMeter()
    # 混合精度计算，降低运算成本
    run_time_logging = RunTimeLogging(frequent=cfg.SOLVER.PRINT_FREQ, total_step=cfg.total_step,
                                      batch_size=cfg.DATASET.BATCH_SIZE, start_step=step)
    ver_best_dict = {k: 0 for k in cfg.EVAL.EVAL_DATASET}
    ver = Verification(cfg)
    for epoch in range(start_epoch, cfg.SOLVER.EPOCHS):
        distiller.train()
        if cfg.DISTILLER.CLASS == "MLKD":
            for _, (_, img_weak, img_strong, _) in enumerate(dataloader):
                optimizer.zero_grad()
                logits_t, logits_s, loss_dict = distiller(image_weak=img_weak.cuda(), image_strong=img_strong.cuda())
                loss = sum([l.mean() for l in loss_dict.values()])
                if t_loss is not None:
                    tloss = t_loss(logits_s, logits_t).mean()
                    loss += tloss
                if cfg.SOLVER.FP16:
                    amp.scale(loss).backward()
                    amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(distiller.module.get_learnable_parameters(), 5)
                    amp.step(optimizer)
                    amp.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(distiller.module.get_learnable_parameters(), 5)
                    optimizer.step()
                loss_am.update(loss.item(), 1)
                del logits_t, logits_s, loss_dict, loss
                step += 1
                run_time_logging(step, loss_am, epoch,
                                 cfg.SOLVER.FP16, schedule_lr.get_last_lr()[0], amp)
        else:
            for _, (index, img, flip_flag, _) in enumerate(dataloader):
                optimizer.zero_grad()
                img = img.cuda()
                # label = label.cuda()
                logits_t, logits_s, loss_dict = distiller(index=index, image=img, flip_flag=flip_flag)

                loss = loss_dict.sum()
                if t_loss is not None:
                    tloss = t_loss(logits_s, logits_t).mean()
                    loss += tloss
                if cfg.SOLVER.FP16:
                    amp.scale(loss).backward()
                    amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(distiller.module.get_learnable_parameters(), 5)
                    amp.step(optimizer)
                    amp.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(distiller.module.get_learnable_parameters(), 5)
                    optimizer.step()
                loss_am.update(loss.item(), 1)
                del logits_t, logits_s, loss_dict, loss
                step += 1
                run_time_logging(step, loss_am, epoch,
                                 cfg.SOLVER.FP16, schedule_lr.get_last_lr()[0], amp)

        # 每隔一定的epoch在验证集上查看模型的性能
        if epoch % cfg.LOG.FREQUENCY == 0:
            set_map("EVAL")
            distiller.eval()
            ver_dict = ver.verification(distiller.module.student)
            for key, (acc, std, xnorm) in ver_dict.items():
                if acc > ver_best_dict[key]:
                    ver_best_dict[key] = acc
                logging.info(f"{key}: acc {acc * 100:.2f} best acc {ver_best_dict[key] * 100:.2f}% std: {std} xnorm: {xnorm}")
            set_map("TRAIN")
        # if epoch % cfg.LOG.IJB_FREQUENCY == 0:
        #     # TODO ijb ROC-
        #     set_map("EVAL")
        #     # 在IJB上测试模型的性能
        #     ver_ijb = ver.ver_ijb(distiller.module.student)
        #     if ver_ijb > ver_best_dict["ijb"]:
        #         ver_best_dict["ijb"] = ver_ijb
        #     set_map("TRAIN")

        schedule_lr.step()
        if epoch % cfg.SOLVER.SAVE_STEP == 0:
            # 在测试集上计算准确度并且保存模型参数
            check_point = {
                "epoch": epoch,
                "distiller_state_dict": distiller.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": schedule_lr.state_dict()
            }
            if not os.path.exists(cfg.RESUME.RESUME_PATH):
                os.makedirs(cfg.RESUME.RESUME_PATH)
            torch.save(check_point,
                       os.path.join(cfg.RESUME.RESUME_PATH, f"checkpoint_{epoch}_epoch.pt"))
            torch.save(distiller.module.student.state_dict(),
                       os.path.join(cfg.RESUME.RESUME_PATH, f"student_{epoch}_epoch.pt"))


# def main(rank, args, cfg):
def main(args, cfg):
    # 加入随机种子
    # if rank == 0:
    #     init_logger(cfg.EXPERIMENT.LOG_DIR, cfg.EXPERIMENT.NAME)
    #     set_map("INFO")
    #     logging.info(f'teacher network: {cfg.DISTILLER.TEACHER}')
    #     logging.info(f'student network: {cfg.DISTILLER.STUDENT}')
    init_logger(cfg.EXPERIMENT.LOG_DIR, cfg.EXPERIMENT.NAME)
    set_map("INFO")
    logging.info(f'teacher network: {cfg.DISTILLER.TEACHER}')
    logging.info(f'student network: {cfg.DISTILLER.STUDENT}')
    # dist.init_process_group("nccl", rank=rank, world_size=cfg.DDP.WORLD_SIZE)
    teacher_model = iresnet50(fp16=cfg.SOLVER.FP16)
    student_model = get_mbf(True, 512, blocks=(1, 4, 6, 2), scale=2)
    print_config(cfg)
    teacher_model.load_state_dict(torch.load(cfg.SOLVER.TEACHER_PTH))
    distiller = get_distiller(cfg, student_model, teacher_model)
    # dataloader = get_dataloader(rank, cfg.DATASET.DATA_DIR, cfg.DATASET.BATCH_SIZE)
    if cfg.DISTILLER.CLASS == "MLKD":
        dataloader = get_dataloader_mlkd(cfg.DATASET.DATA_DIR, cfg.DATASET.BATCH_SIZE)
    else:
        dataloader = get_dataloader(cfg.DATASET.DATA_DIR, cfg.DATASET.BATCH_SIZE)
    amp = torch.cuda.amp.GradScaler(growth_interval=100)
    # train(rank, distiller, dataloader, cfg, amp)
    if cfg.DISTILLER.USE_TLOSS:
        t_loss = TLoss(cfg)
        train(distiller, dataloader, cfg, amp, t_loss=t_loss)
    else:
        train(distiller, dataloader, cfg, amp)
    logging.info("train finished")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='distillation')
    parser.add_argument('--config',
                        default="./configs/KD+logit_stand,res50,mv2.yaml", type=str, help='config file')

    args = parser.parse_args()
    cfg = get_config(args.config)
    # mp.spawn(main, args=(args, cfg), nprocs=cfg.DDP.WORLD_SIZE)
    main(args, cfg)