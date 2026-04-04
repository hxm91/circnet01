import torch
# torch.cuda.init()  # Initialize CUDA, no need 

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
# from PIL import Image
import __init__ as booger
import cv2
import os
import gc
import numpy as np
from matplotlib import pyplot as plt

import wandb  # 新增：导入 wandb

from common.avgmeter import *
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.Lovasz_Softmax import Lovasz_softmax

class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None, gpu_ids=None, transform=None):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path
        self.transform = transform

        # gpu devices
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_ids = gpu_ids

        # 初始化wandb
        wandb.init(project="semantic_segmentation", config=self.ARCH, name=f"Experiment_{time.strftime('%Y%m%d-%H%M%S')}")
        # 
        self.info_epoch = {"train_update": 0,
                           "train_loss": 0,
                           "train_loss_xent": 0,
                           "train_loss_ls": 0,
                           "train_acc": 0,
                           "train_iou": 0,
                           "train_iou_cls": [0] * 20,
                           "valid_loss": 0,
                           "valid_loss_xent": 0,
                           "valid_loss_ls": 0,
                           "valid_acc": 0,
                           "valid_iou": 0,
                           "valid_iou_cls": [0] * 20,
                           "backbone_lr": 0,
                           "decoder_lr": 0,
                           "head_lr": 0,
                           "post_lr": 0}
        self.info_iter = {"train_update": 0,
                          "train_loss": 0,
                          "train_acc": 0,
                          "train_iou": 0,
                          "valid_loss": 0,
                          "valid_acc": 0,
                          "valid_iou": 0,
                          "backbone_lr": 0,
                          "decoder_lr": 0,
                          "head_lr": 0,
                          "post_lr": 0}

        # load data
        parserModule = imp.load_source("parserModule", os.path.join(booger.TRAIN_PATH,'tasks/semantic/dataset', self.DATA["name"], 'parser.py'))
        self.parser = parserModule.Parser(
            root=self.datadir,
            train_sequences=self.DATA["split"]["train"],
            valid_sequences=self.DATA["split"]["valid"],
            test_sequences=None,
            labels=self.DATA["labels"],
            color_map=self.DATA["color_map"],
            learning_map=self.DATA["learning_map"],
            learning_map_inv=self.DATA["learning_map_inv"],
            sensor=self.ARCH["dataset"]["sensor"],
            max_points=self.ARCH["dataset"]["max_points"],
            batch_size=self.ARCH["train"]["batch_size"],
            workers=self.ARCH["train"]["workers"],
            gt=True,
            shuffle_train=True,
            transform=self.transform
        )

        # 计算 loss 的类别权重
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)
        for x_cl, w in enumerate(self.loss_w):
            if DATA["learning_ignore"][x_cl]:
                self.loss_w[x_cl] = 0
        
        # 初始化模型
        with torch.no_grad():
            self.model = Segmentator(self.ARCH, self.parser.get_n_classes(), self.path)
        if len(self.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model = convert_model(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)
        
        # loss function
        if "loss" in self.ARCH["train"] and self.ARCH["train"]["loss"] == "xentropy":
            self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        else:
            raise Exception('Loss not defined in config file')

        if 'Lovasz_softmax' in self.ARCH["train"] and self.ARCH["train"]['Lovasz_softmax']:
            self.LS_criterion = Lovasz_softmax(ignore=0).to(self.device)
        else:
            self.LS_criterion = None

        if len(self.gpu_ids) > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()
            if self.ARCH["train"].get('Lovasz_softmax', False):
                self.LS_criterion = nn.DataParallel(self.LS_criterion).cuda()
        
        # optimizer
        if self.ARCH["post"]["CRF"]["use"] and self.ARCH["post"]["CRF"]["train"]:
            self.lr_group_names = ["post_lr"]
            self.train_dicts = [{'params': self.model.CRF.parameters()}]
        else:
            self.lr_group_names = []
            self.train_dicts = []
        if self.ARCH["backbone"]["train"]:
            self.lr_group_names.append("backbone_lr")
            self.train_dicts.append({'params': self.model.backbone.parameters()})
        if self.ARCH["decoder"]["train"]:
            self.lr_group_names.append("decoder_lr")
            self.train_dicts.append({'params': self.model.decoder.parameters()})
        if self.ARCH["head"]["train"]:
            self.lr_group_names.append("head_lr")
            self.train_dicts.append({'params': self.model.head.parameters()})

        self.optimizer = torch.optim.Adam(self.train_dicts,
                                          lr=self.ARCH["train"]["lr"],
                                          betas=(0.9, 0.999))
        self.optimizer.zero_grad(set_to_none=True)

        steps_per_epoch = self.parser.get_train_size()
        # up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)  # orig
        # final_decay = self.ARCH["train"]["lr_decay"] ** (1/steps_per_epoch) # orig
        # self.scheduler = warmupLR(optimizer=self.optimizer, lr=self.ARCH["train"]["lr"], warmup_steps=up_steps, momentum=self.ARCH["train"]["momentum"], decay=lr_decay)
        up_steps = self.ARCH["train"]["wup_epochs"]
        lr_decay = self.ARCH["train"]["lr_decay"]
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=up_steps, gamma=lr_decay)

        # 使用 wandb 初始化实验记录
        wandb.watch(self.model, log="gradients")

    # 训练
    def train(self):
        best_train_iou = 0.0
        best_val_iou = 0.0
        scaler = torch.amp.GradScaler('cuda')  # 使用混合精度训练
        # 定义训练过程中的权重计算器
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print(f"Ignoring class {i} in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(), self.device, self.ignore_class)

        self.optimizer.zero_grad()
        # start training
        wandb.config.epochs = self.ARCH["train"]["max_epochs"]
        for epoch in range(wandb.config.epochs):
            
            # 更新当前学习率信息
            for name, g in zip(self.lr_group_names, self.optimizer.param_groups):
                self.info_epoch[name] = g['lr']
                self.info_iter[name] = g['lr']
            # 每epoch训练 train_epoch
            # self.model.train()
            train_epoch_iou_cls, train_epoch_iou, train_epoch_acc , train_epoch_loss  = self.train_epoch(
                train_loader=self.parser.get_train_set(), 
                model=self.model, 
                criterion=self.criterion, 
                optimizer=self.optimizer, 
                epoch=epoch, 
                evaluator=self.evaluator, 
                scheduler=self.scheduler, 
                color_fn=self.parser.to_color, 
                report=self.ARCH["train"]["report_batch"], 
                show_scans=self.ARCH["train"]["show_scans"], 
                LS_criterion=self.LS_criterion)
            
            print("*" * 80)
            # 保存训练信息
            # 保存train_iou最好的模型
            if train_epoch_iou > best_train_iou:
                print(f"train_iou improved from {best_train_iou} to {train_epoch_iou}")
                best_train_iou = train_epoch_iou
                torch.save(self.model.state_dict(), os.path.join(self.log, "best_train.pth"))
            # 每epoch验证
            valid_epoch_iou_cls, valid_epoch_iou, valid_epoch_acc, valid_epoch_loss = self.validate_epoch(
                valid_loader=self.parser.get_valid_set(), 
                model=self.model, 
                criterion=self.criterion, 
                epoch=epoch,
                evaluator=self.evaluator, 
                class_func=self.parser.get_xentropy_class_string,
                color_fn=self.parser.to_color,
                save_scans=self.ARCH["train"]["save_scans"], 
                LS_criterion=self.LS_criterion)

            # 保存val_iou最好的模型
            if valid_epoch_iou > best_val_iou:
                print(f"val_iou improved from {best_val_iou} to {valid_epoch_iou}")
                best_val_iou = valid_epoch_iou
                torch.save(self.model.state_dict(), os.path.join(self.log, "best_val.pth"))
            print("*" * 80)
            # 每epoch结束后更新学习率
            # 如果调度器设计为 epoch-level，则在每个 epoch 结束时更新 lr
            self.scheduler.step()
        wandb.finish()
    
    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, color_fn,
                  report=10, show_scans=False, LS_criterion=None):
        
        model.train()
        scaler = torch.amp.GradScaler('cuda')  # 混合精度训练
        accumulation_steps = 2

        evaluator.reset()
        running_loss = AverageMeter()
        running_loss_xent = AverageMeter()
        running_loss_ls = AverageMeter()
        epoch_iou_cls = []
        for ii in range(20):
            epoch_iou_cls.append(AverageMeter())
        epoch_iou = AverageMeter()
        epoch_acc = AverageMeter()
        update_ratio_meter = AverageMeter()
        epoch_size = len(train_loader)
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, *_) in enumerate(train_loader):
            if i+1>5:
                break
            if self.gpu_ids is not None:
                in_vol = in_vol.to(self.device)
                proj_mask = proj_mask.to(self.device)
                proj_labels = proj_labels.to(self.device).long()
            inputs = (in_vol, proj_mask)
            targets = proj_labels

            with torch.amp.autocast('cuda'):
                outputs = model(in_vol, proj_mask)
                loss_xentropy = criterion(torch.log(outputs.clamp(min=1e-8)), targets)
                if LS_criterion is not None:
                    loss_xent = loss_xentropy
                    loss_ls = LS_criterion(outputs, targets)
                    loss = loss_xent + loss_ls
                else:
                    loss = loss_xentropy
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            
            if i % accumulation_steps == 0 or (i+1)==len(train_loader):
                scaler.step(optimizer) # 更新权重
                scaler.update() # 更新scaler
                # optimizer.step()
                optimizer.zero_grad()   # 梯度清零      


            # running_loss += loss.item()
            loss = loss.mean()
            running_loss.update(loss.item(), in_vol.size(0))
            running_loss_xent.update(loss_xent.item(), in_vol.size(0))
            running_loss_ls.update(loss_ls.item(), in_vol.size(0))
            
            with torch.no_grad():
                evaluator.reset()
                argmax = outputs.argmax(dim=1)
                evaluator.addBatch(argmax, targets)
                accuracy = evaluator.getacc()
                miou, iou_per_cls = evaluator.getIoU()
                print(f'Epoch: [{epoch}/{wandb.config.epochs}][{i}/{len(train_loader)}],miou:{miou.item():.4f}')
                # print('iou_per_cls:', iou_per_cls)

            # epoch_iou_cls += iou_per_cls # 计算每个类别的iou
            epoch_iou.update(miou.item(), in_vol.size(0))
            epoch_acc.update(accuracy.item(), in_vol.size(0))
            for cls in range(20):
                epoch_iou_cls[cls].update(iou_per_cls[cls].item(), in_vol.size(0))
            
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) *
                                                value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))

            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            epoch_loss = running_loss
            self.info_iter["train_update"] = update_mean
            self.info_iter["train_loss"] = float(loss)
            self.info_iter["train_acc"] = epoch_acc.val
            self.info_iter["train_iou"] = epoch_iou.val

            del inputs, targets, outputs
            torch.cuda.empty_cache()
            gc.collect()
        # epoch_iou_cls /= epoch_size
        # epoch_iou /= epoch_size
        # epoch_acc /= epoch_size
        # epoch_loss = running_loss / epoch_size
        
        wandb.log({"train_loss": epoch_loss.avg,
                   "train_iou": epoch_iou.avg,
                   "train_acc": epoch_acc.avg,
                   "learning_rate": scheduler.get_last_lr()[0]})
        
        print(f"Epoch [{epoch}/{wandb.config.epochs}], Loss: {epoch_loss.avg:.4f}, LR: {scheduler.get_last_lr()[0]}, mIOU: {epoch_iou.avg:.4f}, mAcc: {epoch_acc.avg:.4f}")
        epoch_iou_cls_avg = []
        for cls in range(20):
            epoch_iou_cls_avg.append(epoch_iou_cls[cls].avg)
        print(f'train_iou per class: {epoch_iou_cls_avg}') # 打印每个类别的iou
        return epoch_iou_cls_avg, epoch_iou.avg, epoch_acc.avg , epoch_loss.avg       
    
    def validate_epoch(self, valid_loader, model, criterion, evaluator, epoch,class_func,color_fn, save_scans=False, LS_criterion=None):
        model.eval()
        evaluator.reset()
        running_loss = AverageMeter()
        running_loss_xent = AverageMeter()
        running_loss_ls = AverageMeter()
        epoch_iou_cls = []
        for ii in range(20):
            epoch_iou_cls.append(AverageMeter())
        epoch_iou = AverageMeter()
        epoch_acc = AverageMeter()
        update_ratio_meter = AverageMeter()
        epoch_size = len(valid_loader)
        with torch.no_grad():
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, *_) in enumerate(valid_loader):
                if i+1>5:
                    break
                if self.gpu_ids is not None:
                    in_vol = in_vol.to(self.device)
                    proj_mask = proj_mask.to(self.device)
                    proj_labels = proj_labels.to(self.device).long()
                inputs = (in_vol, proj_mask)
                targets = proj_labels

                outputs = model(in_vol, proj_mask)
                loss_xentropy = criterion(torch.log(outputs.clamp(min=1e-8)), targets)
                if LS_criterion is not None:
                    loss_xent = loss_xentropy
                    loss_ls = LS_criterion(outputs, proj_labels.long())
                    loss = loss_xent + loss_ls
                else:
                    loss = loss_xentropy

                # loss = loss.mean()
                running_loss.update(loss.mean().item(), in_vol.size(0))
                running_loss_xent.update(loss_xent.mean().item(), in_vol.size(0))
                running_loss_ls.update(loss_ls.mean().item(), in_vol.size(0))

                argmax = outputs.argmax(dim=1)
                evaluator.addBatch(argmax, targets)
                accuracy = evaluator.getacc()
                miou, iou_per_cls = evaluator.getIoU()
                
                epoch_iou.update(miou.item(), in_vol.size(0))
                epoch_acc.update(accuracy.item(), in_vol.size(0))
                for cls in range(20):
                    epoch_iou_cls[cls].update(iou_per_cls[cls].item(), in_vol.size(0))
                
                print(f'Epoch: [{epoch}/{wandb.config.epochs}][{i}/{epoch_size}],miou:{miou.item()}')

                epoch_loss = running_loss.item()
                self.info_iter["valid_loss"] = float(loss)
                self.info_iter["valid_acc"] = epoch_acc.val
                self.info_iter["valid_iou"] = epoch_iou.val

                del inputs, targets, outputs
                torch.cuda.empty_cache()
                gc.collect()
                
                
        wandb.log({"valid_loss": epoch_loss.avg,
                   "valid_iou": epoch_iou.avg,
                   "valid_acc": epoch_acc.avg})
        print(f'valid_loss: {epoch_loss.avg:.4f}, valid_iou: {epoch_iou.avg:.4f}, valid_acc: {epoch_acc.avg:.4f}') # 
        epoch_iou_cls_avg = []
        for cls in range(20):
            epoch_iou_cls_avg.append(epoch_iou_cls[cls].avg)
        print(f'valid_iou per class: {epoch_iou_cls_avg}')
        return epoch_iou_cls_avg, epoch_iou.avg, epoch_acc.avg, epoch_loss.avg