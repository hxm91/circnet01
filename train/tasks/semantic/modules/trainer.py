import torch
# torch.cuda.init()  # Initialize CUDA, no need 
import imp
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
# from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import gc
import numpy as np
from matplotlib import pyplot as plt


import wandb  # 新增：导入 wandb
import sys
sys.path.append('..')
from common.logger import Logger
from common.avgmeter import *
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.Lovasz_Softmax import Lovasz_softmax
# from common.data_parallel import BalancedDataParallel
from tasks.semantic.postproc.KNN import KNN

def check_tensor(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN values.")
    if torch.isinf(tensor).any():
        print(f"{name} contains Inf values.")

class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None, gpu_ids=None, transform=None, range_transform=None): #
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path
        self.transform = transform
        self.range_transform = range_transform # 

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        # gpu devices
        # pdb.set_trace()
        if len(gpu_ids) > 1:
            device_ids = ""
            for i in range(len(gpu_ids)-1):
                device_ids += str(gpu_ids[i]) + ","
            device_ids += str(gpu_ids[-1])
        else:
            device_ids = str(gpu_ids[0])
        # torch.cuda.set_device(device_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
        gpu_ids = list(range(len(gpu_ids)))

        # 初始化wandb
        wandb.init(project="semantic_segmentation_debug", config=self.ARCH, name=f"Experiment_{time.strftime('%Y%m%d-%H%M%S')}")
        # print(f'save config with wandb: {wandb.config}')
        # 
        self.tb_logger_epoch = Logger(self.log + "/tb_epoch")
        self.tb_logger_iter = Logger(self.log + "/tb_iter")
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

        # 加载数据
        parserModule = imp.load_source("parserModule", os.path.join(booger.TRAIN_PATH,'tasks/semantic/dataset', self.DATA["name"], 'parser.py'))
        self.parser = parserModule.Parser(root=self.datadir,
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
                                          transform=self.transform,
                                          range_transform=self.range_transform) # 修改

        # 计算 loss 的类别权重
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)
        # self.loss_w = 1.0 / (np.log(content + 1.02))
        for x_cl, w in enumerate(self.loss_w):
            if DATA["learning_ignore"][x_cl]:
                self.loss_w[x_cl] = 0
        print(f"Loss weights from content: {self.loss_w.data} ")

        
        # 初始化模型
        with torch.no_grad():
            self.model = Segmentator(self.ARCH, self.parser.get_n_classes(), self.path)

        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: GPU ", gpu_ids)

        # 打印model的参数
        '''for name, param in self.model.named_parameters():
            # print(f'{name}: {param.data.shape}')
            print(f"{name:60} requires_grad={param.requires_grad}")'''
        for name, param in self.model.named_parameters():
            if not param.requires_grad and param.dtype.is_floating_point:
                print(f"{name:60} requires_grad={param.requires_grad}")
                param.requires_grad = True
                print(f'{name:60}  now_requires_grad={param.requires_grad}')
        print(f'--------------------------------')
        
    
        if torch.cuda.is_available() and len(gpu_ids) > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if len(gpu_ids) > 1:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = len(gpu_ids)
            print("Let's use", self.n_gpus, "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.multi_gpu = True
            
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"], self.parser.get_n_classes())

        # 损失函数 loss function
        if "loss" in self.ARCH["train"].keys() and self.ARCH["train"]["loss"] == "xentropy":
            # self.criterion = nn.NLLLoss(weight=self.loss_w,ignore_index=0).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.loss_w,ignore_index=0).to(self.device)
        else:
            raise Exception('Loss not defined in config file')

        if 'Lovasz_softmax' in self.ARCH["train"] and self.ARCH["train"]['Lovasz_softmax']:
            self.LS_criterion = Lovasz_softmax(ignore=0).to(self.device)
        else:
            self.LS_criterion = None

        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()  
            if 'LS_criterion' in self.ARCH["train"] and self.ARCH["train"]['Lovasz_softmax']:
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
                                          betas=(0.9, 0.999), weight_decay=self.ARCH["train"]["w_decay"])
        self.optimizer.zero_grad(set_to_none=True)

        steps_per_epoch = self.parser.get_train_size()
        up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)  # orig
        final_decay = self.ARCH["train"]["lr_decay"] ** (1/steps_per_epoch) # orig
        self.scheduler = warmupLR(optimizer=self.optimizer, lr=self.ARCH["train"]["lr"], warmup_steps=up_steps, momentum=self.ARCH["train"]["momentum"], decay=final_decay)
        '''up_steps = self.ARCH["train"]["wup_epochs"]
        lr_decay = self.ARCH["train"]["lr_decay"]
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=up_steps, gamma=lr_decay)'''
        # 

        # 使用 wandb 初始化实验记录
        wandb.watch(self.model, log="gradients")

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)
    
    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                            norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
    # save scalars
        for tag, value in info.items():
            if 'iou_cls' in tag:
                logger.multi_scalar_summary(tag, value, epoch)
            else:
                logger.scalar_summary(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)
    
    @staticmethod
    def pred_2_pc(pred_on_rangeimage, pixel_x,pixel_y,n_of_points,target_len=150000):
        pred = pred_on_rangeimage
        px=pixel_x
        py=pixel_y
        npoints = n_of_points
        '''print(f'size of pred: {pred.shape}')
        print(f'size of pixel_x: {px.shape}')
        print(f'size of pixel_y: {py.shape}')
        print(f'size of npoints: {npoints.shape}')
        print(f'value of target_len: {target_len}')'''
        argmax= pred.argmax(dim=1)
        unprj_argmax=[]
        for l in np.arange(argmax.shape[0]):
            # print(f'l = {l}')
            argmax_l = argmax[l,:,:]
            n_points_l = npoints[l]
            p_x_l = px[l][:n_points_l]
            p_y_l = py[l][:n_points_l]
            '''print(f'size of argmax_l: {argmax_l.shape}')
            print(f'size of n_points_l: {n_points_l.shape}')
            print(f'size of p_x_l: {p_x_l.shape}')
            print(f'size of p_y_l: {p_y_l.shape}')'''

            unproj_argmax_l = argmax_l[p_y_l, p_x_l]
            # print(f' size of unproj_argmax_l: {unproj_argmax_l.shape}')
            # torch.full([self.max_points], -1.0, dtype=torch.int32)
            # unproj_argmax_l_full = torch.zeros(target_len,-1.0, dtype = unproj_argmax_l.dtype, device=unproj_argmax_l.device)
            # unproj_argmax_l_full[:unproj_argmax_l_len] = unproj_argmax_l
            assert unproj_argmax_l.shape[0] < target_len, print(f'len of unproj_argmax_l_{l}: {unproj_argmax_l.shape}')
            pad_size = target_len - unproj_argmax_l.shape[0]
            pad_tensor = torch.full((pad_size,), -1, dtype=unproj_argmax_l.dtype, device=unproj_argmax_l.device)
            unproj_argmax_l_full = torch.cat([unproj_argmax_l, pad_tensor])
            
            unprj_argmax.append(unproj_argmax_l_full)
        unprj_argmax = torch.cat(unprj_argmax, dim = 0)

        return unprj_argmax
    def froz_param(epoch_n,epoch,model,name_str='encoder'):
        '''
        froze the parameters from n_th epoch, wenn name_str in parameter name.
        '''
        n=epoch_n
        if epoch == n:
            for name, param in model.named_parameters():
                if name_str in name:
                    param.requires_grad = False
                    print(f'now parameter in {name_str} are forzen for fine-tuning!')


    

        # 训练
    
    def train(self):
        best_train_iou = 0.0
        best_val_iou = 0.0
        
        # 定义训练过程中的权重计算器
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print(f"Ignoring class {i} in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(), self.device, self.ignore_class)

        torch.autograd.set_detect_anomaly(True)
        # start training
        epochs = self.ARCH["train"]["max_epochs"]
        wandb.config.epochs = self.ARCH["train"]["max_epochs"]
        for epoch in range(epochs):

            '''if epoch+1>10:
                break'''
            
            # 更新当前学习率信息
            groups = self.optimizer.param_groups
            for name, g in zip(self.lr_group_names, groups):
                self.info_epoch[name] = g['lr']
                self.info_iter[name] = g['lr']

            # 每epoch训练 train_epoch
            # return epoch_acc.avg, epoch_iou.avg, epoch_iou_cls_avg, epoch_loss.avg, running_loss_xent.avg, running_loss_ls.avg, update_ratio_meter.avg
            train_epoch_acc, train_epoch_iou, train_epoch_iou_cls , train_epoch_loss, train_epoch_loss_xent, train_epoch_loss_ls,update_mean,lr  = self.train_epoch(train_loader=self.parser.get_train_set(), 
                                                                                                                                                                 model=self.model, 
                                                                                                                                                                 criterion=self.criterion, 
                                                                                                                                                                 LS_criterion=self.LS_criterion,
                                                                                                                                                                 optimizer=self.optimizer, 
                                                                                                                                                                 epoch=epoch, 
                                                                                                                                                                 evaluator=self.evaluator, 
                                                                                                                                                                 scheduler=self.scheduler, 
                                                                                                                                                                 color_fn=self.parser.to_color, 
                                                                                                                                                                 report=self.ARCH["train"]["report_batch"], 
                                                                                                                                                                 show_scans=self.ARCH["train"]["show_scans"],
                                                                                                                                                                 post=self.post)
            
            # 更新 info_epoch
            self.info_epoch['train_update']= update_mean
            self.info_epoch['train_loss'] = train_epoch_loss
            self.info_epoch['train_loss_xent'] = train_epoch_loss_xent
            self.info_epoch['train_loss_ls'] = train_epoch_loss_ls
            self.info_epoch['train_acc'] = train_epoch_acc
            self.info_epoch['train_iou'] = train_epoch_iou
            self.info_epoch['train_iou_cls'] = train_epoch_iou_cls

            # 保存训练信息
            # 保存train_iou最好的模型
            if train_epoch_iou > best_train_iou:
                print(f"train_iou improved from {best_train_iou} to {train_epoch_iou}")
                best_train_iou = train_epoch_iou
                self.model_single.save_checkpoint(self.log, suffix="_best_train")
            print(f"Training of epoch {epoch} finish!")

            # 设置验证步骤
            if (epoch+1) % self.ARCH["train"]["report_epoch"] == 0:
                # return epoch_acc.avg, epoch_iou.avg, epoch_iou_cls_avg, running_loss.avg, running_loss_xent.avg, running_loss_ls.avg
                print(time.strftime("%Y-%m-%d %H:%M:%S"), f"Start validation of epoch {epoch}!" )
                valid_epoch_acc, valid_epoch_iou, valid_epoch_iou_cls, valid_epoch_loss, valid_epoch_loss_xent, valid_epoch_loss_ls, valid_epoch_rand_img = self.validate_epoch(valid_loader=self.parser.get_valid_set(), 
                                                                                                                                                                           model=self.model, 
                                                                                                                                                                           criterion=self.criterion, 
                                                                                                                                                                           evaluator=self.evaluator,
                                                                                                                                                                           epoch=epoch,
                                                                                                                                                                           class_func=self.parser.get_xentropy_class_string,
                                                                                                                                                                           color_fn=self.parser.to_color,
                                                                                                                                                                           save_scans=self.ARCH["train"]["save_scans"], 
                                                                                                                                                                           LS_criterion=self.LS_criterion,
                                                                                                                                                                           post=self.post)
                # 更新信息info_epoch
                self.info_epoch['valid_loss'] = valid_epoch_loss
                self.info_epoch['valid_loss_xent'] = valid_epoch_loss_xent
                self.info_epoch['valid_loss_ls'] = valid_epoch_loss_ls
                self.info_epoch['valid_acc'] = valid_epoch_acc
                self.info_epoch['valid_iou'] = valid_epoch_iou
                self.info_epoch['valid_iou_cls'] = valid_epoch_iou_cls
                # 保存epoch训练结果
                wandb.log({
                    "learning_rate": lr, 
                    "train_loss": train_epoch_loss,
                    "train_loss_xent": train_epoch_loss_xent,
                    "train_loss_ls": train_epoch_loss_ls,
                    "train_iou": train_epoch_iou,
                    "train_acc": train_epoch_acc,
                    "train_iou_car":train_epoch_iou_cls[1],
                    "train_iou_bicycle":train_epoch_iou_cls[2],
                    "train_iou_motorcycle":train_epoch_iou_cls[3],
                    "train_iou_truck":train_epoch_iou_cls[4],
                    "train_iou_other-vehicle":train_epoch_iou_cls[5],
                    "train_iou_persom":train_epoch_iou_cls[6],
                    "train_iou_bicyclist":train_epoch_iou_cls[7],
                    "train_iou_motorcyclist":train_epoch_iou_cls[8],
                    "train_iou_road":train_epoch_iou_cls[9],
                    "train_iou_parking":train_epoch_iou_cls[10],
                    "train_iou_sidewalk":train_epoch_iou_cls[11],
                    "train_iou_oter-ground":train_epoch_iou_cls[12],
                    "train_iou_building":train_epoch_iou_cls[13],
                    "train_iou_fence":train_epoch_iou_cls[14],
                    "train_iou_vegetation":train_epoch_iou_cls[15],
                    "train_iou_trunk":train_epoch_iou_cls[15],
                    "train_iou_terrain":train_epoch_iou_cls[17],
                    "train_iou_pole":train_epoch_iou_cls[18],
                    "train_iou_traffic-sign":train_epoch_iou_cls[19],
                    "valid_loss": valid_epoch_loss,
                    'valid_loss_xent': valid_epoch_loss_xent,
                    'valid_loss_ls': valid_epoch_loss_ls,
                    "valid_iou": valid_epoch_iou,
                    "valid_acc": valid_epoch_acc,
                    "valid_iou_car":valid_epoch_iou_cls[1],
                    "valid_iou_bicycle":valid_epoch_iou_cls[2],
                    "valid_iou_motorcycle":valid_epoch_iou_cls[3],
                    "valid_iou_truck":valid_epoch_iou_cls[4],
                    "valid_iou_other-vehicle":valid_epoch_iou_cls[5],
                    "valid_iou_persom":valid_epoch_iou_cls[6],
                    "valid_iou_bicyclist":valid_epoch_iou_cls[7],
                    "valid_iou_motorcyclist":valid_epoch_iou_cls[8],
                    "valid_iou_road":valid_epoch_iou_cls[9],
                    "valid_iou_parking":valid_epoch_iou_cls[10],
                    "valid_iou_sidewalk":valid_epoch_iou_cls[11],
                    "valid_iou_oter-ground":valid_epoch_iou_cls[12],
                    "valid_iou_building":valid_epoch_iou_cls[13],
                    "valid_iou_fence":valid_epoch_iou_cls[14],
                    "valid_iou_vegetation":valid_epoch_iou_cls[15],
                    "valid_iou_trunk":valid_epoch_iou_cls[15],
                    "valid_iou_terrain":valid_epoch_iou_cls[17],
                    "valid_iou_pole":valid_epoch_iou_cls[18],
                    "valid_iou_traffic-sign":valid_epoch_iou_cls[19]
                    })
            
                # 保存val_iou最好的模型
                if valid_epoch_iou > best_val_iou:
                    print(f"Best mean iou in validation so far till epoch {epoch}, save model!")
                    print("*" * 80)
                    print(f"val_miou improved from {best_val_iou} to {valid_epoch_iou}")
                    best_val_iou = valid_epoch_iou
                    self.model_single.save_checkpoint(self.log, suffix="_best_val")
                else:
                    print(f"val_miou did not improve from {best_val_iou} at epoch {epoch}")
                print("*" * 80)
            else:
                # because no valsid, so no cal_info
                # 保存epoch训练结果
                wandb.log({
                    "learning_rate": lr, 
                    "train_loss": train_epoch_loss,
                    "train_loss_xent": train_epoch_loss_xent,
                    "train_loss_ls": train_epoch_loss_ls,
                    "train_iou": train_epoch_iou,
                    "train_acc": train_epoch_acc,
                    "train_iou_car":train_epoch_iou_cls[1],
                    "train_iou_bicycle":train_epoch_iou_cls[2],
                    "train_iou_motorcycle":train_epoch_iou_cls[3],
                    "train_iou_truck":train_epoch_iou_cls[4],
                    "train_iou_other-vehicle":train_epoch_iou_cls[5],
                    "train_iou_persom":train_epoch_iou_cls[6],
                    "train_iou_bicyclist":train_epoch_iou_cls[7],
                    "train_iou_motorcyclist":train_epoch_iou_cls[8],
                    "train_iou_road":train_epoch_iou_cls[9],
                    "train_iou_parking":train_epoch_iou_cls[10],
                    "train_iou_sidewalk":train_epoch_iou_cls[11],
                    "train_iou_oter-ground":train_epoch_iou_cls[12],
                    "train_iou_building":train_epoch_iou_cls[13],
                    "train_iou_fence":train_epoch_iou_cls[14],
                    "train_iou_vegetation":train_epoch_iou_cls[15],
                    "train_iou_trunk":train_epoch_iou_cls[15],
                    "train_iou_terrain":train_epoch_iou_cls[17],
                    "train_iou_pole":train_epoch_iou_cls[18],
                    "train_iou_traffic-sign":train_epoch_iou_cls[19]
                    })
            
            # save to log
            '''Trainer.save_to_log(logdir=self.log,
                                logger=self.tb_logger_epoch,
                                info=self.info_epoch,
                                epoch=epoch,
                                w_summary=self.ARCH["train"]["save_summary"],
                                model=self.model_single,
                                img_summary=self.ARCH["train"]["save_scans"],
                                imgs=valid_epoch_rand_img)'''
        print(f'Training finish!')
        wandb.finish()
        return
    
    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, color_fn, 
                    report=10, show_scans=False, LS_criterion=None,post=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        running_loss = AverageMeter()
        running_loss_xent = AverageMeter()
        running_loss_ls = AverageMeter()
        epoch_acc = AverageMeter()
        epoch_iou = AverageMeter()
        epoch_iou_cls = []
        for ii in range(20):
            epoch_iou_cls.append(AverageMeter())
        update_ratio_meter = AverageMeter()
        epoch_size = len(train_loader)
        
        fine_tuen_flag = False
        if epoch>=10:
            print (f'now fine tune')
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = False
                    print (f'now params in encoder are forzen!')
                fine_tuen_flag = True

        # empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()
        # set model in train
        model.train()

        scaler = torch.amp.GradScaler()  # 混合精度训练
        accumulation_steps = 1

        end = time.time()

        for i, (in_vol, proj_mask, proj_labels, unproj_labels, path_seq, path_name,  p_x, p_y, proj_range, unproj_range, _, _, _, _, n_points) in enumerate(train_loader):
            # proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points
            data_time.update(time.time() - end)
            # print(f'file name: {path_name}')
            '''if i+1>1:
                break'''

            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
                proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda(non_blocking=True).long()
                p_x = p_x.cuda()
                p_y = p_y.cuda()
                unproj_labels = unproj_labels.cuda()
                proj_range= proj_range.cuda()
                unproj_range = unproj_range.cuda()

            # tensor check
            check_tensor(in_vol, 'in_vol')
            check_tensor(proj_mask,'proj_mask')
            check_tensor(proj_labels, 'proj_labels')
            # print("in_vol: min={}, max={}, mean={}".format( in_vol.min().item(), in_vol.max().item(), in_vol.mean().item()))
            # bsp: in_vol: min=-33.60822296142578, max=9.000232696533203, mean=-0.10179390013217926
            
            # print("proj_mask: min={}, max={}, mean={}".format( proj_mask.float().min().item(), proj_mask.float().max().item(), proj_mask.float().mean().item()))
            # proj_mask: min=0.0, max=1.0, mean=0.7641754150390625

            # check mask/label values
            if proj_mask.float().min().item() < 0.0 or proj_mask.float().max().item()>1.0:
                print("proj_mask: min={}, max={}, mean={}".format( proj_mask.float().min().item(), proj_mask.float().max().item(), proj_mask.float().mean().item()))
            if proj_labels.float().min().item() <0.0 or proj_labels.float().max().item()>19.0:
                print("proj_labels: min={}, max={}, mean={}".format( proj_labels.float().min().item(), proj_labels.float().max().item(), proj_labels.float().mean().item()))
            # proj_labels: min=0.0, max=19.0, mean=8.990110397338867
            # print(f'---------------------------------')
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                outputs = model(in_vol, proj_mask) # shape of output is BCHW, shape of targets is BHW
                # print(f'size of outout of modle: {outputs.shape}') # torch.Size([8, 20, 64, 1024])
                # loss_xentropy = criterion(torch.log(torch.softmax(outputs.clamp(min=1e-8),dim=1)), targets)
                loss_xentropy = criterion(outputs,proj_labels)
                # print(f'loss_xent: {loss_xentropy}')
                if self.multi_gpu:
                    loss_xentropy = loss_xentropy.mean()
                if LS_criterion is None:
                    loss = loss_xentropy                    
                else:
                    loss_ls = LS_criterion(outputs, proj_labels) # loss_ls = LS_criterion(outputs, targets.long())
                    # print(f'loss_ls: {loss_ls}')
                    loss = 0.8*loss_xentropy + 1.0*loss_ls

                loss = loss / accumulation_steps
                loss_xentropy = loss_xentropy / accumulation_steps
                if LS_criterion is not None:
                    loss_ls = loss_ls / accumulation_steps

                
            scaler.scale(loss).backward()
            if (i+1) % accumulation_steps == 0 or (i+1)==len(train_loader):
                scaler.step(optimizer) # 更新权重
                scaler.update() # 更新scaler
                # optimizer.step()
                optimizer.zero_grad()   # 梯度清零  
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # 使用梯度裁剪限制模型中参数的梯度范数，max_norm 可以根据需要调整，例如这里设置为 5.0
            # update lr
            if not fine_tuen_flag: 
                scheduler.step()
            else: lr = 0.0001
            
            # 训练参数完成更新，接下来处理输出数据

            # 计算loss
            # loss = loss.mean()   
            # print(f'loss: {loss}')
            loss = loss.mean()
            loss_xentropy = loss_xentropy.mean()
            if LS_criterion is not None:
                loss_ls = loss_ls.mean() 
            running_loss.update(loss.item(), in_vol.size(0)) # in_vol.size(0) is batch_size
            running_loss_xent.update(loss_xentropy.item(), in_vol.size(0))
            running_loss_ls.update(loss_ls.item(), in_vol.size(0))  
            # print(f'loss: {loss}')

            # 计算各项指标: loss, acc, iou, iou_cls
            
            # 计算iou, iou_cls, acc
            with torch.no_grad():
                evaluator.reset()
                if not post:
                    print(f'proj pred back on pointcloud with using pred_2_pc')
                    unproj_argmax = self.pred_2_pc(outputs, pixel_x=p_x, pixel_y=p_y, n_of_points=n_points)
                else:
                    print(f'proj pred back on pointcloud with using knn')
                    proj_argmax = outputs.argmax(dim=1)
                    unproj_argmax_N = []
                    for n in np.arange(proj_argmax.shape[0]):
                        proj_argmax_n = proj_argmax[n]
                        proj_range_n =proj_range[n]
                        unproj_range_n = unproj_range[n]
                        p_x_n = p_x[n]
                        p_y_n = p_y[n]
                        unproj_argmax_n = self.post(proj_range_n,unproj_range_n,proj_argmax_n,p_x_n,p_y_n)
                        unproj_argmax_N.append(unproj_argmax_n)
                    unproj_argmax=unproj_argmax_N
                    unproj_argmax = torch.cat(unproj_argmax, dim = 0)
                    
                # print(f'size of unproj_argmax: {unproj_argmax.shape}') # size of unproj_argmax: torch.Size([64, 150000])
                # print(f'size of unproj_labels: {unproj_labels.shape}')#size of unproj_labels: torch.Size([8, 150000])  # size of unproj_labels: torch.Size([1, 150000])
                evaluator.addBatch(unproj_argmax, unproj_labels)
                accuracy = evaluator.getacc()
                miou, iou_per_cls = evaluator.getIoU()

            # scheduler.step()
            epoch_acc.update(accuracy.item(), in_vol.size(0))
            epoch_iou.update(miou.item(), in_vol.size(0))
            for cls in range(20):
                epoch_iou_cls[cls].update(iou_per_cls[cls].item(), in_vol.size(0))
            # print(f'Epoch: [{epoch}/{wandb.config.epochs}][{i}/{len(train_loader)}],miou:{miou.item():.4f}, acc:{accuracy.item():.4f}')
            batch_time.update(time.time() - end)
            end = time.time()

            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) * value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))

            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if (i+1) % self.ARCH["train"]["report_batch"] == 0:
                print('Lr: {lr:.3e} | '
                      'Epoch: [{0}][{1}/{2}] | '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                      'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                      'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(epoch, i+1, len(train_loader), batch_time=batch_time,
                                                                 data_time=data_time, loss=running_loss, acc=epoch_acc, iou=epoch_iou, lr=lr))

            self.info_iter["train_update"] = update_mean
            self.info_iter["train_loss"] = float(loss)
            self.info_iter["train_acc"] = epoch_acc.val
            self.info_iter["train_iou"] = epoch_iou.val
            # save as iteration
            '''Trainer.save_to_log(logdir=self.log,
                                logger=self.tb_logger_iter,
                                info=self.info_iter,
                                epoch=epoch*len(train_loader)+i,
                                w_summary=self.ARCH["train"]["save_summary"],
                                model=self.model_single)'''

        
        epoch_iou_cls_avg = []
        for cls in range(20):
            epoch_iou_cls_avg.append(epoch_iou_cls[cls].avg)
        
        
        # wandb.log({"train_loss": epoch_loss.avg, "train_iou": epoch_iou.avg, "train_acc": epoch_acc.avg, "learning_rate": lr})

        print(f'after {epoch+1}-th epoch:')
        print('-'*80)
        print(f"train_Epoch [{epoch}/{wandb.config.epochs}], Loss: {running_loss.avg:.4f}, LR: {lr:.6f}, mIOU: {epoch_iou.avg:.4f}, mAcc: {epoch_acc.avg:.4f}")
        print(f'train_iou per class : {epoch_iou_cls_avg} ') # 打印每个类别的iou

        # train_epoch_acc, train_epoch_iou, train_epoch_iou_cls , train_epoch_loss, train_epoch_loss_xent, train_epoch_loss_ls,update_mean
        return epoch_acc.avg, epoch_iou.avg, epoch_iou_cls_avg, running_loss.avg, running_loss_xent.avg, running_loss_ls.avg, update_ratio_meter.avg, lr      
    
    def validate_epoch(self, valid_loader, model, criterion, evaluator, epoch,class_func,color_fn, save_scans=False, LS_criterion=None,post=None):
        batch_time = AverageMeter()
        running_loss = AverageMeter()
        running_loss_xent = AverageMeter()
        running_loss_ls = AverageMeter()
        epoch_acc = AverageMeter()
        epoch_iou = AverageMeter()
        epoch_iou_cls = []
        for ii in range(20):
            epoch_iou_cls.append(AverageMeter())
        rand_imgs = []
        

        model.eval()
        evaluator.reset()
        epoch_size = len(valid_loader)

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()
        # start valid    
        with torch.no_grad():
            end = time.time()

            for i, (in_vol, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, n_points) in enumerate(valid_loader):
                # in_vol, proj_mask, proj_labels, unproj_labels, path_seq, path_name,  p_x, p_y, _, _, _, _, _, _, n_points
                '''if i+1>1:
                    break'''
                # batch_size = in_vol.shape[0]
                '''p_x = p_x[:, :n_points.item()]
                p_y = p_y[:, :n_points.item()]
                unproj_labels = unproj_labels[:, :n_points.item()]'''
                # print(f'size of p_x: {p_x.shape}') # size of p_x: torch.Size([124668])
                # print(f'size of p_y: {p_y.shape}') # size of p_y: torch.Size([124668])
                # print(f'size of unproj_labels: {unproj_labels.shape}') # size of unproj_labels: torch.Size([124668])
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()
                    unproj_labels = unproj_labels.cuda()
                    proj_range= proj_range.cuda()
                    unproj_range = unproj_range.cuda()

                outputs = model(in_vol, proj_mask)
                # loss_xentropy = criterion(torch.log(torch.softmax(outputs.clamp(min=1e-8),dim=1)), proj_labels)
                loss_xentropy = criterion(outputs,proj_labels)
                if LS_criterion is None:
                    loss = loss_xentropy
                else:
                    loss_ls = LS_criterion(outputs, proj_labels.long())
                    loss = 0.8*loss_ls + 1.0*loss_xentropy

                # 计算loss
                running_loss.update(loss.mean().item(), in_vol.size(0))
                running_loss_xent.update(loss_xentropy.mean().item(), in_vol.size(0))
                running_loss_ls.update(loss_ls.mean().item(), in_vol.size(0))
                
                # 计算acc, iou, iou_cls
                if not post:
                    unproj_argmax = self.pred_2_pc(outputs, pixel_x=p_x, pixel_y=p_y, n_of_points=n_points)
                else:
                    # knn postproc
                    proj_argmax = outputs.argmax(dim=1)
                    unproj_argmax_N = []
                    for n in np.arange(proj_argmax.shape[0]):
                        proj_argmax_n = proj_argmax[n]
                        proj_range_n =proj_range[n]
                        unproj_range_n = unproj_range[n]
                        p_x_n = p_x[n]
                        p_y_n = p_y[n]
                        unproj_argmax_n = self.post(proj_range_n,unproj_range_n,proj_argmax_n,p_x_n,p_y_n)
                        unproj_argmax_N.append(unproj_argmax_n)
                    unproj_argmax=unproj_argmax_N
                    unproj_argmax = torch.cat(unproj_argmax, dim = 0)
                # print(f'size of unproj_argmax: {unproj_argmax.shape}') # size of unproj_argmax: torch.Size([124668, 1024])
                # print(f'size of unproj_labels: {unproj_labels.shape}') # size of unproj_labels: torch.Size([1, 150000])
                evaluator.addBatch(unproj_argmax, unproj_labels)
                accuracy = evaluator.getacc()

                miou, iou_per_cls = evaluator.getIoU()
                epoch_iou.update(miou.item(), in_vol.size(0))
                epoch_acc.update(accuracy.item(), in_vol.size(0))
                for cls in range(20):
                    epoch_iou_cls[cls].update(iou_per_cls[cls].item(), in_vol.size(0))

                if save_scans:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = Trainer.make_log_img(depth_np,
                                                mask_np,
                                                pred_np,
                                                gt_np,
                                                color_fn)
                    rand_imgs.append(out)
                # 计算时间    
                batch_time.update(time.time() - end)
                end = time.time()

        epoch_iou_cls_avg = []
        for cls in range(20):
            epoch_iou_cls_avg.append(epoch_iou_cls[cls].avg)        
        # wandb.log({"valid_loss": running_loss.avg, 'valid_loss_xent': running_loss_xent.avg, 'valid_loss_ls': running_loss_ls.avg, "valid_iou": epoch_iou.avg, "valid_acc": epoch_acc.avg})
        
        
        # valid_epoch_acc, valid_epoch_iou, valid_epoch_iou_cls, valid_epoch_loss, valid_epoch_loss_xent, valid_epoch_loss_ls, valid_epoch_rand_img
        # acc.avg, iou.avg, iou_cls_avg, losses.avg, losses_xentropy.avg, losses_ls.avg, rand_imgs
        return epoch_acc.avg, epoch_iou.avg, epoch_iou_cls_avg, running_loss.avg, running_loss_xent.avg, running_loss_ls.avg, rand_imgs
        
