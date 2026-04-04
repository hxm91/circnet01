#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# mode='circular'

import argparse
import datetime
import yaml
import os
import sys
import shutil
from shutil import copyfile

import __init__ as booger

from tasks.semantic.modules.trainer import Trainer
import tasks.semantic.dataset.kitti.transform as T
from utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Training script for semantic segmentation'
    )
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Dataset to train with. (No default)')
    parser.add_argument('--arch_cfg', '-ac', type=str, required=True,
                        help='Architecture YAML config file. See /config/arch for sample.')
    parser.add_argument('--data_cfg', '-dc', type=str, default='config/labels/semantic-kitti.yaml',
                        help='Classification YAML config file. See /config/labels for sample.')
    # 注意：部分平台可能不支持 %-m 格式，若有问题请改为 %m
    log_dir_default = os.path.join(
        './logs',
        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    )
    parser.add_argument('--log', '-l', type=str, default=log_dir_default,
                        help='Directory to put the log data. Default: ./logs/date+time')
    parser.add_argument('--pretrained', '-p', type=str, default=None,
                        help='Directory to get the pretrained model. If not passed, train from scratch.')
    parser.add_argument('--gpu', nargs='+', type=int, required=True,
                        help='GPU IDs to use. (No default)')
    FLAGS, unparsed = parser.parse_known_args()

    # 打印参数摘要
    print("----------")
    print("INTERFACE:")
    print("dataset:", FLAGS.dataset)
    print("arch_cfg:", FLAGS.arch_cfg)
    print("data_cfg:", FLAGS.data_cfg)
    print("log:", FLAGS.log)
    print("pretrained:", FLAGS.pretrained)
    print("gpu:", FLAGS.gpu)
    print("----------\n")

    # 读取配置文件
    try:
        print(f"Opening arch config file {FLAGS.arch_cfg}")
        with open(FLAGS.arch_cfg, 'r') as f:
            ARCH = yaml.safe_load(f)
    except Exception as e:
        print("Error opening arch YAML file:", e)
        sys.exit(1)

    try:
        print(f"Opening data config file {FLAGS.data_cfg}")
        with open(FLAGS.data_cfg, 'r') as f:
            DATA = yaml.safe_load(f)
    except Exception as e:
        print("Error opening data YAML file:", e)
        sys.exit(1)

    # 创建日志目录：如果已存在则删除后重建
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log, exist_ok=True)
    except Exception as e:
        print("Error creating log directory. Check permissions!", e)
        sys.exit(1)

    # 将日志输出重定向到 log.txt
    log_file = os.path.join(FLAGS.log, 'log.txt')
    sys.stdout = Logger(log_file)

    # 检查 pretrained 模型文件夹是否存在
    if FLAGS.pretrained is not None:
        if os.path.isdir(FLAGS.pretrained):
            print(f"Model folder exists! Using model from {FLAGS.pretrained}")
        else:
            print("Pretrained model folder doesn't exist! Start with random weights...")
    else:
        print("No pretrained directory provided.")

    # 将关键配置文件复制到日志目录以便后续查阅
    try:
        print(f"Copying configuration files to {FLAGS.log} for future reference.")
        copyfile(FLAGS.arch_cfg, os.path.join(FLAGS.log, "arch_cfg.yaml"))
        copyfile(FLAGS.data_cfg, os.path.join(FLAGS.log, "data_cfg.yaml"))
        # 备份 encoder 和 decoder 文件到日志目录
        encoder_src = os.path.join(booger.TRAIN_PATH, 'backbones', ARCH["backbone"]["name"] + '.py')
        encoder_dst = os.path.join(FLAGS.log, 'encoder_' + ARCH["backbone"]["name"] + '.py')
        shutil.copy(encoder_src, encoder_dst)

        decoder_src = os.path.join(booger.TRAIN_PATH, 'tasks/semantic/decoders', ARCH["decoder"]["name"] + '.py')
        decoder_dst = os.path.join(FLAGS.log, 'decoder_' + ARCH["decoder"]["name"] + '.py')
        shutil.copy(decoder_src, decoder_dst)

        trainer_src = os.path.join(booger.TRAIN_PATH, 'tasks/semantic/modules/trainer.py')
        trainer_dst = os.path.join(FLAGS.log, 'trainer.py')
        shutil.copy(trainer_src, trainer_dst)
    except Exception as e:
        print("Error copying files, check permissions. Exiting...", e)
        sys.exit(1)

    # 构造数据增强（transform）
    transform = T.Compose([T.RandomRotation(),
                           T.RandomLeftRightFlip()])

    # 创建 Trainer 对象并启动训练
    trainer = Trainer(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.pretrained, FLAGS.gpu, transform)
    trainer.train()


if __name__ == '__main__':
    main()
