from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def image_summary(self, tag, images, step):
        for i, img in enumerate(images):
            if not isinstance(img, np.ndarray):
                img = np.array(img)

            if img.ndim == 2:
                img = img[None, :, :]          # 1,H,W
            elif img.ndim == 3:
                img = np.transpose(img, (2, 0, 1))  # H,W,C -> C,H,W

            self.writer.add_image(f"{tag}/{i}", img, step)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        self.writer.add_histogram(tag, values, step, bins='tensorflow')
        self.writer.flush()

    def multi_scalar_summary(self, tag, values, step):
        for i, val in enumerate(values):
            self.writer.add_scalar(f"{tag}_{i}", val, step)
        self.writer.flush()

    def close(self):
        self.writer.close()
        
'''#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# status 20250212: this file is not used because error 'cuda out of memory' which occured with tensorflow.
import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置每个 GPU 的内存上限为 400MB， 建立一个显存为400mb的虚拟gpu，固定使用gpu0的400mb显存。
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=400)] 
        )
    except RuntimeError as e:
        print(e)

import numpy as np
from PIL import Image
from io import BytesIO  # Python 3.x


class Logger:
    def __init__(self, log_dir, num=20):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)
        self.num = num

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images.

        Args:
            tag: Base tag for images.
            images: A list (or array) of images. Each image is expected to be a numpy array
                    of shape [H, W, C] (C=1 or 3). If image dtype is not uint8 and max value ≤ 1,
                    it will be scaled to 0–255.
            step: Training step.
        """
        with self.writer.as_default():
            for i, img in enumerate(images):
                # 确保图像为 numpy 数组
                if not isinstance(img, np.ndarray):
                    img = np.array(img)
                # 如果图像类型不是 uint8 且最大值小于等于 1，则将其放大到 [0,255]
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                # 添加 batch 维度（tf.summary.image 要求输入形状为 [batch, H, W, C]）
                if img.ndim == 3:
                    img = np.expand_dims(img, axis=0)
                tf.summary.image(f"{tag}/{i}", img, step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step, buckets=bins)
            self.writer.flush()

    def multi_scalar_summary(self, tag, values, step):
        """Log multiple scalar values, one per index."""
        with self.writer.as_default():
            for i, val in enumerate(values):
                tf.summary.scalar(f"{tag}_{i}", val, step=step)
            self.writer.flush()
'''