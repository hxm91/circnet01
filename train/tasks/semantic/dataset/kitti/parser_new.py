# encoding=utf-8
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('/mnt/nas/code/Solingen_MOT/range/fpsnet/train')

from common.laserscan import LaserScan, SemLaserScan

# 定义文件后缀为元组，便于高效判断
EXTENSIONS_SCAN = ('.bin',)
EXTENSIONS_LABEL = ('.label',)

def is_scan(filename):
    return filename.endswith(EXTENSIONS_SCAN)

def is_label(filename):
    return filename.endswith(EXTENSIONS_LABEL)


class SemanticKitti(Dataset):
    def __init__(self, root, sequences, labels, color_map,
                 learning_map, learning_map_inv, sensor,
                 max_points=150000, gt=True, transform=None):
        # 保存各项参数
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt
        self.transform = transform

        # 注意：训练时使用的类别数取决于 learning_map_inv 的大小
        self.nclasses = len(self.learning_map_inv)

        # 检查目录和参数类型
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from {}".format(self.root))
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        assert isinstance(self.labels, dict)
        assert isinstance(self.color_map, dict)
        assert isinstance(self.learning_map, dict)
        assert isinstance(self.sequences, list)

        self.scan_files = []
        self.label_files = []

        # 遍历每个序列，查找扫描和标签文件
        for seq in self.sequences:
            seq_str = '{0:02d}'.format(int(seq))
            print("Parsing sequence {}".format(seq_str))
            scan_path = os.path.join(self.root, seq_str, "velodyne")
            label_path = os.path.join(self.root, seq_str, "labels")
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_path))
                          for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_path))
                           for f in fn if is_label(f)]
            if self.gt:
                assert len(scan_files) == len(label_files), \
                    "Mismatch between scan and label file counts in sequence {}".format(seq_str)
            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)

        self.scan_files.sort()
        self.label_files.sort()
        print("Using {} scans from sequences {}".format(len(self.scan_files), self.sequences))

        # 预先构造映射查找表，避免在 __getitem__ 时重复构造
        self.learning_map_lut = self._create_lut(self.learning_map)
        self.learning_map_inv_lut = self._create_lut(self.learning_map_inv)
        self.color_map_lut = self._create_lut(self.color_map)

    def __getitem__(self, index):
        # 获取第 index 个扫描文件（以及对应标签文件）
        scan_file = self.scan_files[index]
        if self.gt:
            label_file = self.label_files[index]

        # 根据是否需要 ground truth 选择不同的激光扫描解析器
        if self.gt:
            scan = SemLaserScan(self.color_map, project=True,
                                H=self.sensor_img_H, W=self.sensor_img_W,
                                fov_up=self.sensor_fov_up, fov_down=self.sensor_fov_down,
                                transform=self.transform)
        else:
            scan = LaserScan(project=True,
                             H=self.sensor_img_H, W=self.sensor_img_W,
                             fov_up=self.sensor_fov_up, fov_down=self.sensor_fov_down,
                             transform=self.transform)

        # 打开扫描文件（内部实现完成投影）
        scan.open_scan(scan_file)
        if self.gt:
            scan.open_label(label_file)
            # 利用预先计算好的 LUT 进行映射转换
            scan.sem_label = self.learning_map_lut[scan.sem_label]
            scan.proj_sem_label = self.learning_map_lut[scan.proj_sem_label]

        # 构造 unprojected（原始点云）数据
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full((self.max_points,), -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full((self.max_points,), -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
            unproj_labels = torch.full((self.max_points,), -1, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        # 获取投影数据（图像形式）
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []
        proj_x = torch.full((self.max_points,), -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full((self.max_points,), -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0),
                          proj_xyz.permute(2, 0, 1),
                          proj_remission.unsqueeze(0)])
        # 标准化并遮罩无效像素
        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        # 解析路径信息
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        return (proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name,
                proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz,
                proj_remission, unproj_remissions, unproj_n_points)

    def __len__(self):
        return len(self.scan_files)

    @staticmethod
    def _create_lut(mapdict):
        """
        构造查找表（LUT）：
          若字典中值为列表，则返回形状为 (max_key+100, len(list)) 的数组，
          否则返回一维数组。这里“+100”作为安全冗余。
        """
        if not mapdict:
            raise ValueError("Mapping dictionary is empty")
        maxkey = max(mapdict.keys())
        sample_value = next(iter(mapdict.values()))
        if isinstance(sample_value, list):
            nel = len(sample_value)
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key", key)
        return lut

    @staticmethod
    def map(label, mapdict):
        # 该函数为旧版本实现，建议使用预计算的 LUT
        maxkey = max(mapdict.keys())
        sample_value = next(iter(mapdict.values()))
        if isinstance(sample_value, list):
            nel = len(sample_value)
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key", key)
        return lut[label]


class Parser:
    def __init__(self,
                 root,
                 train_sequences,
                 valid_sequences,
                 test_sequences,
                 labels,
                 color_map,
                 learning_map,
                 learning_map_inv,
                 sensor,
                 max_points,
                 batch_size,
                 workers,
                 gt=True,
                 shuffle_train=True,
                 transform=None):
        super(Parser, self).__init__()
        self.root = root
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.max_points = max_points
        self.batch_size = batch_size
        self.workers = workers
        self.gt = gt
        self.shuffle_train = shuffle_train
        self.transform = transform

        self.nclasses = len(self.learning_map_inv)

        self.train_dataset = SemanticKitti(root=self.root,
                                           sequences=self.train_sequences,
                                           labels=self.labels,
                                           color_map=self.color_map,
                                           learning_map=self.learning_map,
                                           learning_map_inv=self.learning_map_inv,
                                           sensor=self.sensor,
                                           max_points=self.max_points,
                                           gt=self.gt,
                                           transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=self.shuffle_train,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       drop_last=True)
        assert len(self.trainloader) > 0
        self.trainiter = iter(self.trainloader)

        self.valid_dataset = SemanticKitti(root=self.root,
                                           sequences=self.valid_sequences,
                                           labels=self.labels,
                                           color_map=self.color_map,
                                           learning_map=self.learning_map,
                                           learning_map_inv=self.learning_map_inv,
                                           sensor=self.sensor,
                                           max_points=self.max_points,
                                           gt=self.gt)
        self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       drop_last=True)
        assert len(self.validloader) > 0
        self.validiter = iter(self.validloader)

        if self.test_sequences:
            self.test_dataset = SemanticKitti(root=self.root,
                                              sequences=self.test_sequences,
                                              labels=self.labels,
                                              color_map=self.color_map,
                                              learning_map=self.learning_map,
                                              learning_map_inv=self.learning_map_inv,
                                              sensor=self.sensor,
                                              max_points=self.max_points,
                                              gt=False)
            self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          num_workers=self.workers,
                                                          pin_memory=True,
                                                          drop_last=True)
            assert len(self.testloader) > 0
            self.testiter = iter(self.testloader)

    def get_train_batch(self):
        return next(self.trainiter)

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        return next(self.validiter)

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        return next(self.testiter)

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)

    def get_n_classes(self):
        return self.nclasses

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        return SemanticKitti.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        return SemanticKitti.map(label, self.learning_map)

    def to_color(self, label):
        label = SemanticKitti.map(label, self.learning_map_inv)
        return SemanticKitti.map(label, self.color_map)
