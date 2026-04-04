#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np

# ---------------------------
# LaserScan 类：用于读取点云文件，并（可选）将点云投影到图像平面
# ---------------------------
class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ('.bin',)  # 使用元组存储扩展名，便于高效判断

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, transform=None):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.transform = transform
        self.reset()

    def reset(self):
        """Reset scan members and preallocate projection arrays."""
        # 点云和反投影相关数据（使用 1D 数组存储，便于后续索引）
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.remissions = np.zeros((0,), dtype=np.float32)

        # 预分配投影图（图像尺寸为 [H,W] 或 [H,W,3]）
        self.proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        self.unproj_range = np.zeros((0,), dtype=np.float32)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        # proj_x 和 proj_y 将存储每个点在投影图中的索引（1D 数组）
        self.proj_x = np.zeros((0,), dtype=np.int32)
        self.proj_y = np.zeros((0,), dtype=np.int32)
        self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)

    def size(self):
        """Return the number of points in the scan."""
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """Open raw scan file and fill in attributes."""
        self.reset()
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, but was {}".format(type(filename)))
        if not filename.endswith(self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")
        # 如果点云文件较大，可选使用 np.memmap 代替 np.fromfile
        scan = np.fromfile(filename, dtype=np.float32)
        try:
            scan = scan.reshape((-1, 4))
        except ValueError as e:
            raise ValueError("Error reshaping scan data. Check file format: " + str(e))
        points = scan[:, :3]
        remissions = scan[:, 3]
        if self.transform is not None:
            points = self.transform(points)
        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """Set point cloud and remission data, and compute projection if needed."""
        self.reset()
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")
        self.points = points
        if remissions is not None:
            self.remissions = remissions.reshape(-1)  # 确保为1D数组
        else:
            self.remissions = np.zeros((points.shape[0],), dtype=np.float32)
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        """Project the point cloud into a spherical projection image."""
        # 将视野角度转换为弧度（预先计算常量可略微加速）
        fov_up = self.proj_fov_up * np.pi / 180.0
        fov_down = self.proj_fov_down * np.pi / 180.0
        fov = abs(fov_down) + abs(fov_up)

        # 计算每个点的深度（使用向量化计算，np.linalg.norm 已经过优化）
        depth = np.linalg.norm(self.points, axis=1)
        # 分别获取 x, y, z 分量
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # 计算水平角（yaw）和垂直角（pitch）
        yaw = -np.arctan2(scan_y, scan_x)
        # 为避免除零错误，可稍作保护（理论上深度为零的点很少见）
        pitch = np.arcsin(np.clip(scan_z / (depth + 1e-6), -1.0, 1.0))

        # 将角度归一化到 [0,1] 区间，再映射到图像尺寸
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov
        proj_x = np.clip(np.floor(proj_x), 0, self.proj_W - 1).astype(np.int32)
        proj_y = np.clip(np.floor(proj_y), 0, self.proj_H - 1).astype(np.int32)
        # 保留原始计算结果（复制时保证内存独立）
        self.proj_x = proj_x.copy()
        self.proj_y = proj_y.copy()

        self.unproj_range = depth.copy()

        # 对点云按照深度降序排序，确保最近的点覆盖较远的点
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # 将排序后的数据赋值到预分配的投影图中
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        # proj_mask：所有有效点索引（>=0）的像素置为 1
        self.proj_mask = (self.proj_idx >= 0).astype(np.int32)


# ---------------------------
# SemLaserScan 类：扩展 LaserScan，增加语义及实例标签信息
# ---------------------------
class SemLaserScan(LaserScan):
    """LaserScan with semantic labels, colors and instance labels."""
    EXTENSIONS_LABEL = ('.label',)

    def __init__(self, sem_color_dict=None, project=False, H=64, W=1024,
                 fov_up=3.0, fov_down=-25.0, max_classes=300, transform=None):
        super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down, transform)
        self.reset()
        # 构造语义颜色查找表（LUT）
        if sem_color_dict:
            max_sem_key = max(sem_color_dict.keys()) + 1
            self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
            for key, value in sem_color_dict.items():
                self.sem_color_lut[key] = np.array(value, dtype=np.float32) / 255.0
        else:
            max_sem_key = max_classes
            self.sem_color_lut = np.random.uniform(low=0.0, high=1.0, size=(max_sem_key, 3))
            self.sem_color_lut[0] = np.full((3,), 0.1)

        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
        self.inst_color_lut[0] = np.full((3,), 0.1)

    def reset(self):
        """Reset scan members including semantic and instance labels."""
        super(SemLaserScan, self).reset()
        self.sem_label = np.zeros((0,), dtype=np.int32)
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)
        self.inst_label = np.zeros((0,), dtype=np.int32)
        self.inst_label_color = np.zeros((0, 3), dtype=np.float32)
        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        self.proj_inst_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)

    def open_label(self, filename):
        """Open raw label file and fill in semantic attributes."""
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, but was {}".format(type(filename)))
        if not filename.endswith(self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")
        label = np.fromfile(filename, dtype=np.int32).reshape(-1)
        self.set_label(label)

    def set_label(self, label):
        """Set semantic and instance labels from a numpy array."""
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")
        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF
            self.inst_label = label >> 16
        else:
            print("Points shape:", self.points.shape)
            print("Label shape:", label.shape)
            raise ValueError("Scan and Label don't contain the same number of points")
        # 检查映射是否正确
        assert ((self.sem_label + (self.inst_label << 16) == label).all())
        if self.project:
            self.do_label_projection()

    def colorize(self):
        """Colorize pointcloud using semantic and instance labels."""
        self.sem_label_color = self.sem_color_lut[self.sem_label].reshape((-1, 3))
        self.inst_label_color = self.inst_color_lut[self.inst_label].reshape((-1, 3))

    def do_label_projection(self):
        """Project labels and assign color information for visualization."""
        mask = self.proj_idx >= 0
        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]
        self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
        self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]
