import numpy as np
import yaml
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def xyz_to_range(fov, pts, img_h, img_w):
    """
    将点云转换为 Range-Image 表示，使得可以利用二维方法进行三维分割。

    参数:
        fov: tuple, (fov_down, fov_up)（单位：度）
        pts: numpy.ndarray, 原始点云数据，形状 (N,4)，每行为 [x, y, z, remission]
        img_h: int, 输出图像高度
        img_w: int, 输出图像宽度

    返回:
        points_aug: 增强后的点云，每个点附加了 [x, y, z, remission, depth, proj_x, proj_y]
        proj_mask: (img_h, img_w) 的二值图像，像素有点则为1
        proj_range: (img_h, img_w) 的深度图
        proj_xyz: (img_h, img_w, 3) 的三维坐标图
        proj_xyzd: (img_h, img_w, 4) 的 [x, y, z, depth] 图
        proj_remission: (img_h, img_w) 的反射强度图
        proj_idx: (img_h, img_w) 的原始点索引图
        proj_xyzrd: (img_h, img_w, 5) 的 [x, y, z, remission, depth] 图
    """
    # 提取 x, y, z, remission
    points = pts[:, :4].copy()
    x, y, z, remission = points.T

    # 将视场角从度转换为弧度，并计算总 FOV
    fov_down, fov_up = np.deg2rad(fov)
    FOV = np.abs(fov_down) + np.abs(fov_up)

    # 输出图像尺寸
    proj_H, proj_W = img_h, img_w

    # 初始化投影图像
    proj_xyz = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)
    proj_xyzd = np.full((proj_H, proj_W, 4), -1, dtype=np.float32)
    proj_range = np.full((proj_H, proj_W), -1, dtype=np.float32)
    proj_remission = np.full((proj_H, proj_W), -1, dtype=np.float32)
    proj_idx = np.full((proj_H, proj_W), -1, dtype=np.int32)
    proj_xyzrd = np.full((proj_H, proj_W, 5), -1, dtype=np.float32)

    # 计算深度
    depth = np.linalg.norm(points[:, :3], axis=1)
    # 拼接深度信息：点云增加一列 [x, y, z, remission, depth]
    points_with_depth = np.hstack((points, depth.reshape(-1, 1)))
    points_xyzrd_full = points_with_depth.copy()  # [x,y,z,remission,depth]
    points_xyzd_full = np.hstack((points[:, :3], depth.reshape(-1, 1)))

    # 计算投影坐标（视角投影）
    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(z / depth)
    proj_x = np.clip(np.floor(0.5 * (yaw / np.pi + 1.0) * proj_W), 0, proj_W - 1).astype(np.int32)
    proj_y = np.clip(np.floor((1.0 - (pitch - fov_down) / FOV) * proj_H), 0, proj_H - 1).astype(np.int32)

    # 拼接投影索引到点云中，形成增强后的点云数据
    # 每个点的数据为 [x, y, z, remission, depth, proj_x, proj_y]
    points_aug = np.hstack((points_with_depth,
                            proj_x.reshape(-1, 1),
                            proj_y.reshape(-1, 1)))

    # 按深度降序排序（从远到近，保证近处的点可以覆盖远处的点）
    order = np.argsort(depth)[::-1]
    points_aug = points_aug[order]
    sorted_depth = depth[order]
    sorted_proj_x = proj_x[order]
    sorted_proj_y = proj_y[order]
    sorted_remission = remission[order]
    sorted_indices = np.arange(points.shape[0])[order]

    # 将点云信息赋值到各个投影图中
    proj_range[sorted_proj_y, sorted_proj_x] = sorted_depth
    proj_xyz[sorted_proj_y, sorted_proj_x] = points_aug[:, :3]
    proj_xyzd[sorted_proj_y, sorted_proj_x] = points_xyzd_full[order]
    proj_xyzrd[sorted_proj_y, sorted_proj_x] = points_xyzrd_full[order]
    proj_remission[sorted_proj_y, sorted_proj_x] = sorted_remission
    proj_idx[sorted_proj_y, sorted_proj_x] = sorted_indices

    proj_mask = (proj_idx > 0).astype(np.int32)
    return points_aug, proj_mask, proj_range, proj_xyz, proj_xyzd, proj_remission, proj_idx, proj_xyzrd


def load_velo(file_path):
    """
    从 .bin 文件中加载点云数据，返回 [x, y, z, remission] 四元组。
    """
    return np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))


def open_label(file_path):
    """
    从 .label 文件中加载标签数据。
    """
    return np.fromfile(file_path, dtype=np.uint32).reshape((-1))


def get_label(label, pts, proj_idx, img_h, img_w, config_file='../semantic/config/labels/semantic-kitti.yaml'):
    """
    根据点云的标签信息和投影索引，生成投影图上的语义标签图。

    参数:
        label: 原始标签，一维数组
        pts: 点云数据（要求点数与标签数一致）
        proj_idx: 每个像素对应原始点云中的索引 (img_h, img_w)
        img_h, img_w: 图像尺寸
        config_file: YAML 配置文件路径，其中包含 label 映射信息

    返回:
        proj_sem_label: (img_h, img_w) 的语义标签图
    """
    # 加载配置文件（建议配置文件只加载一次）
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # 建立标签映射表
    remapdict = config_data['learning_map']
    max_key = max(remapdict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    # 检查点云与标签数量是否匹配
    if label.shape[0] != pts.shape[0]:
        raise ValueError('点云与标签点数不匹配')

    # 提取语义标签（低16位）和实例标签（高16位）
    sem_label = label & 0xFFFF
    inst_label = label >> 16
    assert (sem_label + (inst_label << 16) == label).all(), "标签解析错误"

    # 使用 LUT 映射原始语义标签
    sem_label = remap_lut[sem_label]

    proj_sem_label = np.zeros((img_h, img_w), dtype=np.uint32)
    mask = proj_idx >= 0
    proj_sem_label[mask] = sem_label[proj_idx[mask]]
    return proj_sem_label


def plot_projection(proj_label, color_sem_dict, label_name_dict, save_path, dpi=1000):
    """
    根据投影标签图，利用查表的方式着色并生成可视化图像。

    参数:
        proj_label: (H, W) 语义标签图
        color_sem_dict: dict, key 为标签值，value 为颜色列表（0-255）
        label_name_dict: dict, key 为标签值，value 为类别名称
        save_path: 保存图像的路径
        dpi: 图像分辨率
    """
    H, W = proj_label.shape

    # 构造查找表（假设标签值较小且连续）
    max_label = max(color_sem_dict.keys())
    lut = np.zeros((max_label + 1, 3), dtype=np.float32)
    for key, color in color_sem_dict.items():
        lut[key] = np.array(color, dtype=np.float32)
    # 直接利用 LUT 索引赋色
    color_proj = lut[proj_label]

    # 构造图例
    unique_labels = np.unique(proj_label)
    patches = [
        mpatches.Patch(color=lut[c] / 255.0, label=label_name_dict.get(c, str(c)))
        for c in unique_labels
    ]

    plt.figure(dpi=dpi, figsize=(W / 100, H / 100 * 2.5))
    plt.imshow(color_proj / 255.0)
    plt.legend(handles=patches, loc=3, ncol=10, bbox_to_anchor=(0.1, 1.2),
               borderaxespad=0., fontsize=6)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_pc(pc, labels,color_sem_dict, label_name_dict, save_path, dpi=1000):
    x = pc[:,0]
    y = pc[:,1]
    max_label = max(color_sem_dict.keys())
    lut = np.zeros((max_label + 1, 3), dtype=np.float32)
    for key, color in color_sem_dict.items():
        lut[key] = np.array(color, dtype=np.float32)
    # 直接利用 LUT 索引赋色
    # color_proj = lut[labels]
    unique_labels = np.unique(labels)
    '''patches = [
        mpatches.Patch(color=lut[c] / 255.0, label=label_name_dict.get(c, str(c)))
        for c in unique_labels
    ]'''

    plt.figure(dpi=dpi)
    plt.scatter(x,y, c=lut)
    # plt.legend(handles=patches, loc=3, ncol=10, bbox_to_anchor=(0.1, 1.2),borderaxespad=0., fontsize=6)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # 文件路径设置
    # file_path = '/mnt/nas/Daten/Kitti_daten/KITTI/segmentation/dataset/sequences/00/velodyne/000000.bin'
    # label_path = '/mnt/nas/Daten/Kitti_daten/KITTI/segmentation/dataset/sequences/00/labels/000000.label'
    file_path = '/home/h/Dokumente/daten/KITTI/segmentation/dataset/sequences/00/velodyne/000001.bin'
    label_path = '/home/h/Dokumente/daten/KITTI/segmentation/dataset/sequences/00/labels/000001.label'
    pred_path = './visdata/00/prediction/000001.label'

    # LiDAR 参数
    fov = (-25.0, 3.0)
    H = 64
    W = 2048

    # 加载点云与标签数据
    scan = load_velo(file_path)
    points, proj_mask, proj_range, proj_xyz, proj_xyzd, proj_remission, proj_idx, proj_xyzrd = xyz_to_range(fov, scan, H, W)
    labels = open_label(label_path)
    preds = open_label(pred_path)

    # 生成投影标签图（语义）
    proj_sem_label = get_label(labels, points, proj_idx, H, W)
    pred_proj_sem_label = get_label(preds, points, proj_idx, H, W)

    # 加载配置文件（建议只加载一次）
    config_file = '../semantic/config/labels/semantic-kitti.yaml'
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # 构造类别名称和颜色查找字典
    learn_map_inv = config_data['learning_map_inv']
    labels_list = config_data['labels']
    label_name_dict = {l: labels_list[learn_map_inv[l]] for l in learn_map_inv}
    color_sem_dict = {l: config_data['color_map'][learn_map_inv[l]] for l in learn_map_inv}

    # 可视化真实标签
    plot_projection(proj_sem_label, color_sem_dict, label_name_dict, './test2_label.png')
    # plot_pc(points,labels, color_sem_dict, label_name_dict, './test1_label_pc.png')
    # 可视化预测标签
    plot_projection(pred_proj_sem_label, color_sem_dict, label_name_dict, './test2_pred.png')
    # plot_pc(points,preds, color_sem_dict, label_name_dict, './test1_pred_pc.png')
    
