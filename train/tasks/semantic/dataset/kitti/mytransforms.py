import random
import math
import tasks.semantic.dataset.kitti.transform as T

# axis of SemanticKITTI:
# X: forward
# Y: left
# Z: up

class RandomRotation(object):
    def __init__(self, max_degree=180):
        self.max_degree = max_degree

    def __call__(self, points):
        angle = random.uniform(-self.max_degree, self.max_degree) / 180 * math.pi
        cos, sin = math.cos(angle), math.sin(angle)

        x = points[:, 0] * cos - points[:, 1] * sin
        y = points[:, 1] * cos + points[:, 0] * sin

        points[:, 0] = x
        points[:, 1] = y
        return points

class RandomScale(object):
    def __init__(self, scale_range=(0.95, 1.05)):
        self.scale_range = scale_range

    def __call__(self, points):
        scale = random.uniform(*self.scale_range)
        points[:, :3] *= scale
        return points

class RandomNoise(object):
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, points):
        noise = np.random.randn(*points[:, :3].shape) * self.std
        points[:, :3] += noise
        return points      

class RandomDropPoints(object):
    # 有可能点数对不上影响projection，先不要加。
    def __init__(self, drop_ratio=0.1):
        self.drop_ratio = drop_ratio

    def __call__(self, points):
        N = points.shape[0]
        keep = np.random.rand(N) > self.drop_ratio
        return points[keep]

class RandomLeftRightFlip(object):
    def __init__(self, p=0.5):
        """
        flip points in left-right direction, remain z direction
        :param p: probability to flip
        """
        self.p = p

    def __call__(self, points):
        """
        :param points: points to be fliped
        :return: flipped points
        """
        # points[:, 1] = - points[:, 1]
        if random.random() < self.p:
            points[:, 1] = - points[:, 1]  # y->-y, x and z remain
        return points
    
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points):
        for t in self.transforms:
            points = t(points)
        return points

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string