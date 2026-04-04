import numpy as np
import random

class RangeCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, proj, mask, label):
        for t in self.transforms:
            proj, mask, label = t(proj, mask, label)
        return proj, mask, label


class RandomHorizontalRoll:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, proj, mask, label):
        if random.random() < self.p:
            W = proj.shape[2]   # proj: C,H,W
            shift = np.random.randint(0, W)
            proj = np.roll(proj, shift, axis=2)
            mask = np.roll(mask, shift, axis=1)
            label = np.roll(label, shift, axis=1)
        return proj, mask, label


class RandomColumnDrop:
    def __init__(self, p=0.5, drop_ratio=0.05, ignore_index=0):
        self.p = p
        self.drop_ratio = drop_ratio
        self.ignore_index = ignore_index

    def __call__(self, proj, mask, label):
        if random.random() < self.p:
            W = proj.shape[2]
            num_drop = max(1, int(W * self.drop_ratio))
            cols = np.random.choice(W, num_drop, replace=False)
            proj[:, :, cols] = 0
            mask[:, cols] = 0
            label[:, cols] = self.ignore_index
        return proj, mask, label


class RandomGaussianNoise:
    def __init__(self, p=0.5, std=0.01, apply_channels=None):
        self.p = p
        self.std = std
        self.apply_channels = apply_channels  # e.g. [0,1,2,3,4]

    def __call__(self, proj, mask, label):
        if random.random() < self.p:
            if self.apply_channels is None:
                noise = np.random.randn(*proj.shape) * self.std
                proj = proj + noise
            else:
                for c in self.apply_channels:
                    proj[c] = proj[c] + np.random.randn(*proj[c].shape) * self.std
        return proj, mask, label


class RandomCutout:
    def __init__(self, p=0.5, max_h=8, max_w=64, ignore_index=0):
        self.p = p
        self.max_h = max_h
        self.max_w = max_w
        self.ignore_index = ignore_index

    def __call__(self, proj, mask, label):
        if random.random() < self.p:
            _, H, W = proj.shape
            cut_h = np.random.randint(1, self.max_h + 1)
            cut_w = np.random.randint(1, self.max_w + 1)
            y = np.random.randint(0, H - cut_h + 1)
            x = np.random.randint(0, W - cut_w + 1)

            proj[:, y:y+cut_h, x:x+cut_w] = 0
            mask[y:y+cut_h, x:x+cut_w] = 0
            label[y:y+cut_h, x:x+cut_w] = self.ignore_index
        return proj, mask, label