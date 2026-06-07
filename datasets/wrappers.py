import functools
import random
import math
import os
import sys

from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
# 将TMM目录的父目录添加到sys.path中
script_dir = os.path.dirname(__file__)  # 获取当前脚本的目录
project_dir = os.path.dirname(script_dir)  # 获取项目根目录
tmm_dir = os.path.join(project_dir, 'TMM')  # 获取TMM目录的路径
if tmm_dir not in sys.path:
    sys.path.append(tmm_dir)

from util import Equirec2Cube

import os

def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


def get_coord(shape):
    ranges = None
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    return ret


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

        self.e2c=Equirec2Cube(self.inp_size,self.inp_size,self.inp_size//2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask, norm, vps = self.dataset[[idx, idx, idx , idx]]
        name=os.path.basename(self.dataset.dataset_2.files[idx])

        size = self.inp_size
        norm = np.transpose(norm, (0, 2, 1))
        cube_rgb = self.e2c.run(img)
        mask = to_mask(mask)
        cube_mask = self.e2c.run(mask)

        cube_mask[cube_mask > 0] = 1
        cube_mask = 1 - cube_mask

        img = resize_fn(img, (size, size))
        mask = resize_fn(mask, (size, size))

        
        mask[mask > 0] = 1
        mask = 1 - mask
        
        #norm=resize_fn(norm,(size,size))
        # Convert pixel values from [0, 255] to [0, 1]
        normal = norm / 255.0
        # Compute the magnitude of the normal vectors
        norms = np.linalg.norm(normal, axis=-1, keepdims=True)
        # Normalize the normal vectors (避免除零)
        norms = np.where(norms == 0, 1, norms)  # 将零值替换为1
        norm = norm / norms

        vps = vps.reshape([3, 3])
        vps = np.vstack([vps, -vps])


        return {
            'inp': img,
            'gt_rgb': img,
            'mask': mask,
            'cube_rgb':cube_rgb,
            'cube_mask':cube_mask,
            'gt_norm':norm,
            'vps':torch.from_numpy(vps),
            'name':name
        }

@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment

        self.count = 0
        self.scale = 0
        self.e2c=Equirec2Cube(self.size_min,self.size_max,self.size_min//2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask, norm, vps = self.dataset[[idx, idx, idx, idx]]
        name=os.path.basename(self.dataset.dataset_2.files[idx])

        size_max = self.size_max
        size_min = self.size_min
        norm = np.transpose(norm, (0, 2, 1))
        mask = to_mask(mask)
        cube_rgb = self.e2c.run(img)
        cube_mask=self.e2c.run(mask)

        img = resize_fn(img, (size_max, size_min))
        mask = resize_fn(mask, (size_max, size_min))
        #norm=resize_fn(norm,(size_max,size_min))
        
        mask[mask > 0] = 1
        mask = 1 - mask

        
        cube_mask[cube_mask > 0] = 1
        cube_mask = 1 - cube_mask

        if self.augment:
            if random.random() < 0.5:
                img = img.flip(-1)
                mask = mask.flip(-1)
        
        
        # Convert pixel values from [0, 255] to [0, 1]
        normal = norm / 255.0
        # Compute the magnitude of the normal vectors
        norms = np.linalg.norm(normal, axis=-1, keepdims=True)
        # Normalize the normal vectors (避免除零)
        norms = np.where(norms == 0, 1, norms)  # 将零值替换为1
        norm = norm / norms

        vps = vps.reshape([3, 3])
        vps = np.vstack([vps, -vps])

        return {
            'inp': img,
            'gt_rgb': img,
            'mask': mask,
            'cube_rgb':cube_rgb,
            'cube_mask':cube_mask,
            'gt_norm':norm,
            'vps':torch.from_numpy(vps),
            'name':name
        }