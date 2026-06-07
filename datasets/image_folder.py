import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('image-folder')
class ImageFolder(Dataset):
    
    def __init__(self, path,split_file=None, split_key=None, first_k=None,
                 repeat=1, cache=False,):
        self.repeat = repeat
        self.cache = False

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filepath, dirnames, filenames in os.walk(path):
            filenames = sorted(os.listdir(path))
            for filename in filenames:
                if self.cache:
                    self.files.append(
                        transforms.ToTensor()(Image.open(os.path.join(filepath, filename)).convert('RGB')))
                else:
                    self.files.append(os.path.join(filepath, filename))

        if first_k is not None:
            self.files = self.files[:first_k]

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        if self.cache:
            return x
        else:
            return transforms.ToTensor()(Image.open(x).convert('RGB'))
    

class NPYDataset(Dataset):
    def __init__(self, path, split_file=None, split_key=None, first_k=None, repeat=1, cache=False):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = [file for file in sorted(os.listdir(path)) if file.endswith('.npy')]
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
                filenames = [filename for filename in filenames if filename.endswith('.npy')]
                
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file_path = os.path.join(path, filename)
            if self.cache:
                # 如果启用缓存，预先加载所有.npy文件到内存中
                self.files.append(np.load(file_path))
            else:
                # 否则，只存储文件路径
                self.files.append(file_path)

        if first_k is not None:
            self.files = self.files[:first_k]

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.files)
        if self.cache:
            # 如果启用了缓存，直接返回数据
            return self.files[idx]
        else:
            # 如果没有启用缓存，从路径加载.npy文件
            return np.load(self.files[idx])

@register('paired-image-folders')
class PairedImageFolders(Dataset):
    def __init__(self, root_path_1, root_path_2, root_path_3=None, vps_path=None, **kwargs):
        self.dataset_1 = ImageFolder(path=root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(path=root_path_2, **kwargs)
        
        # 处理 root_path_3：检查是否为 None 或字符串 'None'
        if root_path_3 is not None and root_path_3 != 'None' and root_path_3 != '':
            self.dataset_3 = ImageFolder(path=root_path_3, **kwargs)
        else:
            self.dataset_3 = None
            
        # 处理 vps_path：检查是否为 None 或字符串 'None'  
        if vps_path is not None and vps_path != 'None' and vps_path != '':
            self.dataset_4 = NPYDataset(path=vps_path, **kwargs)
        else:
            self.dataset_4 = None

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        """
        获取数据项
        支持两种索引方式：
        1. 单个索引：所有数据集使用相同索引
        2. 索引元组：每个数据集使用对应索引
        """
        # 处理索引
        if isinstance(idx, (list, tuple)) and len(idx) >= 2:
            # 索引元组模式：为每个数据集指定不同索引
            idx1 = idx[0]  # RGB 图像索引
            idx2 = idx[1]  # Mask 图像索引
            idx3 = idx[2] if len(idx) > 2 else idx1  # Normal 图像索引（如果提供）
            idx4 = idx[3] if len(idx) > 3 else idx1  # VPS 数据索引（如果提供）
        else:
            # 单索引模式：所有数据集使用相同索引
            idx1 = idx2 = idx3 = idx4 = idx
        
        # 获取必需的数据（RGB 和 Mask）
        rgb_img = self.dataset_1[idx1]      # RGB 图像
        mask_img = self.dataset_2[idx2]     # Mask 图像
        
        # 获取可选的 Normal 数据
        if self.dataset_3 is not None:
            normal_img = self.dataset_3[idx3]
        else:
            # 如果没有 normal 数据，返回占位符张量
            normal_img = torch.zeros(3, 512, 512, dtype=torch.float32)
        
        # 获取可选的 VPS 数据
        if self.dataset_4 is not None:
            vps_data = self.dataset_4[idx4]
        else:
            # 如果没有 vps 数据，返回占位符张量 (3x3=9 个元素)
            vps_data = torch.zeros(9, dtype=torch.float32)
        
        return rgb_img, mask_img, normal_img, vps_data

# if __name__=='__main__':
#     root_path_1='/data/chenni/Data1/Data1/train/gt-ob'
#     root_path_2= '/data/chenni/Data1/Data1/train/mask-ob'
#     root_path_3= '/data/chenni/Data1/Data1/train/normal'
#     vps_path='/data/chenni/vps/structured3D/train'
#     img=PairedImageFolders(root_path_1,root_path_2,root_path_3,vps_path)