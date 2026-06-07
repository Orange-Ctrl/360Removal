from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms

from .util import Equirec2Cube


def read_list(list_file):
    rgb_normal_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            paths = line.strip().split()
            if len(paths) >= 2:  # Assuming there are at least two paths per line
                rgb_normal_list.append((paths[0], paths[1]))  # Append as a tuple
            else:
                print(f"Invalid line found in file: {line}")
    return rgb_normal_list

class Data(data.Dataset):
    """The Structured3D Dataset"""

    def __init__(self, root_dir, list_file,vps_path, height=512, width=1024,transform=None, rescaled=False,
                 disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False,
                 is_training=False
                 ):
        """
        Args:
            root_dir (string): Directory of the Structured3D Dataset.
            list_file (string): Path to the txt file contain the list of image and normal files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir

        print(f'root:{root_dir}\nlist:{list_file}')
        self.rgb_normal_list = read_list(list_file)

        self.w = width
        self.h = height
        self.vps_path = vps_path
        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        self.is_training = is_training

        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
                
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_normal_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        vps_filename = self.rgb_normal_list[idx][0].replace('.png', '.npy')
        filename=os.path.basename(vps_filename)
        vps_filepath = os.path.join(self.vps_path,filename)
        

        # Load the vanishing points .npy file
        vps = np.load(vps_filepath).astype(np.float32)
        vps = vps.reshape([3, 3])
        vps = np.vstack([vps, -vps])

        rgb_path,normal_path=self.rgb_normal_list[idx]
        rgb_name = os.path.join(self.root_dir, rgb_path)
        rgb = cv2.imread(rgb_name)
        if rgb is None:
            raise ValueError(f"Image not loaded correctly.rgb_name:{rgb_name}")
        else:     
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        normal_name = os.path.join(self.root_dir, normal_path)
        normal_map = cv2.imread(normal_name, -1)
        normal_map = cv2.resize(normal_map, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        normal_map = normal_map.astype(np.float32)
        # Convert pixel values from [0, 255] to [0, 1]
        normal_map = normal_map / 255.0

        # Compute the magnitude of the normal vectors
        norms = np.linalg.norm(normal_map, axis=-1, keepdims=True)

        # Normalize the normal vectors
        normal_map = normal_map / norms

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        cube_rgb = self.e2c.run(rgb)
        cube_aug_rgb = self.e2c.run(aug_rgb)

        rgb = self.to_tensor(rgb.copy())
        cube_rgb = self.to_tensor(cube_rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())


        # Normal map tensor conversion
        normal_map = self.to_tensor(normal_map.copy())

        # Return a dictionary of inputs
        inputs['rgb'] = rgb
        inputs['cube_rgb'] = cube_rgb
        inputs['aug_rgb'] = aug_rgb
        inputs['gt_normal'] = normal_map
        inputs['vps'] = torch.from_numpy(vps)

        return inputs
