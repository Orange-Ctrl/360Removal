import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy import ndimage
import numpy as np
from .modules import CoordFillGenerator
from .ffc import FFCResNetGenerator

from .ffc import FFCResnetBlock, ConcatTupleLayer, FFC_BN_ACT, BasicResnetBlock,WaveletBlock

class AttFFC(nn.Module):
    
    def __init__(self, ngf):
        super(AttFFC, self).__init__()
        self.add = FFC_BN_ACT(ngf, ngf, kernel_size=3, stride=1, padding=1,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                           **{"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False})
        self.minus = FFC_BN_ACT(ngf+1, ngf, kernel_size=3, stride=1, padding=1,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                           **{"ratio_gin": 0, "ratio_gout": 0.75, "enable_lfu": False})
        self.mask = FFC_BN_ACT(ngf, 1, kernel_size=3, stride=1, padding=1,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.Sigmoid,
                           **{"ratio_gin": 0.75, "ratio_gout": 0, "enable_lfu": False})

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)

        mask, _ = self.mask((x_l, x_g))

        minus_l, minus_g = self.minus(torch.cat([x_l, x_g, mask], 1))

        add_l, add_g = self.add((x_l - minus_l, x_g - minus_g))

        x_l, x_g = x_l - minus_l + add_l, x_g - minus_g + add_g

        return x_l, x_g


class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleResBlock, self).__init__()
        self.simple_conv = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )    
    def forward(self, x):
        x = self.simple_conv(x)
        return 

from .ffc_baseline import MLPModel
class CoordFill(nn.Module):
    def __init__(self, args, name, mask_prediction=False, attffc=False,
                 scale_injection=False):
        super(CoordFill, self).__init__()
        self.args = args
        self.n_channels = args.n_channels
        self.n_classes = args.n_classes
        self.out_dim = args.n_classes
        self.in_size = 256
        self.name = name
        self.mask_prediction = mask_prediction
        self.attffc = attffc
        self.scale_injection = scale_injection
        self.pred_norm = None
        self.opt = self.get_opt()
        self.asap = CoordFillGenerator(self.opt)

       
        #self.refine = FFCResNetGenerator(4, 3, ngf=64, n_downsampling=3,
        #                          n_blocks=6, res_dilation=1, decode=False)
            
        self.refine = FFCResNetGenerator(4, 4, ngf=64, n_downsampling=3,
                                    n_blocks=6, res_dilation=1, decode=False)


    def get_opt(self):
        from yacs.config import CfgNode as CN
        opt = CN()
        opt.label_nc = 0
        # opt.label_nc = 1
        opt.lr_instance = False
        opt.crop_size = 512
        opt.ds_scale = 32
        opt.aspect_ratio = 1.0
        opt.contain_dontcare_label = False
        opt.no_instance_edge = True
        opt.no_instance_dist = True
        opt.gpu_ids = 0
        opt.output_nc = 3
        opt.hr_width = 64
        opt.hr_depth = 5
        opt.scale_injection = self.scale_injection

        opt.no_one_hot = False
        opt.lr_instance = False
        opt.norm_G = 'batch'

        opt.lr_width = 256
        opt.lr_max_width = 256
        opt.lr_depth = 5
        opt.learned_ds_factor = 1
        opt.reflection_pad = False

        opt.num_layers=18
        opt.equi_h=256
        opt.equi_w=512
        opt.pretrain=False
        opt.fusion_type='cee'
        opt.se_in_fusion=True

        return opt

    def forward(self, inp):
        _,img, mask, cube_rgb,cube_mask, gt_norm, vps = inp
        # print(f"Unused variable shape: {_.shape}")
        # print(f"img shape: {img.shape}")
        # print(f"mask shape: {mask.shape}")
        # print(f"cube_rgb shape: {cube_rgb.shape}")
        # print(f"cube_mask shape: {cube_mask.shape}")
        # print(f"gt_norm shape: {gt_norm.shape}")
        # print(f"vps shape: {vps.shape}")
        hr_hole = img * mask

        #lr_img = F.interpolate(img, size=(2*self.in_size, self.in_size), mode='bilinear')
        #lr_mask = F.interpolate(mask, size=(2*self.in_size, self.in_size), mode='nearest')
        lr_img = F.interpolate(img, size=(self.in_size, self.in_size), mode='bilinear')
        lr_mask = F.interpolate(mask, size=(self.in_size, self.in_size), mode='nearest')
        lr_hole = lr_img * lr_mask

        
        uni_img = F.interpolate(img, size=(self.in_size, self.in_size*2), mode='bilinear')
        uni_mask = F.interpolate(mask, size=(self.in_size, self.in_size*2), mode='bilinear')
        uni_hole=uni_img*uni_mask

        # 从 cube_rgb 获取原始的高度和宽度
        original_height, original_width = cube_rgb.shape[-2:]

        half_height = original_height // 2
        half_width = original_width // 2
        # 使用 interpolate 调整 cube_rgb 的尺寸
        cube_rgb = F.interpolate(cube_rgb, size=(half_height, half_width), mode='bilinear', align_corners=False)
        cube_mask = F.interpolate(cube_mask, size=(half_height, half_width), mode='bilinear', align_corners=False)
        cube_hole=cube_rgb*cube_mask


        uni_feature = self.asap.unifuse(uni_hole,cube_hole)

        self.pred_norm = uni_feature['pred_normal']

        uni_feature_resized = F.interpolate(uni_feature['pred_normal'], size=(hr_hole.size(2),hr_hole.size(3)), mode='bilinear', align_corners=False)
        lr_features = self.asap.lowres_stream(self.refine, torch.cat([lr_hole, lr_mask], dim=1), hr_hole)
        # lr_features = self.asap.lowres_stream(self.refine, torch.cat([cube_hole, cube_mask], dim=1), hr_hole)
        
        output = self.asap.highres_stream(hr_hole, lr_features, uni_feature_resized)

        if self.mask_prediction:
            output = output * (1 - mask) + hr_hole

        return output
    
    def get_pred_norm(self):
        if self.pred_norm is not None:
            return self.pred_norm
        else:
            raise ValueError("pred_norm has not been computed yet.")

    def mask_predict(self, inp):
        img, mask = inp
        hr_hole = img * mask

        lr_img = F.interpolate(img, size=(self.in_size, self.in_size), mode='bilinear')
        lr_mask = F.interpolate(mask, size=(self.in_size, self.in_size), mode='nearest')
        lr_hole = lr_img * lr_mask

        lr_features, temp_mask = self.asap.lowres_stream.mask_predict(self.refine, torch.cat([lr_hole, lr_mask], dim=1), hr_hole, mask)

        output = self.asap.highres_stream.mask_predict(hr_hole, lr_features, mask, temp_mask)
        output = output * (1 - mask) + hr_hole

        return output
    def get_wavelet_loss(self):
        return self.refine.get_wavelet_loss()
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


device=torch.device('cuda')
# device=torch.device('cpu')
from models import register
from argparse import Namespace
@register('asap')
def make_unet(n_channels=3, n_classes=3, no_upsampling=False):
    args = Namespace()

    args.n_channels = n_channels
    args.n_classes = n_classes

    args.no_upsampling = no_upsampling
    return LPTN(args)