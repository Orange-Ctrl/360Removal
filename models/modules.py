import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import BaseNetwork
from .networks import get_nonspade_norm_layer
from .networks import MySeparableBilinearDownsample as BilinearDownsample
from TMM.networks.unifuse_weight import UniFuse_weight
import torch.nn.utils.spectral_norm as spectral_norm
import torch as th
from math import pi
from math import log2
import time
import math


class CoordFillGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(lr_instance=True)
        parser.set_defaults(no_instance_dist=True)
        parser.set_defaults(hr_coor="cosine")
        return parser

    def __init__(self, opt, hr_stream=None, lr_stream=None, fast=False):
        super(CoordFillGenerator, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if (opt.no_instance_edge & opt.no_instance_dist) else 1)
        self.lr_instance = opt.lr_instance
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        self.downsampling = opt.crop_size // opt.ds_scale

        self.highres_stream = PixelQueryNet(self.downsampling, num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth,
                                               no_one_hot=opt.no_one_hot, lr_instance=opt.lr_instance,
                                               **hr_stream)
        # self.unifuse=UniFuse(num_layers=opt.num_layers,equi_h=opt.equi_h,equi_w=opt.equi_w,
        #                      pretrained=False,fusion_type=opt.fusion_type,se_in_fusion=opt.se_in_fusion
        #                      )

        self.unifuse=UniFuse_weight(num_layers=opt.num_layers,equi_h=opt.equi_h,equi_w=opt.equi_w,
                             pretrained=False,fusion_type=opt.fusion_type,se_in_fusion=opt.se_in_fusion
                             )

        num_params = self.highres_stream.num_params
        self.lowres_stream = ParaGenNet(num_params, scale_injection=opt.scale_injection)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def get_lowres(self, im):
        """Creates a lowres version of the input."""
        device = self.use_gpu()
        if(self.learned_ds_factor != self.downsampling):
            myds = BilinearDownsample(int(self.downsampling//self.learned_ds_factor), self.num_inputs,device)
            return myds(im)
        else:
            return im

    def forward(self, highres):
        lowres = self.get_lowres(highres)
        lr_features = self.lowres_stream(lowres)

        # uni_features = self.uniFuse(normals)
        # highres_with_normals=torch.cat((highres,uni_features),dim=1)

        output = self.highres_stream(highres, lr_features)
        # output = self.highres_stream(highres_with_normals, lr_features)
        return output, lr_features#, lowres


def _get_coords(bs, h, w, device, ds):
    """Creates the position encoding for the pixel-wise MLPs"""
    x = th.arange(0, w).float()
    y = th.arange(0, h).float()
    scale = 7 / 8
    x_cos = th.remainder(x, ds).float() / ds
    x_sin = th.remainder(x, ds).float() / ds
    y_cos = th.remainder(y, ds).float() / ds
    y_sin = th.remainder(y, ds).float() / ds
    x_cos = x_cos / (max(x_cos) / scale)
    x_sin = x_sin / (max(x_sin) / scale)
    y_cos = x_cos / (max(y_cos) / scale)
    y_sin = x_cos / (max(y_sin) / scale)
    xcos = th.cos((2 * pi * x_cos).float())
    xsin = th.sin((2 * pi * x_sin).float())
    ycos = th.cos((2 * pi * y_cos).float())
    ysin = th.sin((2 * pi * y_sin).float())
    xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
    xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
    ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
    ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
    coords = th.cat([xcos, xsin, ycos, ysin], 1).to(device)
    return coords.to(device)

# def _get_coords(bs, h, w, device, ds):
#     x = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w).repeat(bs, 1, h, 1)
#     y = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).repeat(bs, 1, 1, w)
#     coords=torch.cat([x, x, y, y], 1)
#     return coords.to(device)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class ParaGenNet(th.nn.Module):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, num_out, scale_injection=False):
        super(ParaGenNet, self).__init__()

        self.num_out = num_out
        self.scale_injection = scale_injection

        ngf = 64
        if self.scale_injection:
            self.out_para = nn.Sequential(
                th.nn.Linear(ngf * 8 + 1, self.num_out)
            )
        else:
            self.out_para = nn.Sequential(
                th.nn.Linear(ngf * 8, self.num_out)
            )

    def forward(self, model, x, x_hr):
        structure = model(x)
        if self.scale_injection:
            scale = (torch.ones(x_hr.size(0), 1, 1, 1) * (structure.size(3) / x_hr.size(3))) \
                    .to(structure.device)
            scale = scale.repeat(1, structure.size(2), structure.size(3), 1)
            structure = torch.cat([structure.permute(0, 2, 3, 1), scale], dim=-1)
            para = self.out_para(structure).permute(0, 3, 1, 2)
        else:
            para = self.out_para(structure.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return para


    def mask_predict(self, model, x, x_hr, mask):
        """
        掩码预测的参数生成方法：从低分辨率特征生成MLP参数
        
        Args:
            model: FFCResNetGenerator模型，用于生成低分辨率特征
            x: 低分辨率输入 [bs, channels, h_lr, w_lr]
            x_hr: 高分辨率输入，用于计算尺度信息
            mask: 掩码 [bs, channels, h, w]
        """
        # 步骤1: 通过绿色模型生成低分辨率特征 F''
        structure = model(x)  # FFCResNetGenerator输出
        
        # 步骤2: 注入目标分辨率信息 r (如果启用scale_injection)
        if self.scale_injection:
            # 计算尺度比例: 低分辨率宽度 / 高分辨率宽度
            scale = (torch.ones(x_hr.size(0), 1, 1, 1) * (structure.size(3) / x_hr.size(3))) \
                .to(structure.device)
            # 扩展到所有空间位置
            scale = scale.repeat(1, structure.size(2), structure.size(3), 1)
            # 拼接特征和尺度信息: cat[Em, r]
            structure = torch.cat([structure.permute(0, 2, 3, 1), scale], dim=-1)
        else:
            # 不注入尺度信息，只调整维度顺序
            structure = structure.permute(0, 2, 3, 1)

        # 步骤3: 获取特征尺寸信息
        bs, h, w, c = structure.size()  # 低分辨率特征尺寸
        k = mask.size(2) // h  # 下采样因子
        
        # 步骤4: 将高分辨率掩码下采样到低分辨率
        # 使用unfold将掩码分割成k×k的tiles
        mask = mask.unfold(2, k, k).unfold(3, k, k)
        # 重塑为 [批次, 低分辨率高度, 低分辨率宽度, k²]
        mask = mask.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h, w, int(k * k))
        
        # 步骤5: 计算低分辨率掩码平均值
        # 对每个tile内的掩码进行平均，得到低分辨率掩码
        lr_mask = torch.mean(mask, dim=-1).view(h * w)
        
        # 步骤6: 重塑特征并选择掩码区域的特征
        structure = structure.view(bs, h * w, c)  # 重塑为 [批次, 像素数, 通道数]
        # 找到非掩码区域的索引 (1-lr_mask表示非掩码区域)
        index = torch.nonzero(1 - lr_mask).squeeze(1)
        # 只保留非掩码区域的特征 Em
        structure = structure[:, index, :]
        
        # 步骤7: 通过线性映射函数f生成MLP参数 фm = f(cat[Em, r])
        para = self.out_para(structure).permute(0, 2, 1)
        
        return para, mask
    
class PixelQueryNet(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling,
                 num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine",
                 no_one_hot=False, lr_instance=False):
        super(PixelQueryNet, self).__init__()

        self.lr_instance = lr_instance
        self.downsampling = downsampling
        self.num_inputs = num_inputs - (1 if self.lr_instance else 0)
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.no_one_hot = no_one_hot
        self.channels = []
        self._set_channels()

        self.num_params = 0
        self.splits = {}
        self._set_num_params()

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling

    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        in_ch = in_ch + int(4)
        """Compute and store the hr-stream layer dimensions."""
        #in_ch = self.num_inputs + 3  # self.num_inputs 是原始 highres 的通道数
        # 然后根据您的具体逻辑继续设置 channels
        self.channels = [in_ch] + [self.width for _ in range(self.depth - 1)] + [self.num_outputs]
        self.channels = [in_ch]
        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)

    def _set_num_params(self):
        nparams = 0
        self.splits = {
            "biases": [],
            "weights": [],
        }

        # # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1]
            nparams = nparams + nco  # FC biases
            self.splits["biases"].append((idx, idx + nco))
            idx = idx + nco

            nparams = nparams + nci * nco  # FC weights
            self.splits["weights"].append((idx, idx + nco * nci))
            idx = idx + nco * nci

        self.num_params = nparams

    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]

    def forward(self, highres, lr_params,pred_norm):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"
        # # 确保法线图的空间维度与 highres 匹配
        assert pred_norm.shape[2:] == highres.shape[2:], "预测法线图的空间维度必须与高分辨率输入匹配"
        # 合并 highres 输入和预测法线图
        highres_with_normals = torch.cat([highres, pred_norm], dim=1)
        if self.lr_instance:
            #highres = highres[:, :-1, :, :]
            highres_with_normals = highres_with_normals[:, :-1, :, :]
        # Fetch sizes
        bs, _, h, w = highres_with_normals.shape
        bs, _, h_lr, w_lr = lr_params.shape
        k = h // h_lr
        self.xy_coords = _get_coords(1, h, w, highres_with_normals.device, h // h_lr)

        highres_with_normals = torch.repeat_interleave(self.xy_coords, repeats=bs, dim=0)
        nci = highres_with_normals.shape[1]

        tiles = highres_with_normals.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)
        out = tiles
        num_layers = len(self.channels) - 1

        # if self.lr_instance:
        #     highres = highres[:, :-1, :, :]

        # # Fetch sizes
        # bs, _, h, w = highres.shape
        # bs, _, h_lr, w_lr = lr_params.shape
        # k = h // h_lr

        # self.xy_coords = _get_coords(1, h, w, highres.device, h // h_lr)

        # highres = torch.repeat_interleave(self.xy_coords, repeats=bs, dim=0)

        # # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # # with channels last (for matmul)
        # # all pixels within a tile of kxk are processed by the same MLPs parameters
        # nci = highres.shape[1]

        # tiles = highres.unfold(2, k, k).unfold(3, k, k)
        # tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
        #     bs, h_lr, w_lr, int(k * k), nci)
        # out = tiles
        # num_layers = len(self.channels) - 1

        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]


            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)

            out = th.matmul(out, w_) + b_

            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01)
            else:
                out = torch.tanh(out)
        #
        # reorder the tiles in their correct position, and put channels first
        out = out.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out

    def mask_predict(self, highres, lr_params, hr_mask, lr_mask):
        """
        掩码预测方法：通过坐标驱动的像素级MLP网络生成掩码区域的像素值
        
        Args:
            highres: 高分辨率输入图像 [bs, channels, h, w]
            lr_params: 低分辨率生成的MLP参数 [bs, num_params, h_lr, w_lr]
            hr_mask: 高分辨率掩码 [bs, channels, h, w]
            lr_mask: 低分辨率掩码 [bs, channels, h_lr, w_lr]
        """
        # 检查参数维度是否正确
        assert lr_params.shape[1] == self.num_params, "incorrect input params"

        # 如果启用实例分割，去掉最后一个通道
        if self.lr_instance:
            highres = highres[:, :-1, :, :]

        # 获取图像尺寸信息
        bs, _, h, w = highres.shape  # 高分辨率尺寸
        bs, h_lr, w_lr, _ = lr_mask.shape  # 低分辨率尺寸
        k = h // h_lr  # 下采样因子，用于tile分割

        # 步骤1: 生成所有像素位置的坐标编码
        # 为每个像素位置生成唯一的坐标编码 [xcos, xsin, ycos, ysin]
        self.xy_coords = _get_coords(1, h, w, highres.device, h // h_lr)
        # 扩展到批次维度
        pe = torch.repeat_interleave(self.xy_coords, repeats=bs, dim=0)
        
        # 步骤2: 将坐标编码分割成k×k的tiles
        # 每个tile内的像素共享相同的MLP参数，实现空间自适应
        nci = pe.shape[1]  # 坐标编码的通道数
        # 将图像分割成k×k的tiles，便于参数共享
        tiles = pe.unfold(2, k, k).unfold(3, k, k)
        # 重塑为 [批次, 低分辨率高度, 低分辨率宽度, k², 通道数]
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)

        # 步骤3: 计算低分辨率掩码并选择需要处理的区域
        # 对掩码进行平均，得到低分辨率掩码
        mask = torch.mean(lr_mask, dim=-1).view(h_lr * w_lr)
        # 找到非掩码区域的索引 (1-mask表示非掩码区域)
        index = torch.nonzero(1 - mask).squeeze(1)
        out = tiles
        num_layers = len(self.channels) - 1

        # 步骤4: 只对非掩码区域进行MLP推理
        # 选择需要处理的tiles
        out = out.view(bs, h_lr * w_lr, int(k * k), nci)[:, index, :, :]
        num = out.size(1)  # 需要处理的区域数量

        # 步骤5: 多层MLP网络推理
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]  # 下一层的通道数

            # 从低分辨率参数中选择当前层的权重和偏置
            bstart, bstop = self._get_bias_indices(idx)   # 偏置索引
            wstart, wstop = self._get_weight_indices(idx) # 权重索引

            w_ = lr_params[:, wstart:wstop]  # 权重参数
            b_ = lr_params[:, bstart:bstop]  # 偏置参数

            # 重塑参数维度以匹配当前处理的区域
            w_ = w_.permute(0, 2, 1).view(bs, num, nci, nco)
            b_ = b_.permute(0, 2, 1).view(bs, num, 1, nco)

            # MLP计算: out = out * w_ + b_
            out = th.matmul(out, w_) + b_

            # 应用激活函数
            # 中间层使用LeakyReLU，最后一层使用tanh
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01)
            else:
                out = torch.tanh(out)

        # 步骤6: 将高分辨率图像也分割成tiles，准备结果映射
        highres = highres.unfold(2, k, k).unfold(3, k, k)
        highres = highres.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), 3).view(bs, h_lr * w_lr, int(k * k), 3)

        # 步骤7: 将生成的像素值映射回正确的位置
        # 只更新非掩码区域的像素值，掩码区域保持原值
        highres[:, index, :, :] = out
        
        # 步骤8: 将tiles重新组合成完整的高分辨率图像
        out = highres.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out