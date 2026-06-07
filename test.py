import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from PIL import Image
from torchvision import transforms
from torchsummary import summary
import numpy as np
import cv2
from torchvision.utils import save_image
import thop
import torch.nn.functional as F
def calculate_params_flops_inference_time(model, device):
    # 设置模型为评估模式
    model.eval()
    model.to(device)  # 确保模型在正确的设备上

    # 创建输入张量，确保它们在相同设备上
    Unused  = torch.randn([16, 3, 512, 512]).to(device)
    img = torch.randn([16, 3, 512, 512]).to(device)
    mask = torch.randn([16, 1, 512, 512]).to(device)
    cube_rgb = torch.randn([16, 3, 256, 1536]).to(device)
    cube_mask= torch.randn([16, 3, 256, 1536]).to(device)
    gt_norm = torch.randn([16, 3, 1024, 512]).to(device)
    
    vps = torch.randn([16, 6, 3]).to(device)
    inputs = ([Unused, img,mask,cube_rgb,cube_mask,gt_norm,vps],)

    # 计算参数数量
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {param_count:,}")

    # 使用 thoop 计算 FLOPs
    # try:
    #     # 计算 FLOPs，thoop 会自动处理设备问题
    #     flops, _ = thop.profile(model, inputs)  # 获取第一个返回值：FLOPs
    #     print(f"Model FLOPs: {flops / 1e9:.3f} GFLOPs")
    # except Exception as e:
    #     print(f"FLOPs calculation failed due to: {e}")
    #     flops = None
    # 测量推理时间
    torch.cuda.synchronize() # 确保之前的 CUDA 操作完成
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_time = 0
    for _ in range(10):
        start.record()
        with torch.no_grad():
            # `diffsal` 模型使用 `model.test` 方法进行推理
            _ = model(*inputs)
            end.record()
            torch.cuda.synchronize() # 等待事件完成
            total_time += start.elapsed_time(end)

    avg_inference_time = total_time / 10 # 计算平均时间（毫秒）
    print(f"Average Inference Time: {avg_inference_time:.3f} ms")
    # return flops




def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))

def tensor2PIL(tensor):
    # img = tensor.cpu().clone()
    # img = img.squeeze(0)
    # img = unloader(img)
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt_rgb']
    gt_rgb_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_rgb_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    t = data_norm['gt_norm']
    gt_norm_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    gt_norm_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    # val_res = utils.Averager()
    val_psnr = utils.Averager()
    val_ssim = utils.Averager()
    val_l1 = utils.Averager()
    val_mae = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')


    for batch in pbar:
        for k, v in batch.items():
            if k!='name':
                batch[k] = v.cuda()

            if k == 'name':
                batch[k]=v

        inp = (batch['inp'] - inp_sub) / inp_div
        gt = (batch['gt_rgb'] - gt_rgb_sub) / gt_rgb_div
        gt_norm = (batch['gt_norm'] - gt_norm_sub) / gt_norm_div
        vps=(batch['vps'])
        cube_rgb=batch["cube_rgb"]
        cube_mask=(batch['cube_mask'])

        if eval_bsize is None:
            with torch.no_grad():
                # pred = model.encoder.mask_predict([inp, batch['mask']])
                pred = model.encoder([inp, gt, batch['mask'],cube_rgb,cube_mask,gt_norm,vps])
        else:
            pred = batched_predict(model, inp, batch['coord'], eval_bsize)

        pred = (pred * (1 - batch['mask']) + gt * batch['mask']) * gt_rgb_div + gt_rgb_sub
        pred.clamp_(0, 1)
        
        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()



        mae = F.l1_loss(pred, batch['gt_rgb'], reduction='mean')
        psnr, ssim, l1 = metric_fn(model, pred, batch['gt_rgb'])
        val_psnr.add(psnr.item(), inp.shape[0])
        val_ssim.add(ssim.item(), inp.shape[0])
        val_l1.add(l1.item(), inp.shape[0])

        val_mae.add(mae.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val psnr{:.4f}'.format(val_psnr.item()))
            pbar.set_description('val ssim{:.4f}'.format(val_ssim.item()))
            pbar.set_description('val lpips{:.4f}'.format(val_l1.item()))
            pbar.set_description('val mae{:.4f}'.format(val_mae.item()))

        # save_path = "./param_498/output/"
        # gt_save_path = "./param_498/gt/"
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        # if not os.path.exists(gt_save_path):
        #     os.mkdir(gt_save_path)
        # a = 0
        # for item in pred:
            
        #     out_i=item
        #     gt_i=batch['gt_rgb'][a]
            
        #     pred_path=os.path.join(save_path,batch['name'][a])
        #     gt_path=os.path.join(gt_save_path,batch['name'][a])
            
        #     out_i = resize_fn(out_i, (2151, 4302))
        #     gt_i = resize_fn(gt_i, (2151, 4302))
            
        #     out_i=out_i.cuda()
        #     gt_i=gt_i.cuda()
            
        #     save_image(out_i,pred_path)
        #     save_image(gt_i,gt_path)
        #     a+=1
    return val_psnr.item(), val_ssim.item(), val_l1.item(), val_mae.item()


from collections import OrderedDict
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True,drop_last=True)

    model = models.make(config['model']).cuda()
    model.encoder.load_state_dict(torch.load(args.model, map_location='cuda:0'))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    calculate_params_flops_inference_time(model=model.encoder,device=model.device)

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)

    print('result psnr: {:.6f}'.format(res[0]))
    print('result ssim: {:.6f}'.format(res[1]))
    print('result lpips: {:.6f}'.format(res[2]))
    print('result mae: {:.6f}'.format(res[3]))
