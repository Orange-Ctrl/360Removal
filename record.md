把FFC模块的输入修改为六面投影：
result psnr: 38.152239        
result ssim: 0.982803
result lpips: 0.017163


之前的：
result psnr: 39.97
result ssim: 0.984
result lpips: 0.014

5e-5:
lamda10_wavelet1_6blocks_tanh_5e-5/_train_SD/encoder-epoch-best.pth
result psnr: 39.073354
result ssim: 0.983156
result lpips: 0.014277

lamda10_wavelet1_6blocks_tanh_5e-5_300_norelu/_train_SD/encoder-epoch-best.pth
params:
result psnr: 39.312206
result ssim: 0.983453
result lpips: 0.014563

ABLATION/ablation_with_FFTBlock
model: #params=94.1M
result psnr: 39.832941
result ssim: 0.983849
result lpips: 0.012795

ABLATION/ablation_with_FFT2
model: #params=60.3M
result lpips: 0.012796result psnr: 39.612088
result ssim: 0.983748
result lpips: 0.012796

  ours with FFT	94.1M		39.833	0.984	0.013
  ours with FFT2	60.3M		39.612	0.984	0.013
