# Interactive 360° Object Removal: A Coordinate Querying Approach via Spherical Normal Constraints and Learnable Wavelet Enhancement

## Overview

This repository provides code for interactive object removal on equirectangular panoramic images. 

## Dataset

Organize the data into `train/`, `val/`, and `test/` splits. Example layout:

```
./360removal/
├── train/
├── val/
└── test/
```

Update the dataset path in your config file to point to the corresponding split before training or evaluation.

## Environment

This code was implemented with Python 3.7.12 and PyTorch 1.13.1. You can install all the requirements via:

```bash
pip install -r requirements.txt
```

## Train

Single GPU training: 

```bash
python train.py --config [CONFIG_PATH]
```

## Test

```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Models

Please find the pre-trained models ./save

## Acknowledgements

 code borrows heavily [LAMA](https://github.com/advimman/lama),[CoordFill](https://github.com/NiFangBaAGe/CoordFill). [or-nerf] 