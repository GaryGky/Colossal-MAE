# Train MAE With Colossal-AI

[![logo](/Users/kaiyuan.gan/code.nus.course/hpc-ai/ColossalAI/docs/images/Colossal-AI_logo.png)](https://www.colossalai.org/)

<div align="center">
   <h3> <a href="https://arxiv.org/abs/2110.14883"> Paper </a> | 
   <a href="https://www.colossalai.org/"> Documentation </a> | 
   <a href="https://github.com/hpcaitech/ColossalAI-Examples"> Examples </a> |   
   <a href="https://github.com/hpcaitech/ColossalAI/discussions"> Forum </a> | 
   <a href="https://medium.com/@hpcaitech"> Blog </a></h3>

   [![Build](https://github.com/hpcaitech/ColossalAI/actions/workflows/PR_CI.yml/badge.svg)](https://github.com/hpcaitech/ColossalAI/actions/workflows/PR_CI.yml)
   [![Documentation](https://readthedocs.org/projects/colossalai/badge/?version=latest)](https://colossalai.readthedocs.io/en/latest/?badge=latest)
   [![codebeat badge](https://codebeat.co/badges/bfe8f98b-5d61-4256-8ad2-ccd34d9cc156)](https://codebeat.co/projects/github-com-hpcaitech-colossalai-main)
</div>
An integrated large-scale model training system with efficient parallelization techniques.

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="300">
</p>

## SetUp

#### Install Colossal AI from source (Recomended)

```bash
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

Install and enable CUDA kernel fusion (compulsory installation when using fused optimizer)

```bash
pip install -v --no-cache-dir --global-option="--cuda_ext" .
```

You may need more details about **installation** on [Colossal-AI](https://github.com/hpcaitech/ColossalAI )

## Pre-training

To pre-train MAE with Colossal AI, firstly write an config file including running parameters.

```python
from colossalai.amp import AMP_TYPE

TOTAL_BATCH_SIZE = 4096
LR = 1.5e-4
WEIGHT_DECAY = 0.05

TENSOR_PARALLEL_SIZE = 1
TENSOR_PARALLEL_MODE = None

NUM_EPOCHS = 800
WARMUP_EPOCHS = 40

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

fp16 = dict(mode=AMP_TYPE.TORCH, )

gradient_accumulation = 2

BATCH_SIZE = TOTAL_BATCH_SIZE // gradient_accumulation

clip_grad_norm = 1.0

LOG_PATH = f"./vit_{TENSOR_PARALLEL_MODE}_imagenet1k_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LR}_{fp16['mode']}_clip_grad{clip_grad_norm}/"

MODEL="mae_vit_large_patch16"
NORM_PIX_LOSS=True
MASK_RATIO=0.75
```

- Here the effective batch size is 64 (`batch_size` per gpu) * 8 (`nodes`) * 8 (gpus per node) = 4096. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is `batch_size` (per gpu) * `nodes` * 8 (gpus per node) * `accum_iter`.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- Here we use `--norm_pix_loss` as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off `--norm_pix_loss`.

## Launch

We provide an 

## Cite us

```c
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```


