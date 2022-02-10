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

MODEL="mae_vit_base_patch16"
NORM_PIX_LOSS=True
MASK_RATIO=0.75