# --------------------------------------------------------
# Modified by Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# whether to use Repeated Augmentations
_C.DATA.REPEATED_AUG = False

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'VIG'
# Model name
_C.MODEL.NAME = 'VIG_tiny_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.0
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# MMpretrain models for test
_C.MODEL.MMCKPT = False


# ViG parameters
_C.MODEL.VIG = CN()
_C.MODEL.VIG.IMG_SIZE = 224
_C.MODEL.VIG.PATCH_SIZE = 16
_C.MODEL.VIG.STRIDE = 16
_C.MODEL.VIG.EMBED_DIM = 192
_C.MODEL.VIG.DEPTH = 12
_C.MODEL.VIG.NUM_HEADS = 3
_C.MODEL.VIG.MLP_RATIO = 4
_C.MODEL.VIG.QKV_BIAS = True
_C.MODEL.VIG.IN_CHANS = 3
_C.MODEL.VIG.IF_CLS_TOKEN = True
_C.MODEL.VIG.USE_MIDDLE_CLS_TOKEN = False
_C.MODEL.VIG.CLASSIFICATION_MODE = "avgpool"
_C.MODEL.VIG.IF_ABS_POS_EMBED = True
_C.MODEL.VIG.IF_RC_PE = False
_C.MODEL.VIG.ATTN_MODEL = "fused_chunk"
_C.MODEL.VIG.HIDDEN_ACT = "swish"
_C.MODEL.VIG.SCAN_MODE = "default"
_C.MODEL.VIG.USE_MSG = False
_C.MODEL.VIG.USE_DIRPE = False
_C.MODEL.VIG.USE_OUT_ACT = False
_C.MODEL.VIG.USE_OUT_GATE = True
_C.MODEL.VIG.USE_CACHE = False
_C.MODEL.VIG.USE_SWIGLU = True
_C.MODEL.VIG.USE_LOWER_BOUND=False
_C.MODEL.VIG.USE_ACT_IN_CONV=True
_C.MODEL.VIG.USE_BIAS_IN_PATCH=True
_C.MODEL.VIG.USE_ACT_IN_PATCH=False
_C.MODEL.VIG.USE_BIAS_IN_DWCONV=False
_C.MODEL.VIG.ROPE_MODE='none'
_C.MODEL.VIG.EXPAND_K = 0.5
_C.MODEL.VIG.EXPAND_V = 1.0
_C.MODEL.VIG.STATE_PASS_MODE = 'default' # ['channel','batch']
_C.MODEL.VIG.PATCH_EMBED_VERSION = 'v1'
_C.MODEL.VIG.INIT_VALUES = None
_C.MODEL.VIG.TOKEN_SHIFT=CN()
_C.MODEL.VIG.TOKEN_SHIFT.SHIFT_MODE = "none"
_C.MODEL.VIG.TOKEN_SHIFT.CHANNEL_GAMMA = 0.25
_C.MODEL.VIG.TOKEN_SHIFT.SHIFT_PIXEL = 1
# bid 
_C.MODEL.VIG.IF_BIDIRECTIONAL = True


# Hier VIG parameters
_C.MODEL.HIERVIG = CN()
_C.MODEL.HIERVIG.PATCH_SIZE = 4
_C.MODEL.HIERVIG.IN_CHANS = 3
_C.MODEL.HIERVIG.DEPTHS = [2, 2, 9, 2]
_C.MODEL.HIERVIG.NUM_HEADS = [ 3, 6, 12, 24 ]
_C.MODEL.HIERVIG.EMBED_DIM = 96
_C.MODEL.HIERVIG.ATTN_MODELS = ["fused_chunk","fused_chunk","fused_recurrent","fused_recurrent"]
_C.MODEL.HIERVIG.EXPAND_K = 0.5
_C.MODEL.HIERVIG.EXPAND_V = 1.0
_C.MODEL.HIERVIG.HIDDEN_ACT = "swish"
_C.MODEL.HIERVIG.USE_OUT_ACT = False
_C.MODEL.HIERVIG.USE_ACT_IN_CONV = True
_C.MODEL.HIERVIG.ROPE_MODE = 'none'
_C.MODEL.HIERVIG.MLP_RATIO = 4.0
_C.MODEL.HIERVIG.MLP_ACT_LAYER = "gelu"
_C.MODEL.HIERVIG.MLP_DROP_RATE = 0.0
_C.MODEL.HIERVIG.PATCH_NORM = True
_C.MODEL.HIERVIG.NORM_LAYER = "ln"
_C.MODEL.HIERVIG.DOWNSAMPLE = "v2"
_C.MODEL.HIERVIG.PATCHEMBED = "v2"
_C.MODEL.HIERVIG.GMLP = False
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# loss scaler setting
_C.TRAIN.LOSS_SCALER = CN()
_C.TRAIN.LOSS_SCALER.USE_SCALE = True
# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
_C.AMP_DTYPE = 'fp16'
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Test traincost only, overwritten by command line argument
_C.TRAINCOST_MODE = False
# for acceleration
_C.FUSED_LAYERNORM = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('num_workers'):
        config.DATA.NUM_WORKERS = args.num_workers
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('dataset'):
        config.DATA.DATASET = args.dataset
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('repeated_aug'):
        config.DATA.REPEATED_AUG = args.repeated_aug
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True
    if _check_args('traincost'):
        config.TRAINCOST_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
