import os
from functools import partial

import torch
# plain vig
from .vig import ViG
# hierarchical vig 
from .vig_h import HierViG




def build_vig_model(config, is_pretrain=False):
    
    model_type = config.MODEL.TYPE
    if model_type in ["vig"]:
        model = ViG(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIG.PATCH_SIZE,
            stride=config.MODEL.VIG.STRIDE,
            depth=config.MODEL.VIG.DEPTH,
            num_heads=config.MODEL.VIG.NUM_HEADS,
            embed_dim=config.MODEL.VIG.EMBED_DIM,
            channels=config.MODEL.VIG.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            if_cls_token=config.MODEL.VIG.IF_CLS_TOKEN,
            use_middle_cls_token=config.MODEL.VIG.USE_MIDDLE_CLS_TOKEN,
            classification_mode=config.MODEL.VIG.CLASSIFICATION_MODE,
            if_abs_pos_embed=config.MODEL.VIG.IF_ABS_POS_EMBED,
            if_rc_pe=config.MODEL.VIG.IF_RC_PE,
            drop_rate=config.MODEL.DROP_RATE,
            attn_model=config.MODEL.VIG.ATTN_MODEL,
            hidden_act=config.MODEL.VIG.HIDDEN_ACT,
            use_out_act=config.MODEL.VIG.USE_OUT_ACT,
            use_out_gate=config.MODEL.VIG.USE_OUT_GATE,
            use_lower_bound=config.MODEL.VIG.USE_LOWER_BOUND,
            use_msg=config.MODEL.VIG.USE_MSG,
            use_swiglu=config.MODEL.VIG.USE_SWIGLU,
            use_act_in_conv=config.MODEL.VIG.USE_ACT_IN_CONV,
            rope_mode=config.MODEL.VIG.ROPE_MODE,
            expand_k=config.MODEL.VIG.EXPAND_K,
            expand_v=config.MODEL.VIG.EXPAND_V,
            use_dirpe=config.MODEL.VIG.USE_DIRPE,
            scan_mode=config.MODEL.VIG.SCAN_MODE,
            patch_embed_version=config.MODEL.VIG.PATCH_EMBED_VERSION,
            use_bias_in_patch=config.MODEL.VIG.USE_BIAS_IN_PATCH,
            use_act_in_patch=config.MODEL.VIG.USE_ACT_IN_PATCH,
            use_bias_in_dwconv=config.MODEL.VIG.USE_BIAS_IN_DWCONV,
            init_values=config.MODEL.VIG.INIT_VALUES,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
        return model
    elif model_type in ["vig_h"]:
        model = HierViG(
            patch_size=config.MODEL.HIERVIG.PATCH_SIZE, 
            in_chans=config.MODEL.HIERVIG.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.HIERVIG.DEPTHS, 
            num_heads=config.MODEL.HIERVIG.NUM_HEADS,
            dims=config.MODEL.HIERVIG.EMBED_DIM, 
            # ===================
            attn_models=config.MODEL.HIERVIG.ATTN_MODELS,
            expand_k=config.MODEL.HIERVIG.EXPAND_K,
            expand_v=config.MODEL.HIERVIG.EXPAND_V,
            hidden_act=config.MODEL.HIERVIG.HIDDEN_ACT,
            use_out_act=config.MODEL.HIERVIG.USE_OUT_ACT,
            use_act_in_conv=config.MODEL.HIERVIG.USE_ACT_IN_CONV,
            rope_mode=config.MODEL.HIERVIG.ROPE_MODE,
            # ===================
            mlp_ratio=config.MODEL.HIERVIG.MLP_RATIO,
            mlp_act_layer=config.MODEL.HIERVIG.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.HIERVIG.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.HIERVIG.PATCH_NORM,
            norm_layer=config.MODEL.HIERVIG.NORM_LAYER,
            downsample_version=config.MODEL.HIERVIG.DOWNSAMPLE,
            patchembed_version=config.MODEL.HIERVIG.PATCHEMBED,
            gmlp=config.MODEL.HIERVIG.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        return model
    else:
        raise NotImplementedError(f"model_type {model_type} is not implemented")
    return None

def build_model(config, is_pretrain=False):
    model = build_vig_model(config, is_pretrain)
    return model




