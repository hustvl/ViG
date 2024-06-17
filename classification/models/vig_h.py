import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers import DropPath, to_2tuple
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchvision.models import VisionTransformer
from fla.models import GLAConfig
from timm.models.vision_transformer import Attention
from transformers.activations import ACT2FN
from fla.modules import FusedRMSNormSwishGate, RMSNorm
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.ops.gla.chunk_fuse import FusedChunkGLAFunction,pad
from einops import rearrange
import torch.nn.functional as F
from fla.modules.activations import swiglu
from  causal_conv1d.causal_conv1d_interface import CausalConv1dFn
from fla.modules.activations import swiglu_linear
from fla.modules.rotary import RotaryEmbedding
from fla.ops.gla.recurrent_fuse import bid_fused_recurrent_gla
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # x: B N H Hc/2
    # freqs_cis:  N, H*Hc/2 or  N Hc/2
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.shape[-1] == x.shape[-1]:
        shape = [1 if i == 2 or i == 0 else d for i, d in enumerate(x.shape)]  # 1, N, 1, Hc/2
    else:
        shape = [d if i != 0 else 1 for i, d in enumerate(x.shape)] # 1, N, H, Hc/2
        # B, N, Hc/2
    return freqs_cis.view(*shape)

class RotaryEmbeddingFast2D(nn.Module):
    def __init__(self,
                 embed_dims,
                 patch_resolution,
                 theta=10000.,
                 base_size=14,
                 init_cfg=None):
        # super(RotaryEmbeddingFast, self).__init__(init_cfg=init_cfg)
        super().__init__()
        super().__init__()
        self.dim = embed_dims
        self.half_dim = embed_dims // 2
        self.patch_resolution = to_2tuple(patch_resolution)
        self.theta = theta
        H,W = self.patch_resolution
        self.base_size = base_size
        self.scale_h = base_size / H
        self.scale_w = base_size / W
        self.freqs_cis = None
        # self.register_buffer('freqs_cis', freqs_cis)
        # self.register_buffer('freqs_sin', freqs_sin)

    def compute_position_embedding(self,step:int =1,bias=0.0):
        H, W = self.patch_resolution
        end = H*W
        flat_patch_pos = torch.arange(0 , end) # N = end
        x_pos = flat_patch_pos % W # N
        x_pos = bias + x_pos * step
        y_pos = flat_patch_pos // W # N
        y_pos = bias + y_pos * step
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 4)[: (self.dim // 4)].float() / self.dim)) # Hc/4
        x_pos = self.scale_w * x_pos
        y_pos = self.scale_h * y_pos
        x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
        y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
        freqs_cis = freqs_cis.reshape(end, -1)
        # import ipdb;ipdb.set_trace()
        return freqs_cis


    def apply_rotary_emb_single(
            self,
            xq: torch.Tensor,
            freqs_cis: torch.Tensor,
    ):
        # xq : B N H Hc
        # import ipdb;ipdb.set_trace()
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
        return xq_out.type_as(xq)

    def forward(self, x, patch_resolution):
        """
        x: [batch, num_patches, num_heads, dim]
        """
        # Check whether the patch resolution is the predefined size
        patch_resolution = to_2tuple(patch_resolution)
        if patch_resolution != self.patch_resolution or self.freqs_cis is None:
            self.patch_resolution = patch_resolution
            self.scale_h = self.base_size / patch_resolution[0]
            self.scale_w = self.base_size / patch_resolution[1]
            freqs_cis = self.compute_position_embedding()
            self.freqs_cis = freqs_cis.to(x.device)
            # self.register_buffer('freqs_cis', freqs_cis.to(x.device))
            # self.register_buffer('freqs_sin', freqs_sin.to(x.device))

        batch, num_patches, num_heads, dim = x.shape
        x = self.apply_rotary_emb_single(x, self.freqs_cis)
        return x

class RotaryEmbeddingFast(nn.Module):
    """Implements 2D rotary embedding (RoPE) for image tokens. Position
    encoding is implemented with sin and cos functions,

        .. math::
            Pos_{cos} = cos(\frac{t}{\theta^{\frac{2i}{d}}} \\
            Pos_{sin} = sin(\frac{t}{\theta^{\frac{2i}{d}}}
    Args:
        embed_dims (int): The feature dimension for each head.
        patch_resolution (int | tuple): The resolution of the
            image, in format (H, W).
        theta (float): The hyperparameter for position coding.
            Defaults to 10000.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 patch_resolution,
                 theta=10000.,
                 init_cfg=None):
        # super(RotaryEmbeddingFast, self).__init__(init_cfg=init_cfg)
        super().__init__()

        self.half_dim = embed_dims // 2
        self.patch_resolution = to_2tuple(patch_resolution)
        self.theta = theta

        freqs_cos, freqs_sin = self.compute_position_embedding()
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

    def compute_position_embedding(self):
        frequency = self.theta**(
            torch.arange(0, self.half_dim, 2).float() / self.half_dim)
        frequency = 1. / frequency

        h, w = self.patch_resolution
        th = torch.arange(h) / h * self.half_dim
        tw = torch.arange(w) / w * self.half_dim

        position_h = (th[:, None] @ frequency[None, :]).repeat(1, 2)
        position_w = (tw[:, None] @ frequency[None, :]).repeat(1, 2)

        height = position_h[:, None, :].expand(h, w, self.half_dim)
        width = position_w[None, :, :].expand(h, w, self.half_dim)
        position = torch.cat((height, width), dim=-1)

        freqs_cos = position.cos().view(-1, position.shape[-1])
        freqs_sin = position.sin().view(-1, position.shape[-1])

        return freqs_cos, freqs_sin

    def forward(self, x, patch_resolution):
        # Check whether the patch resolution is the predefined size
        patch_resolution = to_2tuple(patch_resolution)
        if patch_resolution != self.patch_resolution:
            self.patch_resolution = patch_resolution
            freqs_cos, freqs_sin = self.compute_position_embedding()
            self.register_buffer('freqs_cos', freqs_cos.to(x.device))
            self.register_buffer('freqs_sin', freqs_sin.to(x.device))

        batch, num_heads, num_patches, dim = x.shape

        inputs = x
        x = x.reshape(batch, num_heads, num_patches, -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = x.reshape(batch, num_heads, num_patches, dim)

        return inputs * self.freqs_cos + x * self.freqs_sin


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False,hidden_act: str = 'swish'):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # the final number of params is 4d^2, where d is the hidden size
        # `intermediate_size` is chosen to be (roughly) 2/3 of `hidden_size * hidden_ratio`, and a multiple of 256
        # if intermediate_size is None:
        hidden_ratio, multiple_of = 4, 32
        intermediate_size = int(in_features * hidden_ratio * 2 / 3)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.intermediate_size = intermediate_size

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * self.intermediate_size,bias=False)
        # self.act = act_layer()
        self.fc2 = Linear(self.intermediate_size, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        # x = self.fc1(x)
        # x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        # x = self.fc2(x * self.act(z))
        # x = self.drop(x)
        # import ipdb;ipdb.set_trace()
        y = self.fc1(x)
        gate, y = y.chunk(2, dim=(1 if self.channel_first else -1))
        y = swiglu(gate, y)
        y = self.fc2(y)
        y = self.drop(y)
        return y


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class GLA2D(nn.Module):

    def __init__(
        self,
        d_model: int = 1024,
        expand_v: float = 2.0,
        expand_k: float = 1.0,
        num_heads: int = 4,
        gate_fn: str = 'swish',
        layernorm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        mode: str = 'fused_chunk',
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        use_dirpe: bool = False,
        use_out_act: bool = False,
        use_act_in_conv: bool = True,
        rope_mode: str = 'none',
        channel_first: bool = False,
        # shift_mode='none',
        # channel_gamma=0.25,
        # shift_pixel=1,
        # d_conv: int = 4,
        # conv_bias: bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.channel_first = channel_first
        Linear = Linear2d if channel_first else nn.Linear
        self.d_model = d_model
        # self.shift_mode = shift_mode
        self.d_group = 2
        self.use_dirpe = use_dirpe

        self.mode = mode
        self.value_dim = int(d_model * expand_v)
        self.key_dim = int(d_model * expand_k)
        self.clamp_min = clamp_min

        if self.use_dirpe:
            self.dirpe = nn.Parameter(torch.zeros(2*self.d_model))

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk','bid_fused_chunk','bidv2_fused_chunk'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        self.num_heads = num_heads
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.gate_fn = ACT2FN[gate_fn]
        # import ipdb;ipdb.set_trace()
        if use_out_act:
            self.out_act = ACT2FN[gate_fn]

        self.in_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1,groups=d_model, bias=False),
            ACT2FN['silu'] if use_act_in_conv else nn.Identity(),
        )
        self.qkv_proj = Linear(d_model, self.key_dim+self.key_dim+self.value_dim, bias=False)
        self.gk_proj = nn.Sequential(Linear(d_model,  gate_low_rank_dim,bias=False),
                            Linear(gate_low_rank_dim, self.key_dim*self.d_group,bias=True))


        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)


        self.g_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)
        self.l_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)
        self.g_proj = Linear(d_model, self.value_dim, bias=True)


        self.gate_logit_normalizer = gate_logit_normalizer


        self.rope_mode = rope_mode
        if self.rope_mode == '1d':
        # if use_rope_1d:
            head_qk_dim = self.key_dim // self.num_heads
            assert head_qk_dim <= 256, "head_qk_dim must be less than or equal to 256"
            self.rotary = RotaryEmbedding(dim=head_qk_dim, interleaved=False)
        elif self.rope_mode == '2dv0':
            head_qk_dim = self.key_dim // self.num_heads
            assert head_qk_dim <= 256, "head_qk_dim must be less than or equal to 256"
            self.rotary = RotaryEmbeddingFast(
                embed_dims=head_qk_dim,
                patch_resolution=(14, 14),
                # theta=10000.,
            )
        elif self.rope_mode == '2dv1':
            head_qk_dim = self.key_dim // self.num_heads
            assert head_qk_dim <= 256, "head_qk_dim must be less than or equal to 256"
            self.rotary = RotaryEmbeddingFast2D(
                embed_dims=head_qk_dim,
                patch_resolution=(14, 14),
                # theta=10000.,
            )

        self.reset_parameters()


    def reset_parameters(self):
        if hasattr(self, 'direction_attn_weights'):
            nn.init.constant_(self.direction_attn_weights.weight, 0)
            nn.init.constant_(self.direction_attn_weights.bias, 0)
        if hasattr(self, 'dirpe'):
            trunc_normal_(self.dirpe, std=.02)
        
        if hasattr(self, 'qkv_proj'):
            nn.init.xavier_uniform_(self.qkv_proj.weight, gain=2 ** -2.5)
        else:
            nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
            nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
            nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        if hasattr(self, 'in_proj'):
            nn.init.xavier_uniform_(self.in_proj[0].weight, gain=2 ** -2.5)
            # nn.init.xavier_uniform_(self.in_proj[2].weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.gk_proj[0].weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.gk_proj[1].weight, gain=2 ** -2.5)


    def forward(self, x, **kwargs):
        # import ipdb;ipdb.set_trace()
        mode = self.mode
        if self.channel_first:
            B, C, H, W = x.shape
        else:
            B, H, W, C = x.shape
        # import ipdb;ipdb.set_trace()
        # x_linear = x.flatten(2).transpose(1, 2).contiguous()
        if self.training:
        # if True:
            if self.channel_first:
                local_out = self.in_proj(x)
                xs = local_out
                q,k,v = torch.split(self.qkv_proj(xs).flatten(-2), [self.key_dim,self.key_dim,self.value_dim], dim=1)
                gk = self.gk_proj(xs).flatten(-2)
                fw_gk, bw_gk = gk.chunk(2,dim=1)
                gk = torch.cat([fw_gk,bw_gk.flip(dims=[2])],dim=0)
                gk = rearrange(gk,'b (h d) n -> b h n d',h=self.num_heads)
                if self.rope_mode == '2dv1':
                    # import ipdb;ipdb.set_trace()
                    q1 = rearrange(q, 'b (h d) n -> b n h d', h=self.num_heads)
                    k1 = rearrange(k, 'b (h d) n -> b n h d', h=self.num_heads)
                    q = self.rotary(q1, (H,W))
                    k = self.rotary(k1, (H,W))
                    q, k = q.transpose(1, 2), k.transpose(1, 2)
                elif self.rope_mode == 'none':
                    q = rearrange(q, 'b (h d) n -> b h n d', h=self.num_heads)
                    k = rearrange(k, 'b (h d) n -> b h n d', h=self.num_heads)
                v = rearrange(v, 'b (h d) n -> b h n d', h=self.num_heads)
                q = torch.cat([q,q.flip(dims=[2])],dim=0)
                k = torch.cat([k,k.flip(dims=[2])],dim=0)
                v = torch.cat([v,v.flip(dims=[2])],dim=0)
            else:
                x_2d = x.permute(0,3,1,2).contiguous()
                local_out = self.in_proj(x_2d)
                xs = local_out.permute(0,2,3,1).contiguous().flatten(1,2)
                q,k,v = torch.split(self.qkv_proj(xs), [self.key_dim,self.key_dim,self.value_dim], dim=2)
                gk = self.gk_proj(xs)
                fw_gk, bw_gk = gk.chunk(2,dim=2)
                gk = torch.cat([fw_gk,bw_gk.flip(dims=[1])],dim=0)
                gk = rearrange(gk,'b n (h d) -> b h n d',h=self.num_heads)
                if self.rope_mode == '2dv1':
                    q1 = rearrange(q, 'b n (h d) -> b n h d', h=self.num_heads)
                    k1 = rearrange(k, 'b n (h d) -> b n h d', h=self.num_heads)
                    q = self.rotary(q1, (H,W))
                    k = self.rotary(k1, (H,W))
                    q, k = q.transpose(1, 2), k.transpose(1, 2)
                elif self.rope_mode == 'none':
                    q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
                    k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
                v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
                q = torch.cat([q,q.flip(dims=[2])],dim=0)
                k = torch.cat([k,k.flip(dims=[2])],dim=0)
                v = torch.cat([v,v.flip(dims=[2])],dim=0)

            gk = (F.logsigmoid(gk) / self.gate_logit_normalizer)

            if self.clamp_min is not None:
                gk = torch.clamp_min(gk, self.clamp_min)

            if mode == 'fused_recurrent':
                o,_ = fused_recurrent_gla(q, k, v, gk, None)
            elif mode == 'fused_chunk':
                # import ipdb;ipdb.set_trace()
                o,_ = fused_chunk_gla(q, k, v, gk)
            elif mode == 'chunk':
                o,_ = chunk_gla(q, k, v, gk)
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")

            o_f, o_b = o.chunk(2)
            o = (o_f + o_b.flip(dims=[2])) / 2
        else:
            if self.channel_first:
                # import ipdb;ipdb.set_trace()
                local_out = self.in_proj(x)
                xs = local_out
                q,k,v = torch.split(self.qkv_proj(xs).flatten(-2), [self.key_dim,self.key_dim,self.value_dim], dim=1)
                gk = self.gk_proj(xs).flatten(-2)
                fw_gk, bw_gk = gk.chunk(2,dim=1)
                gk = torch.cat([fw_gk,bw_gk],dim=0)
                gk = rearrange(gk,'b (h d) n -> b h n d',h=self.num_heads)
                if self.rope_mode == '2dv1':
                    # import ipdb;ipdb.set_trace()
                    q1 = rearrange(q, 'b (h d) n -> b n h d', h=self.num_heads)
                    k1 = rearrange(k, 'b (h d) n -> b n h d', h=self.num_heads)
                    q = self.rotary(q1, (H,W))
                    k = self.rotary(k1, (H,W))
                    q, k = q.transpose(1, 2), k.transpose(1, 2)
                elif self.rope_mode == 'none':
                    q = rearrange(q, 'b (h d) n -> b h n d', h=self.num_heads)
                    k = rearrange(k, 'b (h d) n -> b h n d', h=self.num_heads)
                v = rearrange(v, 'b (h d) n -> b h n d', h=self.num_heads)
            else:
                # import ipdb;ipdb.set_trace()
                x_2d = x.permute(0,3,1,2).contiguous()
                local_out = self.in_proj(x_2d)
                xs = local_out.permute(0,2,3,1).contiguous().flatten(1,2)
                q,k,v = torch.split(self.qkv_proj(xs), [self.key_dim,self.key_dim,self.value_dim], dim=2)
                gk = self.gk_proj(xs)
                fw_gk, bw_gk = gk.chunk(2,dim=2)
                gk = torch.cat([fw_gk,bw_gk],dim=0)
                gk = rearrange(gk,'b n (h d) -> b h n d',h=self.num_heads)
                if self.rope_mode == '2dv1':
                    q1 = rearrange(q, 'b n (h d) -> b n h d', h=self.num_heads)
                    k1 = rearrange(k, 'b n (h d) -> b n h d', h=self.num_heads)
                    q = self.rotary(q1, (H,W))
                    k = self.rotary(k1, (H,W))
                    q, k = q.transpose(1, 2), k.transpose(1, 2)
                elif self.rope_mode == 'none':
                    q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
                    k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
                v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
            
            gk = (F.logsigmoid(gk) / self.gate_logit_normalizer)
            if self.clamp_min is not None:
                gk = torch.clamp_min(gk, self.clamp_min)
            o,_ = bid_fused_recurrent_gla(q, k, v, gk, None)
            # o /= 2
            
        o = rearrange(o, 'b h l d -> b l h d')

        # import ipdb;ipdb.set_trace()
        if self.channel_first:
            g = self.g_proj(xs)
            g = g.sigmoid()
            g = rearrange(g, 'b (h d) ih iw -> b (ih iw) h d', h=self.num_heads)
        else:
            g = self.g_proj(xs)
            g = g.sigmoid()
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
        o = self.g_norm(o)
        local_out = rearrange(local_out, 'b (h d) ih iw -> b (ih iw) h d', h=self.num_heads)
        local_out = self.l_norm(local_out)
        o = o * g + local_out * (1-g)
        o = rearrange(o, 'b l h d -> b l (h d)')

        if hasattr(self, 'out_act'):
            # import ipdb;ipdb.set_trace()
            o = self.out_act(o)
        # import ipdb;ipdb.set_trace()
        o = self.o_proj(o)
        if self.channel_first:
            return o.permute(0, 2, 1).reshape(B, -1, H,W)
        else:
            return o.reshape(B, H, W, -1)


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        attn_model="fused_chunk",
        config: GLAConfig = None,
        num_head = 3,
        expand_k = 0.5,
        expand_v = 1.0,
        hidden_act = "swish",
        use_out_act=False,
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = True
        self.mlp_branch = True
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = GLA2D(
                d_model=hidden_dim, 
                expand_k=expand_k,
                expand_v=expand_v,
                num_heads=num_head,
                gate_fn=hidden_act,
                layernorm_eps=config.rms_norm_eps,
                mode=attn_model,
                use_out_act=use_out_act,
                use_act_in_conv=config.use_act_in_conv,
                rope_mode=config.rope_mode,
                channel_first=channel_first,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class HierViG(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        num_heads=[3, 6, 12, 24],
        # =========================
        attn_models=["fused_chunk","fused_chunk","fused_recurrent","fused_recurrent"],
        expand_k = 0.5,
        expand_v = 1.0,
        hidden_act = "swish",
        use_out_act=False,
        use_act_in_conv=True,
        rope_mode="none",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        **kwargs,
    ):
        super().__init__()
        self.attn_models = attn_models
        config = GLAConfig(
            # attn_mode=attn_model,
            hidden_act=hidden_act,
            use_out_act=use_out_act,
            use_act_in_conv=use_act_in_conv,
            rope_mode=rope_mode,
        )

        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        self.num_heads = num_heads
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                num_head = self.num_heads[i_layer],
                attn_model = attn_models[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                config = config,
                expand_k = expand_k,
                expand_v = expand_v,
                hidden_act = hidden_act,
                use_out_act=use_out_act,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        num_head=3,
        attn_model="fused_chunk",
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        config = GLAConfig(),
        expand_k = 0.5,
        expand_v = 1.0,
        hidden_act = "swish",
        use_out_act=False,
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                attn_model=attn_model,
                # ===========================
                config = config,
                num_head = num_head,
                expand_k = expand_k,
                expand_v = expand_v,
                hidden_act = hidden_act,
                use_out_act=use_out_act,
                # ===========================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        # import ipdb;ipdb.set_trace()
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x



    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
class Backbone_HierViG(HierViG):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs

