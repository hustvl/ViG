# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from __future__ import annotations
import torch
import torch.nn as nn
from functools import partial
from typing import List, Optional, Tuple, Union

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


from fla.models import GLAConfig
from timm.models.layers import DropPath, to_2tuple
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

def fp32_fused_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: int = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    in_dtype = v.dtype
    q = q.to(torch.float)
    k = k.to(torch.float)
    v = v.to(torch.float)
    g = g.to(torch.float)


    if scale == -1:
        scale = q.shape[-1] ** -0.5
    if initial_state is not None:
        initial_state = initial_state.detach()
    seq_len = v.shape[-2]
    d_head_v = v.shape[-1]
    q, k, v, g = map(lambda x: pad(x), [q, k, v, g])
    o, final_state = FusedChunkGLAFunction.apply(
        q, k, v, g, scale, initial_state, output_final_state)
    o = o[..., :seq_len, :d_head_v]
    return o.to(in_dtype), final_state

class MineAttention(Attention):
    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class GatedLinearAttention(nn.Module):

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
        use_out_gate: bool = True,
        rope_mode: str = 'none',
        use_act_in_conv: bool = True,
        use_bias_in_dwconv: bool = False,
        *args, **kwargs
    ) -> GatedLinearAttention:
        super().__init__()
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
        self.use_out_gate = use_out_gate

        if use_out_act:
            self.out_act = ACT2FN[gate_fn]


        self.in_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1,groups=d_model, bias=use_bias_in_dwconv),
            ACT2FN['silu'] if use_act_in_conv else nn.Identity(),
        )
        self.qkv_proj = nn.Linear(d_model, self.key_dim+self.key_dim+self.value_dim, bias=False)
        self.gk_proj = nn.Sequential(nn.Linear(d_model,  gate_low_rank_dim, bias=False),
                                    nn.Linear(gate_low_rank_dim, self.key_dim*2, bias=True))

        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)


        self.g_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)
        self.l_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)
        self.g_proj = nn.Linear(d_model, self.value_dim, bias=True)

        self.gate_logit_normalizer = gate_logit_normalizer

        self.rope_mode = rope_mode
        if self.rope_mode == '1d':
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
        if hasattr(self, 'g_proj'):
            nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.gk_proj[0].weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.gk_proj[1].weight, gain=2 ** -2.5)
    def forward(self, x,lower_bound=0, **kwargs):
        
        mode = self.mode
        B, L, C = x.shape
        if self.training:
        # if True:
            if kwargs['token_position'] is not None:

                token_position = kwargs['token_position']
                x_patch = torch.cat([x[:, :token_position, :], x[:, (token_position+1):, :]], dim=1)
                x_2d = rearrange(x_patch, 'b (h w) c -> b c h w', h=kwargs['patch_resolution'][0], w=kwargs['patch_resolution'][1]).contiguous()
                conv2d_out = self.in_proj(x_2d).flatten(2) # B,C,H,W
                local_out = torch.cat([conv2d_out[:, :, :token_position], x[:, token_position:token_position+1, :].transpose(1,2), conv2d_out[:, :, token_position:]], dim=2)
            else:
                x_2d = rearrange(x, 'b (h w) c -> b c h w', h=kwargs['patch_resolution'][0], w=kwargs['patch_resolution'][1]).contiguous()
                local_out = self.in_proj(x_2d).flatten(2) # B,C,H,W
            xs = local_out.permute(0,2,1).contiguous()
            q,k,v = torch.split(self.qkv_proj(xs), [self.key_dim,self.key_dim,self.value_dim], dim=2)
            gk = self.gk_proj(xs)
            fw_gk, bw_gk = gk.chunk(2,dim=2)
            gk = torch.cat([fw_gk,bw_gk.flip(dims=[1])],dim=0)
            gk = rearrange(gk,'b n (h d) -> b h n d',h=self.num_heads)
            if self.rope_mode == '1d':

                q1 = rearrange(q, 'b n (h d) -> b n h d', h=self.num_heads)
                k1 = rearrange(k, 'b n (h d) -> b n h d', h=self.num_heads)
                q, k = self.rotary(q1, k1, 0)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
            elif self.rope_mode == '2dv0':

                q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
                k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
                q = self.rotary(q, kwargs['patch_resolution'])
                k = self.rotary(k, kwargs['patch_resolution'])
            elif self.rope_mode == '2dv1':

                q1 = rearrange(q, 'b n (h d) -> b n h d', h=self.num_heads)
                k1 = rearrange(k, 'b n (h d) -> b n h d', h=self.num_heads)
                if kwargs['token_position'] is not None:

                    q1 = torch.cat([q1[:, :token_position, :], q1[:, (token_position+1):, :]], dim=1)
                    k1 = torch.cat([k1[:, :token_position, :], k1[:, (token_position+1):, :]], dim=1)
                    q = self.rotary(q1, kwargs['patch_resolution'])
                    k = self.rotary(k1, kwargs['patch_resolution'])
                    q = torch.cat([q[:, :token_position, :,:], q1[:, token_position:token_position+1, :,:], q[:, token_position:,:,:]], dim=1)
                    k = torch.cat([k[:, :token_position, :,:], k1[:, token_position:token_position+1, :,:], k[:, token_position:,:,:]], dim=1)

                else:
                    q = self.rotary(q1, kwargs['patch_resolution'])
                    k = self.rotary(k1, kwargs['patch_resolution'])
                q, k = q.transpose(1, 2), k.transpose(1, 2)
            elif self.rope_mode == 'none':
                q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
                k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
            else:
                raise NotImplementedError(f"Not supported rope mode `{self.rope_mode}`.")
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
                o,_ = fused_chunk_gla(q, k, v, gk)
            elif mode == 'chunk':
                o,_ = chunk_gla(q, k, v, gk)
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")

            o_f, o_b = o.chunk(2)
            o = (o_f + o_b.flip(dims=[2])) / 2
        else:
            x_2d = rearrange(x, 'b (h w) c -> b c h w', h=kwargs['patch_resolution'][0], w=kwargs['patch_resolution'][1]).contiguous()
            local_out = self.in_proj(x_2d).flatten(2) # B,C,H,W
            xs = local_out.permute(0,2,1).contiguous()
            q,k,v = torch.split(self.qkv_proj(xs), [self.key_dim,self.key_dim,self.value_dim], dim=2)
            gk = self.gk_proj(xs)
            fw_gk, bw_gk = gk.chunk(2,dim=2)
            gk = torch.cat([fw_gk,bw_gk],dim=0)
            gk = rearrange(gk,'b n (h d) -> b h n d',h=self.num_heads)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
            gk = (F.logsigmoid(gk) / self.gate_logit_normalizer)
            if self.clamp_min is not None:
                gk = torch.clamp_min(gk, self.clamp_min)
            o,_ = bid_fused_recurrent_gla(q, k, v, gk, None) # we mean the direction-wise output in the kernel
            o /= 2
        o = rearrange(o, 'b h l d -> b l h d')


        local_out = rearrange(local_out, 'b (h d) l -> b l h d', h=self.num_heads)
        local_out = self.l_norm(local_out)
        
        o = self.g_norm(o)
        g = self.g_proj(xs).sigmoid()
        # g = lower_bound + (1-lower_bound) * g
        g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
        o = o * g + local_out * (1-g)
        o = rearrange(o, 'b l h d -> b l (h d)')
        if hasattr(self, 'out_act'):
            o = self.out_act(o)
        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                gk=gk, q=q, k=k, v=v,
                g=g,
            ))
        o = self.o_proj(o)
        return o




class GLAMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ) -> GLAMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        return swiglu_linear(gate, y, self.down_proj.weight.to(y.dtype), self.down_proj.bias)

class MLP(nn.Module):
    def __init__(self,         
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        drop: float = 0.,
    ) -> MLP:
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
            hidden_features = hidden_size * hidden_ratio
        
        self.act = ACT2FN[hidden_act]
        
        self.fc1 = nn.Linear(hidden_size, hidden_features)
        # self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, hidden_size)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GLABlock(nn.Module):
    def __init__(self, config: GLAConfig, layer_idx: int, drop_path: float = 0., is_attn: bool = False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.hidden_size = config.hidden_size
        # import ipdb;ipdb.set_trace()
        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.layer_scale = (config.init_values is not None)
        if self.layer_scale:
            self.gamma1 = nn.Parameter(config.init_values * torch.ones((config.hidden_size)), requires_grad=True)
            self.gamma2 = nn.Parameter(config.init_values * torch.ones((config.hidden_size)), requires_grad=True)
        if not is_attn:
            self.attn = GatedLinearAttention(
                d_model=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_attention_heads,
                gate_fn=config.hidden_act,
                layernorm_eps=config.rms_norm_eps,
                mode=config.attn_mode,
                clamp_min=config.clamp_min,
                fuse_norm=config.fuse_norm,
                use_dirpe=config.use_dirpe,
                use_out_act=config.use_out_act,
                use_out_gate=config.use_out_gate,
                layer_idx=layer_idx,
                is_attn=is_attn,
                rope_mode = config.rope_mode,
                use_act_in_conv = config.use_act_in_conv,
                use_bias_in_dwconv=config.use_bias_in_dwconv,
                # use_rope_1d=config.use_rope_1d,
                # use_rope_2d=config.use_rope_2d,
            )
        else:
            # import ipdb;ipdb.set_trace()
            self.attn = MineAttention(config.hidden_size, num_heads=config.num_attention_heads, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        if config.use_swiglu:
            self.mlp = GLAMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act
            )
        else:
            self.mlp = MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                # out_features=config.hidden_size,
                hidden_act=config.hidden_act,
                # drop=config.hidden_dropout_prob,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        # currently not supported
        attn_weights, present_key_value = None, None

        hidden_states = self.attn_norm(hidden_states)
        
        hidden_states = self.attn(hidden_states,lower_bound=lower_bound,**kwargs)
        if self.layer_scale:
            hidden_states, residual = self.mlp_norm(self.drop_path(self.gamma1 * hidden_states), residual, True)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + self.drop_path(self.gamma2 * hidden_states)
        else:
            hidden_states, residual = self.mlp_norm(self.drop_path(hidden_states), residual, True)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + self.drop_path(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
class V2PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,use_bias_in_patch=True,use_act_in_patch=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        stride = patch_size[0] // 2
        kernel_size = stride + 1
        padding = 1
        if use_act_in_patch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding,bias=use_bias_in_patch),
                LayerNorm2d(embed_dim // 2),
                ACT2FN["swish"],
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1,bias=use_bias_in_patch),
                LayerNorm2d(embed_dim),
                ACT2FN["swish"],
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding,bias=use_bias_in_patch),
                LayerNorm2d(embed_dim // 2),
                ACT2FN["swish"],
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1,bias=use_bias_in_patch),
                LayerNorm2d(embed_dim),
            )


    def forward(self, x, scan_mode="default"):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        _,_,new_H, new_W = x.shape
        if self.flatten:
            if scan_mode == "default":
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            elif scan_mode == "zigzag":
                # import ipdb;ipdb.set_trace()
                x[:,:,1::2] = x[:,:,1::2].flip(-1)
                x = x.contiguous().flatten(2).transpose(1, 2)
        return x, (new_H, new_W)

class V1PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, scan_mode="default"):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        _,_,new_H, new_W = x.shape
        if self.flatten:
            if scan_mode == "default":
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            elif scan_mode == "zigzag":
                x[:,:,1::2] = x[:,:,1::2].flip(-1)
                x = x.contiguous().flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (new_H, new_W)
    
class ViG(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 stride=16,
                 depth=12, 
                 num_heads=2,
                 embed_dim=192, 
                 channels=3, 
                 num_classes=1000,
                 if_cls_token=True,
                 if_abs_pos_embed=True,
                 if_rc_pe=False,
                 use_msg=False,
                 use_dirpe=False,
                 use_middle_cls_token=False,
                 classification_mode="avgpool",
                 drop_rate=0.,
                 pt_hw_seq_len=14,
                 attn_model="fused_chunk",
                 hidden_act="swish",
                 use_out_act=False,
                 use_out_gate=True,
                 use_lower_bound=False,
                 use_swiglu=False,
                 use_act_in_conv=True,
                 rope_mode='none',
                 drop_path_rate=0.1,
                 expand_k=0.5,
                 expand_v=1,
                 scan_mode="default",
                 patch_embed_version='v1',
                 use_bias_in_patch=True,
                 use_act_in_patch=False,
                 init_values=None,
                 use_bias_in_dwconv=False,
                 **kwargs):
        super().__init__()
        self.use_msg = use_msg
        self.use_lower_bound = use_lower_bound
        _PATCH_EMBED= dict(
            v1=V1PatchEmbed,
            v2=V2PatchEmbed,
        )
        # init gla config
        config = GLAConfig(
            hidden_size=embed_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            attn_mode=attn_model,
            expand_k=expand_k,
            expand_v=expand_v,
            hidden_act=hidden_act,
            use_out_act=use_out_act,
            use_out_gate=use_out_gate,
            use_dirpe=use_dirpe,
            init_values=init_values,
            use_swiglu=use_swiglu,
            rope_mode=rope_mode,
            use_act_in_conv=use_act_in_conv,
            use_bias_in_dwconv=use_bias_in_dwconv,
        )
        self.config = config
        self.classification_mode = classification_mode
        self.scan_mode = scan_mode
        # import ipdb;ipdb.set_trace()
        if self.classification_mode == "mid_clstok":
            if_cls_token = True
            use_middle_cls_token = True
        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        PatchEmbed = _PATCH_EMBED.get(patch_embed_version,None)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim,use_bias_in_patch=use_bias_in_patch,use_act_in_patch=use_act_in_patch)
        num_patches = self.patch_embed.num_patches
        self.patch_resolution = (img_size//patch_size, img_size//patch_size)
        self.use_middle_cls_token = use_middle_cls_token
        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            
        if if_abs_pos_embed:
            if not if_rc_pe:
                if if_cls_token:
                    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
                else:
                    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            else:
                self.row_pos_embed = nn.Parameter(torch.randn(1, self.patch_resolution[1], embed_dim))
                self.col_pos_embed = nn.Parameter(torch.randn(self.patch_resolution[0],1 , embed_dim))

            self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_lower_bound:
            self.lower_bounds = nn.Parameter(
                torch.ones(depth, self.embed_dim), requires_grad=True
            )

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        is_attn = [False] * depth
        is_attn[-1] = False
        # import ipdb;ipdb.set_trace()
        self.blocks = nn.ModuleList(
            [GLABlock(config, layer_idx, dpr[layer_idx], is_attn[layer_idx]) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.init_weights()
        # self.apply(self._init_weights)
        if if_cls_token:
            trunc_normal_(self.cls_token, std=.02)
        if if_abs_pos_embed:
            if if_rc_pe:
                trunc_normal_(self.row_pos_embed,std=.02)
                trunc_normal_(self.col_pos_embed,std=.02)
            else:
                trunc_normal_(self.pos_embed, std=.02)
    def init_weights(self):
        # import ipdb;ipdb.set_trace()
        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m,GatedLinearAttention):
                m.reset_parameters()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x, scan_mode=self.scan_mode)
        _, M, _ = x.shape
        token_position = None
        if hasattr(self, 'cls_token'):
            if self.use_middle_cls_token:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                token_position = M // 2
                x = torch.cat((x[:, :token_position, :], cls_tokens, x[:, token_position:, :]), dim=1)
            else:
                token_position = -1
                cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((x, cls_tokens), dim=1)
        if hasattr(self, 'pos_embed'):
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode='bicubic',
                num_extra_tokens=0)
            x = self.pos_drop(x)
        elif hasattr(self, 'row_pos_embed'):
            pos_embedding = (self.row_pos_embed + self.col_pos_embed).reshape(1, self.patch_resolution[0] * self.patch_resolution[1], -1)
            x = x + resize_pos_embed(
                pos_embedding,
                self.patch_resolution,
                patch_resolution,
                mode='bicubic',
                num_extra_tokens=0)
            x = self.pos_drop(x)
        lower_bounds = [0] * len(self.blocks)
        if self.use_lower_bound:
            lower_bounds = self.lower_bounds
            lower_bounds = F.softmax(lower_bounds, dim=0)
            lower_bounds = torch.cumsum(lower_bounds, dim=0)
            lower_bounds -= lower_bounds[0, ...].clone()

        num_roll = 0
        
        for i, blk in enumerate(self.blocks):
            lower_bound = lower_bounds[i]
            x = blk(x, patch_resolution=patch_resolution, token_position=token_position, lower_bound=lower_bound)
            x = x[0]
            if self.use_msg:
                if num_roll %2==0:
                    x = torch.roll(x, shifts=1, dims=1)
                    token_position += 1
                else:
                    x = torch.roll(x, shifts=-1, dims=1)
                    token_position -= 1
                num_roll += 1
        x = self.norm(x)
        if self.classification_mode == "avgpool":
            patch_token = x.reshape(B, *patch_resolution, -1)
            patch_token = patch_token.permute(0, 3, 1, 2)
            x = self.gap(patch_token).flatten(1)
            return x
        elif hasattr(self, 'cls_token'):
            return x[:, token_position]
        elif self.classification_mode == "feat":
            patch_token = x.reshape(B, *patch_resolution, -1)
            patch_token = patch_token.permute(0, 3, 1, 2)
            return patch_token
            
    def forward(self, x, **kwargs):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=0):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = F.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)



