# -*- coding: utf-8 -*-

# Sect4.2 of Linear Transformers Are Secretly Fast Weight Programmers https://arxiv.org/abs/2102.11174


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import RMSNorm
from fla.ops.delta_rule import fused_recurrent_linear_attn_delta_rule, chunk_linear_attn_delta_rule


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)

@torch.jit.script
def l2_norm(x):
    return x / x.norm(p=2, dim=-1, keepdim=True)

# https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/layers.py#L86C1-L146C1
class DeltaNet(nn.Module):
    def __init__(
        self,
        d_model: int = 1024,
        expand_v: float = 1.0,
        expand_k: float = 1.0,
        num_heads: int = 4,
        mode: str = 'fused_chunk',
        chunk_size: int = 32,
        *args, **kwargs
    ) -> DeltaNet:
        super().__init__()
        self.d_model = d_model
        self.mode = mode 
        assert mode in ['fused_chunk', 'fused_recurrent', 'chunk'], f"Not suppoerted mode `{mode}`."
        self.chunk_size = chunk_size
        self.value_dim = int(d_model * expand_v)
        self.key_dim = int(d_model * expand_k)

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        self.num_heads = num_heads
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.beta_proj = nn.Linear(d_model, self.num_heads, bias=False)

        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.norm = RMSNorm(self.head_v_dim)

    def forward(self, x):
        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        # q = l2_norm(elu_p1(q))
        # k = l2_norm(elu_p1(k))
        q = l2_norm(q)
        k = l2_norm(k)
        beta = rearrange(self.beta_proj(x), 'b n h -> b h n').sigmoid()
        if self.mode == 'fused_recurrent':
            o = fused_recurrent_linear_attn_delta_rule(q, k, v, beta)
        elif self.mode == 'fused_chunk':
            o = chunk_linear_attn_delta_rule(q, k, v, beta, self.chunk_size, fused_chunk=True)
        elif self.mode == 'chunk':
            o = chunk_linear_attn_delta_rule(q, k, v, beta, self.chunk_size, fused_chunk=False)
        else:
            raise NotImplementedError(f"Not supported mode `{self.mode}`.")

        o = self.norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.o_proj(o)
        return o


if __name__ == '__main__':
    import torch
    batch = 4
    seq_len = 1024
    d_model = 1024
    x = torch.randn(batch, seq_len, d_model).to(torch.bfloat16).cuda().requires_grad_(True)
    model = DeltaNet(d_model=d_model).to(torch.bfloat16).cuda()
    y = model(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)
