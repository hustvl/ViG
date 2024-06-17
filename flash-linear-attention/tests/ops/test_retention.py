# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.retention import (chunk_retention, fused_chunk_retention,
                               parallel_retention)
from fla.ops.retention.naive import naive_retention


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("D", [64, 100])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("T", [300, 512])
def test_chunk(dtype, D, expand_ratio, T, B=4, H=4):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float32 else 1e-1
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D * expand_ratio), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    ref = naive_retention(q, k, v)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # triton implementation
    tri = chunk_retention(q, k, v)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    assert ref.allclose(tri, 0, atol), f"o diff: {torch.abs(ref - tri).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("D", [64, 100])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("T", [300, 512])
def test_fused_chunk(dtype, D, expand_ratio, T, B=4, H=4):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float32 else 1e-1
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D * expand_ratio), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    ref = naive_retention(q, k, v)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # triton implementation
    tri = fused_chunk_retention(q, k, v)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    assert ref.allclose(tri, 0, atol), f"o diff: {torch.abs(ref - tri).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("D", [64, 100])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("T", [300, 512])
def test_parallel(dtype, D, expand_ratio, T, B=4, H=4):
    torch.manual_seed(42)
    atol = 1e-2 if dtype == torch.float32 else 1e-1
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D * expand_ratio), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    ref = naive_retention(q, k, v)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # triton implementation
    tri = parallel_retention(q, k, v)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    assert ref.allclose(tri, 0, atol), f"o diff: {torch.abs(ref - tri).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"
