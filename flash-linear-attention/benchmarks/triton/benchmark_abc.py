# -*- coding: utf-8 -*-

import torch
import triton
from torch.nn import functional as F

from fla.ops.abc import chunk_abc
from fla.ops.gla import chunk_gla
from fla.ops.retention import chunk_retention

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['seq_len'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['abc',  'gla', 'abc_bwd', 'gla_bwd', 'retention_bwd', 'flash_bwd'],
        # label name for the lines
        line_names=['abc',  'gla', 'abc_bwd', 'gla_bwd', 'retention_bwd', 'flash_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'),
                ('cyan', ':'), ('yellow', 'dotted'), ('black', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(seq_len, provider):
    device = 'cuda'
    dtype = torch.bfloat16
    requires_grad = True
    batch_size, n_heads, d_head, n_slots = 16, 8, 128, 64

    q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
    if provider.startswith('flash'):
        q = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, requires_grad=requires_grad, dtype=dtype)
    if provider.startswith('gla'):
        g = F.logsigmoid(torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype))
        g = g.clamp_min(-5).requires_grad_(requires_grad)
    if provider.startswith('abc'):
        s = torch.randn(batch_size, n_heads, seq_len, n_slots, device=device, requires_grad=requires_grad, dtype=dtype)

    do = torch.ones_like(v, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'abc':
        results = triton.testing.do_bench(lambda: chunk_abc(q, k, v, s), quantiles=quantiles)
    elif provider == 'gla':
        results = triton.testing.do_bench(lambda: chunk_gla(q, k, v, g), quantiles=quantiles)
    elif provider == 'abc_bwd':
        results = triton.testing.do_bench(lambda: chunk_abc(q, k, v, s).backward(do), quantiles=quantiles)
    elif provider == 'gla_bwd':
        results = triton.testing.do_bench(lambda: chunk_gla(q, k, v, g).backward(do), quantiles=quantiles)
    elif provider == 'retention_bwd':
        results = triton.testing.do_bench(lambda: chunk_retention(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'flash_bwd':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True).backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
