# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"

import torch
import torch.nn.functional as F
from benchmark import benchmark_combined, benchmark_forward

from fla.ops.gla.chunk_fuse import batch_bid_fused_chunk_gla, seq_bid_fused_chunk_gla


def time_fwd_bwd(func, *args, **kwargs):
    time_fb = benchmark_forward(func, *args, **kwargs)
    return time_fb[1].mean


repeats = 30
device = 'cuda'
dtype = torch.float32

# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
# 224, 256, 512
bs_seqlen_vals = [(32,196), (32,256), (32,512)]
causal_vals = [True]
headdim_vals = [64, 128, 256, 512]
headdim = 64
nheads_vals = [3,6,12]
dropout_p = 0.0

methods = (["batch_bid_fused_chunk_gla", "seq_bid_fused_chunk_gla"])

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for nheads in nheads_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (causal, headdim, batch_size, seqlen)
            # nheads = dim // headdim
            dim = headdim * nheads

            q = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            k = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            g = F.logsigmoid(torch.randn(2*batch_size, nheads, seqlen, headdim, device=device,
                             requires_grad=True)).clamp_min(-5).requires_grad_(True)
            v = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)

            fb = time_fwd_bwd(
                batch_bid_fused_chunk_gla, q, k, v, g, verbose=False, amp=False
            )
            time_f_b[config, "batch_bid_fused_chunk_gla"] = fb
            # time_b[config, "fused_chunk"] = b

            q2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            k2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            g2 = F.logsigmoid(torch.randn(2*batch_size, nheads, seqlen, headdim, device=device,
                              requires_grad=True, dtype=dtype)).clamp_min(-5).requires_grad_(True)
            v2 = torch.randn(batch_size, nheads, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)

            f_b = time_fwd_bwd(
                seq_bid_fused_chunk_gla, q2, k2, v2, g2,  verbose=False, amp=False
            )
            time_f_b[config, "seq_bid_fused_chunk_gla"] = f_b

            print(f"### causal={causal}, nhead={nheads}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")

            # print(f"### causal={causal}, nhead={nheads}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")

            for method in methods:
                # time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                print(f"{method} fwd : {time_f_b[config, method]:.10f} ")

                # speed_f[config, method] = efficiency(
                #     flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                #     time_f[config, method]
                # )
                # speed_b[config, method] = efficiency(
                #     flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                #     time_b[config, method]
                # )
                # speed_f_b[config, method] = efficiency(
                #     flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                #     time_f_b[config, method]
                # )
                # print(
                #     f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                #     f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                #     f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                # )


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
