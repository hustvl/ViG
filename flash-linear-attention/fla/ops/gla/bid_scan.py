
# triton cross scan, 2x speed than pytorch implementation =========================
import torch
import triton
import triton.language as tl
#(x, y, BC, BT, d_head, seq_len, NT)
@triton.jit
def triton_bid_scan(
    x, # (batch_size, n_heads, seq_len, d_head)
    y, # (2*batch_size, n_heads, seq_len, d_head)
    BC: tl.constexpr,
    BT: tl.constexpr,
    d_head: tl.constexpr,
    n_heads: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    NT: tl.constexpr,
):
    i_c, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # Calculate actual indexes within batch and head
    batch_idx = i_bh // n_heads
    head_idx = i_bh % n_heads

    # Calculate the starting index for each block
    block_start_seq = i_t * BT
    block_start_depth = i_c * BC

    # Ranges within the block
    seq_range = tl.arange(0, BT)
    depth_range = tl.arange(0, BC)

    # Compute global indexes
    seq_idx = block_start_seq + seq_range
    depth_idx = block_start_depth + depth_range

    # Prevent out-of-bounds memory access
    mask = (seq_idx < seq_len)[:, None] & (depth_idx < d_head)

    # Offset for normal and mirrored output
    offset_normal = batch_idx * n_heads * seq_len * d_head + head_idx * seq_len * d_head + seq_idx[:, None] * d_head + depth_idx
    offset_mirrored = (batch_idx * n_heads * seq_len * d_head + head_idx * seq_len * d_head + (seq_len - seq_idx - 1)[:, None] * d_head + depth_idx) + batch_size * n_heads * seq_len * d_head

    # Load and store operations
    x_values = tl.load(x + offset_normal, mask=mask)
    tl.store(y + offset_normal, x_values, mask=mask)
    tl.store(y + offset_mirrored, x_values, mask=mask)

"""
given y: [2*batch_size, n_heads, seq_len, d_head]
y_f, y_b = y.chunk(2, dim=0)
x = y_f + y_b.flip(dims=[2])
return x: [batch_size, n_heads, seq_len, d_head]
"""
@triton.jit
def triton_bid_merge(
    y, # Pointer to the input tensor data (2*batch_size, n_heads, seq_len, d_head)
    x, # Pointer to the output tensor data (batch_size, n_heads, seq_len, d_head)
    BC: tl.constexpr,
    BT: tl.constexpr,
    d_head: tl.constexpr,
    n_heads: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    NT: tl.constexpr,
):
    i_c = tl.program_id(0)  # Program along depth (d_head)
    i_t = tl.program_id(1)  # Program along seq_len
    i_bh = tl.program_id(2)  # Program along batch_size * n_heads

    # Calculate actual indexes within batch and head
    batch_idx = i_bh // n_heads
    head_idx = i_bh % n_heads

    # Calculate the starting index for each block
    block_start_seq = i_t * BT
    block_start_depth = i_c * BC

    # Ranges within the block
    seq_range = tl.arange(0, BT)
    depth_range = tl.arange(0, BC)

    # Compute global indexes
    seq_idx = block_start_seq + seq_range
    depth_idx = block_start_depth + depth_range

    # Prevent out-of-bounds memory access
    mask = (seq_idx < seq_len)[:, None] & (depth_idx < d_head)

    # Compute offsets for input and output
    offset_normal = batch_idx * n_heads * seq_len * d_head + head_idx * seq_len * d_head + seq_idx[:, None] * d_head + depth_idx
    offset_mirrored = (batch_idx * n_heads * seq_len * d_head + head_idx * seq_len * d_head + (seq_len - seq_idx - 1)[:, None] * d_head + depth_idx) + batch_size * n_heads * seq_len * d_head

    # Load from original and mirrored positions
    normal_vals = tl.load(y + offset_normal, mask=mask)
    mirrored_vals = tl.load(y + offset_mirrored, mask=mask)

    # Combine the values by adding them directly (or other logic as required)
    combined_vals = normal_vals + mirrored_vals

    # Store the result back to x
    tl.store(x + offset_normal, combined_vals, mask=mask)



## beigin debug
# import torch
# def torch_bid_scan(x):
#     """
#     x: [batch_size, n_heads, seq_len, d_head]
#     """
#     y = torch.cat([x, x.flip(dims=[2])], dim=0)
#     return y

# def trit_bid_scan(x):
#     batch_size, n_heads, seq_len, d_head = x.shape
#     batch_size, n_heads, seq_len, d_head = int(batch_size), int(n_heads), int(seq_len), int(d_head)
#     BC, BT = min(triton.next_power_of_2(d_head), 1), min(triton.next_power_of_2(seq_len), 64)
#     NT, NC = triton.cdiv(seq_len, BT), triton.cdiv(d_head, BC)
#     # ctx.shape = (batch_size, n_heads, seq_len, d_head)
#     # ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
#     x = x.contiguous()
#     y = x.new_empty((2*batch_size, n_heads, seq_len, d_head))
#     triton_bid_scan[(NC, NT, batch_size*n_heads)](x, y, BC, BT, d_head, n_heads,batch_size, seq_len,  NT)
#     return y
# x = torch.randn(2, 4, 8, 16, device="cuda").contiguous()
# import ipdb;ipdb.set_trace()
# a = torch_bid_scan(x)
# b = trit_bid_scan(x)
# assert torch.allclose(a, b)

# def torch_bid_merge(x):
#     """
#     x: [2*batch_size, n_heads, seq_len, d_head]
#     """
#     x_f, x_b = x.chunk(2, dim=0)
#     y = x_f + x_b.flip(dims=[2])
#     return y
# def trit_bid_merge(x):
#     """
#     x: [2*batch_size, n_heads, seq_len, d_head]
#     """
#     double_batch_size, n_heads, seq_len, d_head = x.shape
#     batch_size = double_batch_size // 2
#     BC, BT = min(triton.next_power_of_2(d_head), 1), min(triton.next_power_of_2(seq_len), 64)
#     NT, NC = triton.cdiv(seq_len, BT), triton.cdiv(d_head, BC)
#     x = x.contiguous()
#     y = x.new_empty((batch_size, n_heads, seq_len, d_head))
#     triton_bid_merge[(NC, NT, batch_size*n_heads)](x, y, BC, BT, d_head, n_heads,batch_size, seq_len,  NT)
#     return y
# x = torch.randn(2, 4, 8, 16, device="cuda").contiguous()
# import ipdb;ipdb.set_trace()
# a = torch_bid_merge(x)
# b = trit_bid_merge(x)
# assert torch.allclose(a, b)
class BidScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        x: [batch_size, n_heads, seq_len, d_head]
        """
        batch_size, n_heads, seq_len, d_head = x.shape
        batch_size, n_heads, seq_len, d_head = int(batch_size), int(n_heads), int(seq_len), int(d_head)
        BC, BT = min(triton.next_power_of_2(d_head), 1), min(triton.next_power_of_2(seq_len), 64)
        NT, NC = triton.cdiv(seq_len, BT), triton.cdiv(d_head, BC)
        ctx.shape = (batch_size, n_heads, seq_len, d_head)
        ctx.triton_shape = (BC, BT, NC, NT)
        x = x.contiguous()
        y = x.new_empty((2*batch_size, n_heads, seq_len, d_head))
        triton_bid_scan[(NC, NT, batch_size*n_heads)](x, y, BC, BT, d_head, n_heads,batch_size, seq_len,  NT)
        return y
    
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: [2*batch_size, n_heads, seq_len, d_head]
        batch_size, n_heads, seq_len, d_head = ctx.shape
        BC, BT, NC, NT = ctx.triton_shape
        y = y.contiguous().view(2*batch_size, n_heads, seq_len, d_head)
        x = y.new_empty((batch_size, n_heads, seq_len, d_head))
        triton_bid_merge[(NC, NT, batch_size*n_heads)](y, x, BC, BT, d_head, n_heads,batch_size, seq_len,  NT)
        return x


# class CrossMergeTriton(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, y: torch.Tensor):
#         B, K, C, H, W = y.shape
#         B, C, H, W = int(B), int(C), int(H), int(W)
#         BC, BH, BW = min(triton.next_power_of_2(C), 1), min(triton.next_power_of_2(H), 64), min(triton.next_power_of_2(W), 64)
#         NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
#         ctx.shape = (B, C, H, W)
#         ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
#         y = y.contiguous().view(B, 4, C, H, W)
#         x = y.new_empty((B, C, H, W))
#         triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
#         return x.view(B, C, -1)
    
#     @staticmethod
#     def backward(ctx, x: torch.Tensor):
#         # out: (b, d, l)
#         B, C, H, W = ctx.shape
#         BC, BH, BW, NC, NH, NW = ctx.triton_shape
#         x = x.contiguous()
#         y = x.new_empty((B, 4, C, H, W))
#         triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
#         return y


# class CrossScanTriton1b1(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x: torch.Tensor):
#         B, K, C, H, W = x.shape
#         B, C, H, W = int(B), int(C), int(H), int(W)
#         BC, BH, BW = min(triton.next_power_of_2(C), 1), min(triton.next_power_of_2(H), 64), min(triton.next_power_of_2(W), 64)
#         NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
#         ctx.shape = (B, C, H, W)
#         ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
#         x = x.contiguous()
#         y = x.new_empty((B, 4, C, H, W))
#         triton_cross_scan_1b1[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
#         return y.view(B, 4, C, -1)
    
#     @staticmethod
#     def backward(ctx, y: torch.Tensor):
#         # out: (b, k, d, l)
#         B, C, H, W = ctx.shape
#         BC, BH, BW, NC, NH, NW = ctx.triton_shape
#         y = y.contiguous().view(B, 4, C, H, W)
#         x = y.new_empty((B, 4, C, H, W))
#         triton_cross_merge_1b1[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
#         return x


