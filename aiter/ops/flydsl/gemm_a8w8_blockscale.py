"""FlyDSL-style standalone blockscale FP8 GEMM for attention projections.

Implements A8W8 blockscale GEMM (per_1x128 quantization) using Triton as
the kernel backend, wrapped with a FlyDSL-compatible API.

This serves as the FlyDSL alternative to CK's gemm_a8w8_blockscale for
the 6 attention projection shapes in GLM-5:
  (N=2048, K=2048), (N=2048, K=6144), (N=2624, K=6144),
  (N=3072, K=6144), (N=6144, K=2048), (N=6144, K=6144)

Scale layout: x_scale [M, K//128], w_scale [N//128, K//128]
"""

import os
import torch
import triton
import triton.language as tl

SCALE_BLOCK_SIZE = 128
_SCALE_BLOCK_SIZE_CONSTEXPR = tl.constexpr(128)


@triton.jit
def _gemm_a8w8_blockscale_kernel(
    A_ptr, B_ptr, C_ptr,
    x_scale_ptr, w_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_xs_m, stride_xs_k,
    stride_ws_n, stride_ws_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLK: tl.constexpr,
):
    """FP8 blockscale GEMM kernel.

    For each BLOCK_K tile (= 128 = scale_block_size):
      1. Load FP8 A and B tiles
      2. Compute partial GEMM in FP32
      3. Load blockscale values for this K-block
      4. Scale the partial result
      5. Accumulate to main accumulator
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_K)

    for kb in range(0, num_k_blocks):
        k_start = kb * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        mask_m = offs_m < M
        mask_k = offs_k < K
        mask_n = offs_n < N

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        tile_acc = tl.dot(a, b).to(tl.float32)

        xs_ptrs = x_scale_ptr + offs_m * stride_xs_m + kb * stride_xs_k
        ws_ptrs = w_scale_ptr + (offs_n // SCALE_BLK) * stride_ws_n + kb * stride_ws_k

        x_s = tl.load(xs_ptrs, mask=mask_m, other=1.0)
        w_s = tl.load(ws_ptrs, mask=mask_n, other=1.0)

        scale_matrix = x_s[:, None] * w_s[None, :]
        accumulator += tile_acc * scale_matrix

    c = accumulator.to(tl.bfloat16)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_out)


def flydsl_gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FlyDSL-style blockscale FP8 GEMM.

    Args:
        x: (M, K) FP8 activation tensor
        w: (N, K) FP8 weight tensor (note: N-major, transposed before compute)
        x_scale: (M, K // 128) f32 activation blockscale
        w_scale: (N // 128, K // 128) f32 weight blockscale
        out: optional (M, N) output tensor
        out_dtype: output dtype (default bfloat16)

    Returns:
        (M, N) tensor in out_dtype
    """
    M, K = x.shape
    N = w.shape[0]
    K_w = w.shape[1]
    assert K == K_w, f"K mismatch: x has K={K}, w has K={K_w}"
    assert K % SCALE_BLOCK_SIZE == 0, f"K={K} must be divisible by {SCALE_BLOCK_SIZE}"

    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=x.device)

    w_t = w.t().contiguous()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = SCALE_BLOCK_SIZE

    if M <= 16:
        BLOCK_M = 16
    elif M <= 32:
        BLOCK_M = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _gemm_a8w8_blockscale_kernel[grid](
        x, w_t, out,
        x_scale, w_scale,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
        x_scale.stride(0), x_scale.stride(1) if x_scale.dim() > 1 else 1,
        w_scale.stride(0), w_scale.stride(1) if w_scale.dim() > 1 else 1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SCALE_BLK=SCALE_BLOCK_SIZE,
    )

    return out


# GLM-5 attention projection shapes
GLM5_SHAPES = [
    (2048, 2048),
    (2048, 6144),
    (2624, 6144),
    (3072, 6144),
    (6144, 2048),
    (6144, 6144),
]
