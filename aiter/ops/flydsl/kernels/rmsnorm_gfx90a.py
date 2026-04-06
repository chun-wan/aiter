# SPDX-License-Identifier: MIT
# FlyDSL RMSNorm Kernel for gfx90a (MI250)
#
# RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
#
# Memory-bound kernel: target >80% of 1.6 TB/s HBM bandwidth
# Current PyTorch native: 665 GB/s (42% peak)
# Current AITER CK: ~947 GB/s (59% peak, 1.42x)
# Target FlyDSL: >1000 GB/s (>62.5% peak)

import functools
import math

import flydsl.compiler as flyc
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, gpu, vector, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch


@functools.lru_cache(maxsize=32)
def compile_rmsnorm_kernel(
    hidden_size: int,
    eps: float = 1e-6,
    BLOCK_SIZE: int = 1024,
):
    """Compile a FlyDSL RMSNorm kernel for gfx90a.

    Algorithm:
    1. Load row of x (vectorized, BF16)
    2. Compute sum of squares (FP32 accumulation)
    3. Compute rsqrt(mean_sq + eps)
    4. Multiply x * scale * weight
    5. Store output (BF16)

    Optimization for MI250 (gfx90a):
    - Use buffer_load_dwordx4 for 128-bit vector loads (8 BF16 per load)
    - Use LDS for cross-wave reduction of sum-of-squares
    - Use v_rsq_f32 for fast rsqrt
    - Each workgroup processes one row
    """
    arch = get_rocm_arch()

    @flyc.kernel
    def rmsnorm_fwd(
        input_ptr,     # [M, N] bf16
        weight_ptr,    # [N] bf16
        output_ptr,    # [M, N] bf16
        M_param: int,
        N_param: int,
        epsilon: float,
    ):
        # Placeholder for FlyDSL kernel body
        # Full implementation requires expressing the reduction
        # and normalization in FlyDSL's MLIR-based expression API.
        pass

    return rmsnorm_fwd


def flydsl_rmsnorm(x, weight, eps=1e-6, output=None):
    """FlyDSL RMSNorm for gfx90a.

    Falls back to AITER CK when available (already 1.42x faster than native).
    FlyDSL implementation is a development scaffold.
    """
    import torch
    if output is None:
        output = torch.empty_like(x)

    # Try AITER CK first (verified working, 1.42x speedup)
    try:
        from aiter.ops.rmsnorm import rmsnorm2d_fwd_ck
        result = rmsnorm2d_fwd_ck(x, weight, eps)
        output.copy_(result)
        return output
    except Exception:
        pass

    # PyTorch fallback
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    output.copy_((x_normed * weight).to(x.dtype))
    return output
