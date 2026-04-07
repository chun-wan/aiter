"""
FlyDSL GEMM Operations for Gemma4-31B

Replaces:
  - torch._scaled_mm (ROCmFP8ScaledMMLinearKernel) for FP8 GEMM
  - torch.mm (hipBLASLt) for BF16 GEMM

FlyDSL has existing preshuffle_gemm.py that can be adapted.

Gemma4 GEMM shapes (TP=1):
  Gate+Up:     N=43008, K=5376  (2x intermediate_size fused)
  Down:        N=5376,  K=21504
  QKV sliding: N=16384, K=5376
  QKV full:    N=20480, K=5376
  O proj:      N=5376,  K=8192
  LM head:     N=262144, K=5376

MFMA instructions:
  BF16: mfma_f32_16x16x16bf16_1k (16x16 output, K=16)
  FP8:  mfma_f32_16x16x32_fp8_fp8 (16x16 output, K=32)

Tiling for preshuffle GEMM:
  tile_m=128, tile_n=128, tile_k=64 (typical for MI300X)
  LDS: 2-stage pipeline (A + B tiles double-buffered)
  Shared memory: ~64KB for gfx942
"""

import torch


def fp8_scaled_gemm_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Reference FP8 scaled GEMM: output = (A * a_scale) @ (B * b_scale)

    FlyDSL kernel would use:
    - mfma_f32_16x16x32_fp8_fp8 for FP8 matrix multiply
    - F32 accumulator for precision
    - Per-channel/per-tensor scale application
    - buffer_load for coalesced A/B tile loads
    - SmemAllocator for LDS staging (double-buffer)
    """
    a_f32 = a.float() * a_scale.float()
    b_f32 = b.float() * b_scale.float()
    return (a_f32 @ b_f32.T).to(out_dtype)


def bf16_gemm_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Reference BF16 GEMM.

    FlyDSL kernel would use:
    - mfma_f32_16x16x16bf16_1k for BF16 matrix multiply
    - Preshuffle layout for B matrix (weight)
    - F32 accumulator
    - 2-stage LDS pipeline for latency hiding
    """
    return (a.float() @ b.float().T).to(out_dtype)


# Gemma4 GEMM shape configurations for FlyDSL compilation
GEMMA4_GEMM_CONFIGS = {
    'gate_up_tp1': {'M_range': (1, 4096), 'N': 43008, 'K': 5376, 'transA': 'N', 'transB': 'N'},
    'down_tp1': {'M_range': (1, 4096), 'N': 5376, 'K': 21504, 'transA': 'N', 'transB': 'N'},
    'qkv_sliding_tp1': {'M_range': (1, 4096), 'N': 16384, 'K': 5376, 'transA': 'T', 'transB': 'N'},
    'qkv_full_tp1': {'M_range': (1, 4096), 'N': 20480, 'K': 5376, 'transA': 'T', 'transB': 'N'},
    'o_proj_tp1': {'M_range': (1, 4096), 'N': 5376, 'K': 8192, 'transA': 'T', 'transB': 'N'},
    'gate_up_tp8': {'M_range': (1, 4096), 'N': 5376, 'K': 672, 'transA': 'N', 'transB': 'N'},
    'down_tp8': {'M_range': (1, 4096), 'N': 672, 'K': 2688, 'transA': 'N', 'transB': 'N'},
}
