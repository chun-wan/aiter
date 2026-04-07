"""
FlyDSL GPU Kernel implementations for Gemma4-31B.

Organized by priority:
  P0: Attention kernels (decode, prefill, cache, merge)
  P1: GEMM kernels (FP8 preshuffle, BF16 preshuffle)
  P2: Utility kernels (RMSNorm, SiLU*Mul, RoPE, Softmax)
"""
