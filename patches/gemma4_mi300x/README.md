# Gemma4-31B MI300X — AITER Optimizations

## Overview
AITER kernel optimizations and FlyDSL attention kernel development for Gemma4-31B on MI300X (gfx942).

## Contents

### FlyDSL Attention Kernels (`flydsl_attention/`)
- `gemma4_prefill_attention.py` — Prefill FlashAttention-2 with dual head_dim (256/512) support
- `gemma4_decode_attention.py` — Decode paged attention with split-KV 2-stage
- `gemma4_attention_backend.py` — vLLM attention backend integration
- `kernels/` — FlyDSL kernel implementations (reshape_cache, GEMM, utility ops)
- `integration/aiter_allreduce.py` — AITER AllReduce Manager (quick_reduce INT4, custom ASM)
- `tests/` — Correctness and benchmark tests

### AITER AllReduce Integration (`aiter_allreduce.py`)
AllReduce Manager with three strategies:
- `qr` — AITER quick_allreduce with INT4 compression
- `asm` — AITER custom_allreduce (ASM kernel)
- `fused` — Fused AllReduce + RMSNorm

## AITER Environment Variables (MI300X)
```bash
VLLM_ROCM_USE_AITER=1            # Master switch
VLLM_ROCM_USE_AITER_RMSNORM=1    # Fused RMSNorm kernel
VLLM_ROCM_USE_AITER_LINEAR=1     # FP8 GEMM dispatch
VLLM_ROCM_USE_AITER_MHA=1        # Multi-head attention
```

## Known Issues
- AITER #1702: `causal=True` hardcoded in `rocm_aiter_unified_attn.py` blocks Gemma4 bidirectional attention
  - Fix: See `chun-wan/vllm:gemma4_mi300` patches
- `fuse_allreduce_rms` and `fuse_gemm_comms` compilation passes not available in ROCm vLLM 0.18.x

## Performance Impact
| Optimization | Impact (con=40) |
|---|---|
| AITER full enablement | +15.7% throughput |
| + hipBLASLt tuning | +1% additional |
| + N-gram spec decode (5) | +24.3% additional |
| **Combined** | **+42.6% vs baseline** |
