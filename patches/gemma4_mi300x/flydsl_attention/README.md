# FlyDSL Attention Kernels for Gemma4-31B

## Status

| Component | Status | Description |
|-----------|--------|-------------|
| Decode reference impl | Done | PyTorch reference with paged KV, GQA, sliding window |
| Prefill reference impl | Done | PyTorch reference with causal, bidirectional, sliding |
| FlyDSL decode kernel | Skeleton | `@flyc.kernel` structure defined, needs MFMA body |
| FlyDSL prefill kernel | Skeleton | `@flyc.kernel` structure defined, needs MFMA body |
| vLLM backend integration | Done | `FlyDSLAttentionBackend` with dual head_dim dispatch |
| Correctness tests | Passing | max_diff < 0.004 vs PyTorch SDPA |
| Benchmarks | Baseline | Reference PyTorch times recorded |

## Architecture

```
                    ┌─────────────────────────┐
                    │  vLLM Attention Layer    │
                    │  --attention-backend     │
                    │     FLYDSL_ATTN          │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │  FlyDSLAttentionBackend  │
                    │  gemma4_attention_       │
                    │  backend.py              │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
    ┌─────────▼─────────┐        ┌──────────▼─────────┐
    │  Decode Paged      │        │  Prefill Flash      │
    │  Attention         │        │  Attention          │
    │  (2-stage split-KV)│        │  (online softmax)   │
    └─────────┬─────────┘        └──────────┬──────────┘
              │                             │
    ┌─────────▼─────────┐        ┌──────────▼──────────┐
    │  head_dim=256      │        │  Causal mask        │
    │  (sliding window)  │        │  Bidirectional mask │
    │  head_dim=512      │        │  Sliding window mask│
    │  (full attention)  │        │                     │
    └───────────────────┘        └─────────────────────┘
```

## Gemma4 Attention Specifications

| Layer Type | Count | head_dim | num_kv_heads | Mask | RoPE theta |
|-----------|-------|----------|-------------|------|------------|
| Sliding | 50 | 256 | 16 | Causal + window=1024 | 10,000 |
| Full | 10 | 512 | 4 | Causal (full context) | 1,000,000 |
| Vision | varies | 256 | 16 | Bidirectional | 10,000 |

## Next Steps: FlyDSL Kernel Implementation

The `@flyc.kernel` functions in `gemma4_decode_attention.py` and
`gemma4_prefill_attention.py` contain structural skeletons that need
the actual MFMA computation bodies filled in.

Key FlyDSL primitives to use:
- `rocdl.mfma_f32_16x16x16bf16_1k` for Q@K^T and P@V
- `buffer_ops.buffer_load/store` for efficient global memory access
- `SmemAllocator` for LDS tile management
- `rocdl.exp2` for softmax exponential
- `rocdl.ds_bpermute` for warp-level reduction
- `rocdl.sched_mfma/sched_vmem` for instruction scheduling

Reference: FlyDSL preshuffle GEMM kernel (`kernels/preshuffle_gemm.py`)
for MFMA + LDS + buffer ops patterns.

## Running Tests

```bash
# Inside the gemma4-vllm container:
python3 /workspace/flydsl_attention/tests/test_decode_correctness.py
python3 /workspace/flydsl_attention/tests/test_prefill_correctness.py
python3 /workspace/flydsl_attention/tests/test_benchmark.py
```

## Reference Benchmark Baseline (PyTorch)

```
Decode (head_dim=256):
  batch=1  seq=128:  7.08 ms
  batch=1  seq=1024: 6.52 ms
  batch=1  seq=4096: 6.37 ms
  batch=32 seq=1024: 210.69 ms

Prefill (head_dim=256):
  seq=256  causal:   5.02 ms
  seq=1024 causal:   5.07 ms
  seq=4096 causal:   40.83 ms
  seq=256  bidir:    3.92 ms
```

FlyDSL kernel targets: 3-5x faster than PyTorch reference,
competitive with or faster than Triton attention.
