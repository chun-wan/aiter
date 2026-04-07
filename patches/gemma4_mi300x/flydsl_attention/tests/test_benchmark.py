"""
Benchmark FlyDSL attention vs Triton attention.

Measures latency and throughput for both decode and prefill paths.
"""

import time
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from flydsl_attention.gemma4_decode_attention import FlyDSLDecodeAttention
from flydsl_attention.gemma4_prefill_attention import FlyDSLPrefillAttention


def benchmark_decode(head_dim: int, batch_size: int, seq_len: int, num_iters: int = 100):
    num_heads = 32
    num_kv_heads = 16 if head_dim == 256 else 4
    kv_group_num = num_heads // num_kv_heads
    page_size = 16
    sm_scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(batch_size, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    max_pages = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * max_pages
    k_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    v_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    req_to_tokens = torch.zeros(batch_size, seq_len, device='cuda', dtype=torch.int32)
    for b in range(batch_size):
        for i in range(seq_len):
            req_to_tokens[b, i] = b * max_pages * page_size + i
    seq_lens = torch.tensor([seq_len] * batch_size, device='cuda', dtype=torch.int32)

    attn = FlyDSLDecodeAttention(head_dim=head_dim, page_size=page_size)

    # Warmup
    for _ in range(5):
        attn.forward(q, k_cache, v_cache, req_to_tokens, seq_lens, sm_scale, kv_group_num)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        attn.forward(q, k_cache, v_cache, req_to_tokens, seq_lens, sm_scale, kv_group_num)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_iters) * 1000
    return avg_ms


def benchmark_prefill(head_dim: int, batch_size: int, seq_len: int, is_causal: bool, num_iters: int = 20):
    num_heads = 32
    sm_scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    seq_lens = torch.tensor([seq_len] * batch_size, device='cuda', dtype=torch.int32)

    attn = FlyDSLPrefillAttention(head_dim=head_dim, is_causal=is_causal)

    # Warmup
    for _ in range(3):
        attn.forward(q, k, v, seq_lens, sm_scale)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        attn.forward(q, k, v, seq_lens, sm_scale)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_iters) * 1000
    return avg_ms


def run_all_benchmarks():
    print("=" * 70)
    print("FlyDSL Attention Benchmark (Reference PyTorch Implementation)")
    print("=" * 70)
    print()

    print("--- Decode Attention ---")
    configs = [
        (256, 1, 128, "sliding, short seq"),
        (256, 1, 1024, "sliding, medium seq"),
        (256, 1, 4096, "sliding, long seq"),
        (256, 32, 1024, "sliding, batched"),
        (512, 1, 1024, "full, medium seq"),
    ]
    for head_dim, batch, seq_len, desc in configs:
        ms = benchmark_decode(head_dim, batch, seq_len)
        print(f"  head_dim={head_dim:3d} batch={batch:2d} seq_len={seq_len:5d} ({desc}): {ms:.2f} ms")

    print()
    print("--- Prefill Attention ---")
    configs = [
        (256, 1, 256, True, "causal, short"),
        (256, 1, 1024, True, "causal, medium"),
        (256, 1, 4096, True, "causal, long"),
        (512, 1, 1024, True, "full head, causal"),
        (256, 1, 256, False, "bidirectional"),
    ]
    for head_dim, batch, seq_len, causal, desc in configs:
        ms = benchmark_prefill(head_dim, batch, seq_len, causal)
        print(f"  head_dim={head_dim:3d} batch={batch:2d} seq_len={seq_len:5d} ({desc}): {ms:.2f} ms")

    print()
    print("NOTE: These are PyTorch reference implementation times.")
    print("FlyDSL kernel implementation should be significantly faster.")
    print("=" * 70)


if __name__ == "__main__":
    if torch.cuda.is_available():
        run_all_benchmarks()
    else:
        print("CUDA not available")
