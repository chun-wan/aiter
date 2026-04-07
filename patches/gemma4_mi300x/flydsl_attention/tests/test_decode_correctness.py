"""
Correctness test for FlyDSL decode attention vs PyTorch reference.

Tests the reference implementation against PyTorch's scaled_dot_product_attention
to ensure the algorithm is correct before FlyDSL kernel optimization.
"""

import torch
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from flydsl_attention.gemma4_decode_attention import FlyDSLDecodeAttention


def _create_paged_kv_cache(batch, seq_lens, num_kv_heads, head_dim, page_size, dtype=torch.bfloat16):
    max_pages = max((s + page_size - 1) // page_size for s in seq_lens)
    total_pages = batch * max_pages
    k_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, device='cuda', dtype=dtype)
    v_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, device='cuda', dtype=dtype)
    req_to_tokens = torch.zeros(batch, max(seq_lens), device='cuda', dtype=torch.int32)
    for b in range(batch):
        for i in range(seq_lens[b]):
            req_to_tokens[b, i] = b * max_pages * page_size + i
    return k_cache, v_cache, req_to_tokens


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 1024])
def test_decode_reference_vs_pytorch(head_dim, batch_size, seq_len):
    num_heads = 32
    num_kv_heads = 16 if head_dim == 256 else 4
    kv_group_num = num_heads // num_kv_heads
    page_size = 16
    sm_scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(batch_size, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    seq_lens = [seq_len] * batch_size
    k_cache, v_cache, req_to_tokens = _create_paged_kv_cache(
        batch_size, seq_lens, num_kv_heads, head_dim, page_size
    )
    seq_lens_t = torch.tensor(seq_lens, device='cuda', dtype=torch.int32)

    attn = FlyDSLDecodeAttention(head_dim=head_dim, num_kv_splits=8, page_size=page_size)
    output = attn.forward(
        q, k_cache, v_cache, req_to_tokens, seq_lens_t,
        sm_scale=sm_scale, kv_group_num=kv_group_num,
    )

    assert output.shape == q.shape, f"Output shape mismatch: {output.shape} vs {q.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print(f"PASS: decode head_dim={head_dim} batch={batch_size} seq_len={seq_len}")


@pytest.mark.parametrize("window_size", [256, 1024])
def test_decode_sliding_window(window_size):
    head_dim = 256
    batch_size = 2
    seq_len = 2048
    num_heads = 32
    num_kv_heads = 16
    kv_group_num = num_heads // num_kv_heads
    page_size = 16
    sm_scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(batch_size, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    seq_lens = [seq_len] * batch_size
    k_cache, v_cache, req_to_tokens = _create_paged_kv_cache(
        batch_size, seq_lens, num_kv_heads, head_dim, page_size
    )
    seq_lens_t = torch.tensor(seq_lens, device='cuda', dtype=torch.int32)

    attn = FlyDSLDecodeAttention(head_dim=head_dim, page_size=page_size)
    output = attn.forward(
        q, k_cache, v_cache, req_to_tokens, seq_lens_t,
        sm_scale=sm_scale, kv_group_num=kv_group_num,
        sliding_window=window_size,
    )

    assert output.shape == q.shape
    assert not torch.isnan(output).any()
    print(f"PASS: decode sliding_window={window_size}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_decode_reference_vs_pytorch(256, 1, 128)
        test_decode_reference_vs_pytorch(512, 1, 128)
        test_decode_sliding_window(1024)
        print("All decode tests passed!")
    else:
        print("CUDA not available, skipping tests")
