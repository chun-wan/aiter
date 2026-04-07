"""
Correctness test for FlyDSL prefill flash attention vs PyTorch reference.
"""

import torch
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from flydsl_attention.gemma4_prefill_attention import FlyDSLPrefillAttention


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("seq_len", [64, 256, 1024])
@pytest.mark.parametrize("is_causal", [True, False])
def test_prefill_reference_vs_pytorch(head_dim, seq_len, is_causal):
    batch_size = 2
    num_heads = 32
    sm_scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    seq_lens = torch.tensor([seq_len] * batch_size, device='cuda', dtype=torch.int32)

    attn = FlyDSLPrefillAttention(head_dim=head_dim, is_causal=is_causal)
    output = attn.forward(q, k, v, seq_lens, sm_scale)

    # Compare with PyTorch SDPA
    q_pt = q.transpose(1, 2).float()
    k_pt = k.transpose(1, 2).float()
    v_pt = v.transpose(1, 2).float()
    ref = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=is_causal)
    ref = ref.transpose(1, 2).to(torch.bfloat16)

    max_diff = (output.float() - ref.float()).abs().max().item()
    mean_diff = (output.float() - ref.float()).abs().mean().item()
    print(f"head_dim={head_dim} seq_len={seq_len} causal={is_causal}: "
          f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")

    assert max_diff < 0.05, f"Max diff too large: {max_diff}"


@pytest.mark.parametrize("window_size", [256, 1024])
def test_prefill_sliding_window(window_size):
    head_dim = 256
    batch_size = 1
    seq_len = 2048
    num_heads = 32
    sm_scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    seq_lens = torch.tensor([seq_len] * batch_size, device='cuda', dtype=torch.int32)

    attn = FlyDSLPrefillAttention(head_dim=head_dim, is_causal=True)
    output = attn.forward(q, k, v, seq_lens, sm_scale, sliding_window=window_size)

    assert output.shape == q.shape
    assert not torch.isnan(output).any()
    print(f"PASS: prefill sliding_window={window_size}")


def test_prefill_bidirectional():
    """Test bidirectional attention for Gemma4 vision tokens."""
    head_dim = 256
    batch_size = 1
    seq_len = 128
    num_heads = 32
    sm_scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    seq_lens = torch.tensor([seq_len] * batch_size, device='cuda', dtype=torch.int32)

    attn = FlyDSLPrefillAttention(head_dim=head_dim, is_causal=True)
    output = attn.forward(q, k, v, seq_lens, sm_scale, is_bidirectional=True)

    q_pt = q.transpose(1, 2).float()
    k_pt = k.transpose(1, 2).float()
    v_pt = v.transpose(1, 2).float()
    ref = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=False)
    ref = ref.transpose(1, 2).to(torch.bfloat16)

    max_diff = (output.float() - ref.float()).abs().max().item()
    print(f"Bidirectional: max_diff={max_diff:.6f}")
    assert max_diff < 0.05, f"Bidirectional attention max diff too large: {max_diff}"


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_prefill_reference_vs_pytorch(256, 64, True)
        test_prefill_reference_vs_pytorch(256, 64, False)
        test_prefill_reference_vs_pytorch(512, 64, True)
        test_prefill_sliding_window(1024)
        test_prefill_bidirectional()
        print("All prefill tests passed!")
    else:
        print("CUDA not available, skipping tests")
