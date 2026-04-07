"""
FlyDSL Decode Attention Stage 1 — Q*K^T + Online Softmax + P*V

Replaces: triton_decode_attention.py _fwd_kernel_stage1 (@triton.jit, ~200 lines)

Split-KV decode attention Stage 1:
  Each workgroup processes one (batch, head, kv_split) combination.
  Computes partial (m, l, O) for the KV tokens in this split.
  Stage 2 will reduce across splits.

MFMA for Q·K^T (dot product):
  - Q is a single vector [1, HEAD_DIM] (current decode token)
  - K is a block [BLOCK_N, HEAD_DIM] from paged KV cache
  - For HEAD_DIM=256: 16 × mfma_f32_16x16x16bf16_1k per KV position
  - For HEAD_DIM=512: 32 × mfma_f32_16x16x16bf16_1k per KV position

Paged KV Cache Access:
  - Block table: Req_to_tokens[batch, kv_pos] → physical_token_idx
  - physical_token_idx → (page_idx, page_offset)
  - K = K_Cache[page_idx, page_offset, kv_head, :]
  - buffer_load with indirect addressing

GQA (Grouped Query Attention):
  - kv_group_num = num_q_heads / num_kv_heads
  - cur_kv_head = cur_head // kv_group_num
  - Multiple Q heads share same K/V head

FlyDSL @flyc.kernel pseudocode:
```python
@flyc.kernel
def decode_stage1_kernel(
    Q: fx.Tensor,           # [batch, num_heads, head_dim]
    K_Cache: fx.Tensor,     # [num_pages, page_size, num_kv_heads, head_dim]
    V_Cache: fx.Tensor,
    Req_to_tokens: fx.Tensor,  # [batch, max_seq_len]
    B_Seqlen: fx.Tensor,    # [batch]
    Att_Out: fx.Tensor,     # [batch, num_heads, num_kv_splits, 1+1+head_dim]
    sm_scale: fx.Int32,
    kv_group_num: fx.Int32,
    HEAD_DIM: fx.Constexpr[int],    # 256 or 512
    BLOCK_N: fx.Constexpr[int],     # 8 (KV positions per iteration)
    NUM_KV_SPLITS: fx.Constexpr[int],
    PAGE_SIZE: fx.Constexpr[int],
):
    tid = gpu.thread_idx.x    # 0..63 (1 warp)
    bid_batch = gpu.block_idx.x
    bid_head = gpu.block_idx.y
    bid_split = gpu.block_idx.z

    kv_head = bid_head // kv_group_num

    # -- Load Q vector to registers --
    q_rsrc = buffer_ops.create_buffer_resource(Q)
    q_offset = (bid_batch * num_heads * HEAD_DIM + bid_head * HEAD_DIM + tid * 4) * 2
    q_vec = buffer_ops.buffer_load(q_rsrc, q_offset, vec_width=4)
    # Each thread holds 4 BF16 elements of Q
    # 64 threads × 4 = 256 elements (full HEAD_DIM=256)

    # -- Compute KV range for this split --
    seq_len = load_scalar(B_Seqlen, bid_batch)
    split_size = (seq_len + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS
    kv_start = bid_split * split_size
    kv_end = min(kv_start + split_size, seq_len)

    # -- Initialize online softmax --
    m_i = arith.constant(-inf, T.f32)     # running max
    l_i = arith.constant(0.0, T.f32)      # running sum
    o_i = zeros(HEAD_DIM // 4, T.f32_vec4) # running output

    # -- Loop over KV positions in this split --
    for kv_pos_base in range(kv_start, kv_end, BLOCK_N):
        for kv_offset in range_constexpr(BLOCK_N):
            kv_pos = kv_pos_base + kv_offset
            if kv_pos >= kv_end:
                continue

            # Resolve physical page from block table
            req_rsrc = buffer_ops.create_buffer_resource(Req_to_tokens)
            phys_token = buffer_ops.buffer_load(req_rsrc, (bid_batch * max_seq + kv_pos) * 4, vec_width=1)
            page_idx = phys_token // PAGE_SIZE
            page_offset = phys_token % PAGE_SIZE

            # Load K vector from paged cache
            k_rsrc = buffer_ops.create_buffer_resource(K_Cache)
            k_offset = (page_idx * PAGE_SIZE * num_kv_heads * HEAD_DIM +
                       page_offset * num_kv_heads * HEAD_DIM +
                       kv_head * HEAD_DIM + tid * 4) * 2
            k_vec = buffer_ops.buffer_load(k_rsrc, k_offset, vec_width=4)

            # -- Q · K^T dot product --
            # Each thread computes partial dot of its 4 elements
            partial_dot = dot4(q_vec, k_vec)  # 4-element dot product
            # Warp reduction: sum partial_dot across 64 threads
            score = warp_reduce_sum(partial_dot)  # via ds_bpermute
            score = score * sm_scale

            # -- Online softmax update --
            m_new = arith.maximumf(m_i, score)
            correction = rocdl.exp2(T.f32, (m_i - m_new) * LOG2E)
            p = rocdl.exp2(T.f32, (score - m_new) * LOG2E)
            l_i = correction * l_i + p
            o_i = o_i * correction

            # -- Accumulate P * V --
            v_rsrc = buffer_ops.create_buffer_resource(V_Cache)
            v_vec = buffer_ops.buffer_load(v_rsrc, v_offset, vec_width=4)
            o_i = o_i + p * v_vec

            m_i = m_new

    # -- Store partial (m, l, O) to Att_Out --
    # Att_Out[batch, head, split, 0] = m_i
    # Att_Out[batch, head, split, 1] = l_i
    # Att_Out[batch, head, split, 2:2+HEAD_DIM] = o_i
    out_rsrc = buffer_ops.create_buffer_resource(Att_Out)
    buffer_ops.buffer_store(m_i, out_rsrc, m_offset)
    buffer_ops.buffer_store(l_i, out_rsrc, l_offset)
    buffer_ops.buffer_store(o_i, out_rsrc, o_offset)
```
"""

import torch
import math

LOG2E = math.log2(math.e)


class FlyDSLDecodeStage1:
    """Decode attention stage 1: Q*K^T + online softmax + P*V."""

    def __init__(self, head_dim: int, num_kv_splits: int = 8, page_size: int = 16):
        self.head_dim = head_dim
        self.num_kv_splits = num_kv_splits
        self.page_size = page_size

    def forward(self, q, k_cache, v_cache, req_to_tokens, seq_lens,
                sm_scale, kv_group_num=1, sliding_window=-1):
        """Reference implementation matching the FlyDSL kernel algorithm."""
        batch, num_heads = q.shape[0], q.shape[1]
        head_dim = q.shape[2]
        output = torch.empty_like(q)

        for b in range(batch):
            seq_len = seq_lens[b].item()
            tokens = req_to_tokens[b, :seq_len]

            for h in range(num_heads):
                kv_h = h // kv_group_num
                q_vec = q[b, h].float()

                page_idx = tokens // self.page_size
                offsets = tokens % self.page_size
                k_vecs = k_cache[page_idx, offsets, kv_h].float()
                v_vecs = v_cache[page_idx, offsets, kv_h].float()

                scores = (q_vec @ k_vecs.T) * sm_scale

                if sliding_window > 0:
                    pos = torch.arange(seq_len, device=q.device)
                    mask = (seq_len - 1 - pos) < sliding_window
                    scores = scores.masked_fill(~mask, float('-inf'))

                weights = torch.softmax(scores, dim=-1)
                output[b, h] = (weights @ v_vecs).to(q.dtype)

        return output
