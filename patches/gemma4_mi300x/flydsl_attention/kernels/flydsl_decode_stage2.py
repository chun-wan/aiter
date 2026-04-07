"""
FlyDSL Decode Attention Stage 2 — Cross-Split Softmax Reduction

Replaces: triton_decode_attention.py _fwd_kernel_stage2 (@triton.jit, ~60 lines)

Stage 2 reduces partial (m, l, O) results from Stage 1 across NUM_KV_SPLITS.

Algorithm:
  1. Load all partial (m_i, l_i, O_i) for i in [0, NUM_KV_SPLITS)
  2. Find global max: m_global = max(m_0, m_1, ..., m_{S-1})
  3. For each split i:
     correction_i = exp2((m_i - m_global) * LOG2E)
     l_corrected_i = correction_i * l_i
     O_corrected_i = correction_i * O_i
  4. l_total = sum(l_corrected_i)
     O_total = sum(O_corrected_i)
  5. Output = O_total / l_total

FlyDSL Implementation:
  - rocdl.exp2 for correction factors
  - rocdl.ds_bpermute for warp-level max/sum reduction across splits
  - buffer_ops.buffer_load/store for Att_Out and Output tensors

Thread mapping:
  grid = (batch, num_heads)
  block = (128,) — threads cooperate on HEAD_DIM elements
  Each thread handles HEAD_DIM/128 elements of the output vector

FlyDSL @flyc.kernel pseudocode:
```python
@flyc.kernel
def decode_stage2_kernel(
    Att_Out: fx.Tensor,    # [batch, num_heads, num_splits, 2+head_dim]
    Output: fx.Tensor,     # [batch, num_heads, head_dim]
    B_Seqlen: fx.Tensor,
    NUM_KV_SPLITS: fx.Constexpr[int],
    HEAD_DIM: fx.Constexpr[int],
):
    tid = gpu.thread_idx.x
    bid_batch = gpu.block_idx.x
    bid_head = gpu.block_idx.y

    # -- Load all partial results --
    att_rsrc = buffer_ops.create_buffer_resource(Att_Out)
    m_vals = []  # F32 array [NUM_KV_SPLITS]
    l_vals = []
    o_vals = []  # each is F32 vec of HEAD_DIM/128 elements

    for s in range_constexpr(NUM_KV_SPLITS):
        base = compute_att_out_offset(bid_batch, bid_head, s)
        m_vals[s] = buffer_ops.buffer_load(att_rsrc, base, vec_width=1)
        l_vals[s] = buffer_ops.buffer_load(att_rsrc, base + 4, vec_width=1)
        o_vals[s] = buffer_ops.buffer_load(att_rsrc, base + 8 + tid*4*2, vec_width=4)

    # -- Find global max --
    m_global = m_vals[0]
    for s in range_constexpr(NUM_KV_SPLITS - 1):
        m_global = arith.maximumf(m_global, m_vals[s+1])

    # -- Correct and sum --
    l_total = arith.constant(0.0, T.f32)
    o_total = zeros(vec_width, T.f32)

    for s in range_constexpr(NUM_KV_SPLITS):
        correction = rocdl.exp2(T.f32, (m_vals[s] - m_global) * LOG2E)
        l_corrected = correction * l_vals[s]
        o_corrected = correction * o_vals[s]
        l_total = l_total + l_corrected
        o_total = o_total + o_corrected

    # -- Normalize --
    rcp_l = rocdl.rcp(T.f32, l_total)  # hardware reciprocal
    o_final = o_total * rcp_l

    # -- Store output --
    out_rsrc = buffer_ops.create_buffer_resource(Output)
    out_offset = (bid_batch * num_heads * HEAD_DIM + bid_head * HEAD_DIM + tid * 4) * 2
    buffer_ops.buffer_store(convert_f32_to_bf16(o_final), out_rsrc, out_offset)
```
"""

import torch
import math

LOG2E = math.log2(math.e)


class FlyDSLDecodeStage2:
    """Decode attention stage 2: cross-split softmax reduction."""

    def __init__(self, head_dim: int, num_kv_splits: int = 8):
        self.head_dim = head_dim
        self.num_kv_splits = num_kv_splits

    def forward(self, partial_results, seq_lens):
        """
        Reduce partial attention results across KV splits.

        partial_results: [batch, num_heads, num_splits, 2 + head_dim]
          [:,:,:,0] = m (partial max)
          [:,:,:,1] = l (partial sum)
          [:,:,:,2:] = O (partial weighted output)
        """
        batch, num_heads, num_splits, _ = partial_results.shape
        head_dim = partial_results.shape[-1] - 2
        output = torch.empty(batch, num_heads, head_dim,
                            device=partial_results.device, dtype=torch.bfloat16)

        m_vals = partial_results[:, :, :, 0]  # [B, H, S]
        l_vals = partial_results[:, :, :, 1]
        o_vals = partial_results[:, :, :, 2:]  # [B, H, S, D]

        m_global = m_vals.max(dim=-1, keepdim=True).values  # [B, H, 1]
        corrections = torch.exp2((m_vals - m_global) * LOG2E)
        l_corrected = (corrections * l_vals).sum(dim=-1, keepdim=True)
        o_corrected = (corrections.unsqueeze(-1) * o_vals).sum(dim=-2)
        output = (o_corrected / l_corrected).to(torch.bfloat16)

        return output
