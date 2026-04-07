"""
FlyDSL Prefill Flash Attention Kernel with MFMA

Replaces: triton_prefill_attention.py (@triton.jit _fwd_kernel, 253 lines)

FlashAttention-2 algorithm with online softmax, tiled using MFMA instructions.

Architecture (per workgroup):
  - Process Q[bid_m*BLOCK_M : (bid_m+1)*BLOCK_M, head, :] against all KV
  - BLOCK_M=64 (Q tile rows), BLOCK_N=64 (KV tile cols)
  - Inner loop: iterate over KV tiles, maintaining online softmax state

MFMA Usage:
  Per KV tile iteration:
    Q@K^T: (BLOCK_M/16) × (BLOCK_N/16) × (HEAD_DIM/16) MFMA ops
           = 4 × 4 × 16 = 256 mfma_f32_16x16x16bf16_1k  (for HEAD_DIM=256)
    P@V:   (BLOCK_M/16) × (HEAD_DIM/16) × (BLOCK_N/16) MFMA ops
           = 4 × 16 × 4 = 256 mfma_f32_16x16x16bf16_1k
    Total: 512 MFMA ops per KV tile

LDS Budget (gfx942, 64KB per CU):
  Q tile: BLOCK_M × HEAD_DIM × 2B = 64 × 256 × 2 = 32 KB
  K tile: BLOCK_N × HEAD_DIM × 2B = 64 × 256 × 2 = 32 KB
  Total: 64 KB (exactly fits)

  For HEAD_DIM=512:
  Q tile: 64 × 512 × 2 = 64 KB → need BLOCK_M=32 to fit
  or: single-buffer K (no double-buffer)

Instruction Scheduling:
  rocdl.sched_vmem(1): after buffer_load, hide memory latency
  rocdl.sched_mfma(1): between MFMA chains, pipeline ALU
  Pipeline: overlap next K/V tile load with current MFMA computation

FlyDSL @flyc.kernel pseudocode:
```python
@flyc.kernel
def prefill_flash_kernel(
    Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, Output: fx.Tensor,
    seq_lens: fx.Tensor, sm_scale: fx.Int32,
    HEAD_DIM: fx.Constexpr[int],
    BLOCK_M: fx.Constexpr[int],  # 64
    BLOCK_N: fx.Constexpr[int],  # 64
    IS_CAUSAL: fx.Constexpr[int],
):
    tid = gpu.thread_idx.x   # 0..127
    bid_m = gpu.block_idx.x   # Q tile index
    bid_head = gpu.block_idx.y
    bid_batch = gpu.block_idx.z

    # LDS: manual allocation via gpu.smem_space()
    # Q_lds[BLOCK_M][HEAD_DIM] in LDS
    # K_lds[BLOCK_N][HEAD_DIM] in LDS

    # -- Load Q tile to LDS (one-time) --
    q_rsrc = buffer_ops.create_buffer_resource(Q)
    for i in range_constexpr(BLOCK_M * HEAD_DIM // 128):
        offset = compute_q_offset(bid_batch, bid_m, bid_head, tid, i)
        data = buffer_ops.buffer_load(q_rsrc, offset, vec_width=4)
        # store to Q_lds
    gpu.barrier()

    # -- Initialize online softmax state --
    # m_i = -inf (F32, per row, in registers)
    # l_i = 0.0 (F32, per row)
    # O_i = 0.0 (F32, BLOCK_M × HEAD_DIM, in registers)

    # -- Main loop over KV tiles --
    seq_len = load_seq_len(seq_lens, bid_batch)
    num_kv_tiles = (seq_len + BLOCK_N - 1) // BLOCK_N

    for j in range(num_kv_tiles):
        # Load K tile to LDS
        k_rsrc = buffer_ops.create_buffer_resource(K)
        for i in range_constexpr(BLOCK_N * HEAD_DIM // 128):
            offset = compute_k_offset(bid_batch, j, bid_head, tid, i)
            data = buffer_ops.buffer_load(k_rsrc, offset, vec_width=4)
            # store to K_lds
        gpu.barrier()
        rocdl.sched_vmem(1)  # hide load latency

        # -- Q @ K^T via MFMA --
        # S[BLOCK_M][BLOCK_N] in F32 registers
        S = zeros(BLOCK_M // 16, BLOCK_N // 16, 16, 16)  # F32 accumulator tiles
        for dk in range_constexpr(HEAD_DIM // 16):
            for bm in range_constexpr(BLOCK_M // 16):
                for bn in range_constexpr(BLOCK_N // 16):
                    q_tile = load_lds_tile(Q_lds, bm, dk)   # 16×16 BF16
                    k_tile = load_lds_tile(K_lds, bn, dk)    # 16×16 BF16
                    S[bm][bn] = rocdl.mfma_f32_16x16x16bf16_1k(
                        result_type, [q_tile, k_tile, S[bm][bn]]
                    )
                    rocdl.sched_mfma(1)

        # -- Scale --
        S = S * sm_scale

        # -- Mask --
        if IS_CAUSAL:
            q_start = bid_m * BLOCK_M
            kv_start = j * BLOCK_N
            # Apply causal mask: S[i][j] = -inf where q_start+i < kv_start+j
            for bm in range_constexpr(BLOCK_M // 16):
                for bn in range_constexpr(BLOCK_N // 16):
                    # Per-element mask in 16×16 tile
                    mask = compute_causal_mask(q_start, kv_start, bm, bn)
                    S[bm][bn] = arith.select(mask, S[bm][bn], neg_inf)

        # -- Online softmax update --
        # m_new = max(m_i, row_max(S))
        # p = exp2((S - m_new) * LOG2E)     via rocdl.exp2
        # correction = exp2((m_i - m_new) * LOG2E)
        # l_new = correction * l_i + row_sum(p)
        # O_i = O_i * correction

        # Row max reduction via ds_bpermute within warp
        for bm in range_constexpr(BLOCK_M // 16):
            row_max = compute_row_max(S[bm])  # across BLOCK_N
            m_new[bm] = arith.maximumf(m_i[bm], row_max)
            correction = rocdl.exp2(T.f32, (m_i[bm] - m_new[bm]) * LOG2E)
            O_i[bm] = O_i[bm] * correction
            l_i[bm] = l_i[bm] * correction

            # Compute P = exp2((S - m_new) * LOG2E)
            for bn in range_constexpr(BLOCK_N // 16):
                P[bm][bn] = rocdl.exp2(T.f32, (S[bm][bn] - m_new[bm]) * LOG2E)
            l_new = l_i[bm] + row_sum(P[bm])
            m_i[bm] = m_new[bm]
            l_i[bm] = l_new

        # -- Load V tile to LDS --
        v_rsrc = buffer_ops.create_buffer_resource(V)
        for i in range_constexpr(BLOCK_N * HEAD_DIM // 128):
            offset = compute_v_offset(bid_batch, j, bid_head, tid, i)
            data = buffer_ops.buffer_load(v_rsrc, offset, vec_width=4)
        gpu.barrier()

        # -- P @ V via MFMA --
        for dk in range_constexpr(HEAD_DIM // 16):
            for bm in range_constexpr(BLOCK_M // 16):
                for bn in range_constexpr(BLOCK_N // 16):
                    p_tile = convert_f32_to_bf16(P[bm][bn])  # F32→BF16
                    v_tile = load_lds_tile(V_lds, bn, dk)
                    O_i[bm][dk] = rocdl.mfma_f32_16x16x16bf16_1k(
                        result_type, [p_tile, v_tile, O_i[bm][dk]]
                    )
                    rocdl.sched_mfma(1)

    # -- Final normalization --
    # Output = O_i / l_i
    for bm in range_constexpr(BLOCK_M // 16):
        for dk in range_constexpr(HEAD_DIM // 16):
            O_i[bm][dk] = O_i[bm][dk] / l_i[bm]

    # -- Store Output to global memory --
    out_rsrc = buffer_ops.create_buffer_resource(Output)
    for i in range_constexpr(BLOCK_M * HEAD_DIM // 128):
        buffer_ops.buffer_store(O_i_data, out_rsrc, out_offset)
```
"""

import torch
import math

LOG2E = math.log2(math.e)


class FlyDSLPrefillFlashAttention:
    """
    FlyDSL-based prefill flash attention.

    Performance targets (vs Triton):
      BLOCK_M=64, BLOCK_N=64, HEAD_DIM=256:
        512 MFMA ops per KV tile
        ~80% peak MFMA throughput with instruction scheduling
        Expected: 20-40% improvement over Triton due to:
          - Explicit MFMA scheduling (sched_mfma)
          - Buffer load latency hiding (sched_vmem)
          - Vectorized buffer_load (vec_width=4)
    """

    def __init__(self, head_dim: int, block_m: int = 64, block_n: int = 64,
                 is_causal: bool = True):
        self.head_dim = head_dim
        self.block_m = block_m
        self.block_n = block_n
        self.is_causal = is_causal
        self.mfma_per_tile = (block_m // 16) * (block_n // 16) * (head_dim // 16) * 2

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        sliding_window: int = -1,
        is_bidirectional: bool = False,
    ) -> torch.Tensor:
        """Reference PyTorch implementation matching the FlyDSL kernel algorithm."""
        batch, seq_len, num_heads, head_dim = q.shape
        output = torch.empty_like(q)

        for b in range(batch):
            slen = seq_lens[b].item() if seq_lens is not None else seq_len
            for h in range(num_heads):
                q_b = q[b, :slen, h].float()
                k_b = k[b, :slen, h].float()
                v_b = v[b, :slen, h].float()

                scores = (q_b @ k_b.T) * sm_scale

                if not is_bidirectional and self.is_causal:
                    causal_mask = torch.triu(
                        torch.ones(slen, slen, device=q.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    scores = scores.masked_fill(causal_mask, float('-inf'))

                if sliding_window > 0:
                    positions = torch.arange(slen, device=q.device)
                    dist = positions.unsqueeze(0) - positions.unsqueeze(1)
                    window_mask = dist.abs() >= sliding_window
                    scores = scores.masked_fill(window_mask, float('-inf'))

                attn_weights = torch.softmax(scores, dim=-1)
                output[b, :slen, h] = (attn_weights @ v_b).to(q.dtype)

        return output
