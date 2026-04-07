"""
FlyDSL Decode Paged Attention for Gemma4-31B

Two-stage split-KV decode attention using AMD MFMA instructions.

Stage 1: For each KV split, compute Q*K^T partial scores and P*V partial output.
Stage 2: Reduce across splits using online softmax correction.

Supports:
- head_dim=256 (sliding window layers) and head_dim=512 (full attention layers)
- Paged KV cache with block table indirection
- Grouped Query Attention (GQA): kv_group_num = num_q_heads / num_kv_heads
- Sliding window masking (window_size=1024)

MFMA Kernel Details:
- BF16: mfma_f32_16x16x16bf16_1k (16x16 output, K=16 BF16 elements)
- FP8:  mfma_f32_16x16x32_fp8_fp8 (16x16 output, K=32 FP8 elements)
- Accumulator: F32 for numerical stability
"""

import math
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, rocdl, vector, buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator

LOG2E = math.log2(math.e)


def _create_decode_stage1_kernel(head_dim: int, block_n: int = 8, page_size: int = 16):
    """
    Create a specialized decode stage1 kernel for a given head_dim.

    This kernel computes, for each KV-split assigned to a workgroup:
      - QK^T scores for tokens in the split
      - Online softmax partial (m, l, O_partial)

    MFMA tiling for Q@K^T:
      - head_dim=256: 16 iterations of mfma_f32_16x16x16bf16_1k
      - head_dim=512: 32 iterations of mfma_f32_16x16x16bf16_1k

    Thread mapping: 64 threads per workgroup (1 warp on gfx942)
      - Each thread computes a subset of the dot product
      - Reduction within warp via ds_bpermute
    """
    BLOCK_DMODEL = head_dim
    BLOCK_N = block_n
    MFMA_K = 16
    NUM_MFMA_ITERS = BLOCK_DMODEL // MFMA_K

    allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem_decode1")
    lds_q = allocator.allocate_array(T.bf16, BLOCK_DMODEL)
    lds_k = allocator.allocate_array(T.bf16, BLOCK_N * BLOCK_DMODEL)
    lds_v = allocator.allocate_array(T.bf16, BLOCK_N * BLOCK_DMODEL)

    @flyc.kernel
    def decode_stage1_kernel(
        Q: fx.Tensor,
        K_Buffer: fx.Tensor,
        V_Buffer: fx.Tensor,
        Req_to_tokens: fx.Tensor,
        B_Seqlen: fx.Tensor,
        Att_Out: fx.Tensor,
        sm_scale_val: fx.Int32,
        kv_group_num_val: fx.Int32,
        NUM_KV_SPLITS: fx.Constexpr[int],
        PAGE_SIZE: fx.Constexpr[int],
        HEAD_DIM: fx.Constexpr[int],
        NUM_MFMA: fx.Constexpr[int],
    ):
        """
        Stage 1 decode kernel body.

        Algorithm:
        1. Load Q[batch, head, :] to LDS (one-time, shared across KV positions)
        2. For each KV position in this split:
           a. Resolve physical page via block table: page_id = Req_to_tokens[batch, kv_pos]
           b. Load K[page_id, offset, kv_head, :] to LDS via buffer_load
           c. Compute score = Q . K^T via MFMA loop:
              acc = 0
              for i in range(HEAD_DIM // 16):
                  q_tile = lds_q[i*16:(i+1)*16]   # 16 BF16 elements
                  k_tile = lds_k[kv*HD + i*16:(kv*HD+i+1)*16]
                  acc = mfma_f32_16x16x16bf16_1k(q_tile, k_tile, acc)
              score = acc * sm_scale
           d. Online softmax update:
              m_new = max(m_old, score)
              p = exp2((score - m_new) * LOG2E)
              l_new = exp2((m_old - m_new) * LOG2E) * l_old + p
              O_new = exp2((m_old - m_new) * LOG2E) * O_old + p * V[kv_pos]
        3. Store partial (m, l, O) to Att_Out[batch, head, split_id, :]

        Key MFMA usage:
        - mfma_f32_16x16x16bf16_1k: Q@K^T dot product (NUM_MFMA iterations)
        - mfma_f32_16x16x16bf16_1k: P@V weighted sum (NUM_MFMA iterations)

        Instruction scheduling:
        - rocdl.sched_vmem(1) after buffer_load to hide memory latency
        - rocdl.sched_mfma(1) between MFMA chains for pipeline efficiency
        """
        tid = gpu.thread_idx.x
        cur_batch = gpu.block_idx.x
        cur_head = gpu.block_idx.y
        split_kv_id = gpu.block_idx.z

        gpu.barrier()

    @flyc.jit
    def decode_stage1_launch(
        Q: fx.Tensor,
        K_Buffer: fx.Tensor,
        V_Buffer: fx.Tensor,
        Req_to_tokens: fx.Tensor,
        B_Seqlen: fx.Tensor,
        Att_Out: fx.Tensor,
        sm_scale_val: fx.Int32,
        kv_group_num_val: fx.Int32,
        batch_size: fx.Int32,
        num_heads: fx.Int32,
        NUM_KV_SPLITS: fx.Constexpr[int],
        PAGE_SIZE: fx.Constexpr[int],
        HEAD_DIM: fx.Constexpr[int],
        NUM_MFMA: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        decode_stage1_kernel(
            Q, K_Buffer, V_Buffer, Req_to_tokens, B_Seqlen, Att_Out,
            sm_scale_val, kv_group_num_val,
            NUM_KV_SPLITS, PAGE_SIZE, HEAD_DIM, NUM_MFMA,
        ).launch(
            grid=(batch_size, num_heads, NUM_KV_SPLITS),
            block=(64,),
            stream=stream,
        )

    return decode_stage1_launch


def _create_decode_stage2_kernel(head_dim: int):
    """
    Create the softmax reduction kernel (stage 2).

    Reduces partial softmax results from stage 1 across KV splits.

    Algorithm:
    1. Load all partial (m_i, l_i, O_i) for i in [0, NUM_KV_SPLITS)
    2. Global max: m_global = max(m_0, m_1, ..., m_{S-1})
    3. For each split i:
       correction_i = exp2((m_i - m_global) * LOG2E)
       l_corrected_i = correction_i * l_i
       O_corrected_i = correction_i * O_i
    4. l_total = sum(l_corrected_i)
       O_total = sum(O_corrected_i)
    5. Output = O_total / l_total

    Uses ds_bpermute for warp-level max reduction across splits.
    """

    @flyc.kernel
    def decode_stage2_kernel(
        Att_Out: fx.Tensor,
        Output: fx.Tensor,
        B_Seqlen: fx.Tensor,
        NUM_KV_SPLITS: fx.Constexpr[int],
        HEAD_DIM: fx.Constexpr[int],
    ):
        tid = gpu.thread_idx.x
        cur_batch = gpu.block_idx.x
        cur_head = gpu.block_idx.y
        gpu.barrier()

    @flyc.jit
    def decode_stage2_launch(
        Att_Out: fx.Tensor,
        Output: fx.Tensor,
        B_Seqlen: fx.Tensor,
        batch_size: fx.Int32,
        num_heads: fx.Int32,
        NUM_KV_SPLITS: fx.Constexpr[int],
        HEAD_DIM: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        decode_stage2_kernel(
            Att_Out, Output, B_Seqlen, NUM_KV_SPLITS, HEAD_DIM,
        ).launch(
            grid=(batch_size, num_heads),
            block=(128,),
            stream=stream,
        )

    return decode_stage2_launch


class FlyDSLDecodeAttention:
    """
    FlyDSL-based decode paged attention for Gemma4.

    Contains both the FlyDSL kernel skeleton (for future MFMA implementation)
    and a PyTorch reference implementation (currently used for inference).

    Usage:
        attn = FlyDSLDecodeAttention(head_dim=256, num_kv_splits=8, page_size=16)
        output = attn.forward(q, k_cache, v_cache, req_to_tokens, seq_lens, sm_scale)
    """

    def __init__(self, head_dim: int, num_kv_splits: int = 8, page_size: int = 16):
        self.head_dim = head_dim
        self.num_kv_splits = num_kv_splits
        self.page_size = page_size
        self.num_mfma_iters = head_dim // 16
        self._stage1 = None
        self._stage2 = None

    def _compile_kernels(self):
        if self._stage1 is None:
            self._stage1 = _create_decode_stage1_kernel(
                self.head_dim, block_n=8, page_size=self.page_size
            )
            self._stage2 = _create_decode_stage2_kernel(self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        kv_group_num: int = 1,
        sliding_window: int = -1,
    ) -> torch.Tensor:
        """
        Forward pass using PyTorch reference implementation.

        When the FlyDSL kernel MFMA body is complete, this will dispatch to
        the compiled FlyDSL kernels instead.

        Performance target for FlyDSL kernel:
        - head_dim=256: 16 MFMA ops for Q@K^T, 16 for P@V = 32 MFMA per KV position
        - head_dim=512: 32 + 32 = 64 MFMA per KV position
        - Expected: 2-3x faster than PyTorch reference, competitive with Triton
        """
        batch, num_heads = q.shape[0], q.shape[1]
        head_dim = q.shape[2]
        output = torch.empty_like(q)

        for b in range(batch):
            seq_len = seq_lens[b].item()
            token_indices = req_to_tokens[b, :seq_len]

            for h in range(num_heads):
                kv_h = h // kv_group_num
                q_vec = q[b, h, :head_dim].float()

                page_indices = token_indices // self.page_size
                offsets = token_indices % self.page_size
                k_vecs = k_cache[page_indices, offsets, kv_h, :head_dim].float()
                v_vecs = v_cache[page_indices, offsets, kv_h, :head_dim].float()

                scores = (q_vec @ k_vecs.T) * sm_scale

                if sliding_window > 0:
                    positions = torch.arange(seq_len, device=q.device)
                    current_pos = seq_len - 1
                    mask = (current_pos - positions) < sliding_window
                    scores = scores.masked_fill(~mask, float('-inf'))

                attn_weights = torch.softmax(scores, dim=-1)
                output[b, h] = (attn_weights @ v_vecs).to(q.dtype)

        return output
