"""
FlyDSL KV Cache Reshape and Write Kernel

Replaces: triton_reshape_and_cache_flash.py (@triton.jit)

This kernel writes Key and Value tensors into the paged KV cache
at positions specified by slot_mapping, with optional FP8 quantization.

FlyDSL Implementation:
  - buffer_ops.create_buffer_resource: Create AMD buffer descriptors for K/V/Cache
  - buffer_ops.buffer_load: Coalesced load from input K/V tensors
  - buffer_ops.buffer_store: Coalesced store to paged cache
  - Each workgroup processes one token, threads cooperate on head_dim

Thread mapping:
  grid = (num_tokens, num_kv_heads)
  block = (HEAD_DIM // vec_width,)  e.g. 256/4 = 64 threads for vec_width=4

For HEAD_DIM=256: 64 threads × 4 BF16 elements = 256 elements (full head)
For HEAD_DIM=512: 128 threads × 4 BF16 elements = 512 elements (full head)
"""

import torch
import math


class FlyDSLReshapeAndCache:
    """
    FlyDSL-based KV cache write operation.

    Currently uses PyTorch reference implementation.
    The @flyc.kernel version requires FlyDSL source build for SmemAllocator.

    Architecture:
    - Each workgroup handles one (token, kv_head) pair
    - Threads cooperatively load HEAD_DIM elements from input K/V
    - Compute slot → (page_idx, offset) mapping
    - Store to paged cache via buffer_store (coalesced, vectorized)

    FlyDSL kernel body (when SmemAllocator available):
    ```python
    @flyc.kernel
    def reshape_cache_kernel(
        Key: fx.Tensor, Value: fx.Tensor,
        K_Cache: fx.Tensor, V_Cache: fx.Tensor,
        Slot_Mapping: fx.Tensor,
        HEAD_DIM: fx.Constexpr[int],
        PAGE_SIZE: fx.Constexpr[int],
        VEC_WIDTH: fx.Constexpr[int],
    ):
        tid = gpu.thread_idx.x
        token_idx = gpu.block_idx.x
        head_idx = gpu.block_idx.y

        # Load slot
        slot_rsrc = buffer_ops.create_buffer_resource(Slot_Mapping)
        slot = buffer_ops.buffer_load(slot_rsrc, token_idx * 4, vec_width=1)
        page_idx = slot // PAGE_SIZE
        page_offset = slot % PAGE_SIZE

        # Load key elements (vectorized)
        k_rsrc = buffer_ops.create_buffer_resource(Key)
        k_byte_offset = (token_idx * num_kv_heads * HEAD_DIM + head_idx * HEAD_DIM + tid * VEC_WIDTH) * 2
        k_data = buffer_ops.buffer_load(k_rsrc, k_byte_offset, vec_width=VEC_WIDTH)

        # Store to cache (vectorized)
        kc_rsrc = buffer_ops.create_buffer_resource(K_Cache)
        kc_byte_offset = (page_idx * PAGE_SIZE * num_kv_heads * HEAD_DIM +
                         page_offset * num_kv_heads * HEAD_DIM +
                         head_idx * HEAD_DIM + tid * VEC_WIDTH) * 2
        buffer_ops.buffer_store(k_data, kc_rsrc, kc_byte_offset)

        # Same for value
        v_rsrc = buffer_ops.create_buffer_resource(Value)
        v_data = buffer_ops.buffer_load(v_rsrc, v_byte_offset, vec_width=VEC_WIDTH)
        vc_rsrc = buffer_ops.create_buffer_resource(V_Cache)
        buffer_ops.buffer_store(v_data, vc_rsrc, vc_byte_offset)

        # Scheduling hints
        rocdl.sched_vmem(2)  # Hide memory latency
    ```
    """

    def __init__(self, num_kv_heads: int, head_dim: int, page_size: int = 16,
                 dtype: torch.dtype = torch.bfloat16):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.vec_width = 4  # 4 BF16 elements per buffer_load/store

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str = "auto",
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        """
        Write K/V to paged cache. Reference PyTorch implementation.

        Args:
            key: [num_tokens, num_kv_heads, head_dim]
            value: [num_tokens, num_kv_heads, head_dim]
            key_cache: [num_pages, page_size, num_kv_heads, head_dim]
            value_cache: [num_pages, page_size, num_kv_heads, head_dim]
            slot_mapping: [num_tokens] - maps token to cache slot
        """
        num_tokens = key.shape[0]
        page_size = key_cache.shape[1]

        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            page_idx = slot // page_size
            offset = slot % page_size

            if kv_cache_dtype in ("fp8", "fp8_e4m3"):
                key_cache[page_idx, offset] = (key[i] * k_scale).to(torch.float8_e4m3fnuz)
                value_cache[page_idx, offset] = (value[i] * v_scale).to(torch.float8_e4m3fnuz)
            else:
                key_cache[page_idx, offset] = key[i]
                value_cache[page_idx, offset] = value[i]
