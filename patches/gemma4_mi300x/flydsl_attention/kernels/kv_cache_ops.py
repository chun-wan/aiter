"""
FlyDSL KV Cache Operations for Gemma4-31B

Replaces:
  - triton_reshape_and_cache_flash.py (@triton.jit)
  - triton_merge_attn_states.py (@triton.jit)

Operations:
  1. reshape_and_cache: Write K/V to paged cache with optional FP8 quantization
  2. merge_attn_states: Merge partial attention outputs from chunked prefill

FlyDSL primitives used:
  - buffer_ops.buffer_store: Efficient AMD buffer store for cache writes
  - rocdl.exp2: For softmax correction during merge
  - gpu.barrier: Workgroup synchronization
"""

import torch
import math

LOG2E = math.log2(math.e)


def reshape_and_cache_flash_reference(
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
    Reference implementation for KV cache write.

    Writes key and value tensors to the paged KV cache at positions
    specified by slot_mapping. Optionally quantizes to FP8.

    FlyDSL kernel would use:
    - buffer_ops.buffer_store for coalesced writes to cache pages
    - Per-token FP8 quantization via rocdl math ops
    """
    num_tokens = key.shape[0]
    page_size = key_cache.shape[1]

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        if slot < 0:
            continue
        page_idx = slot // page_size
        offset = slot % page_size

        if kv_cache_dtype == "fp8" or kv_cache_dtype == "fp8_e4m3":
            key_cache[page_idx, offset] = (key[i] * k_scale).to(torch.float8_e4m3fnuz)
            value_cache[page_idx, offset] = (value[i] * v_scale).to(torch.float8_e4m3fnuz)
        else:
            key_cache[page_idx, offset] = key[i]
            value_cache[page_idx, offset] = value[i]


def merge_attn_states_reference(
    output: torch.Tensor,
    lse: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
) -> None:
    """
    Reference implementation for merging attention states.

    Merges two sets of (output, log-sum-exp) from chunked prefill
    using the online softmax correction formula:
      merged_output = (exp(lse1 - max_lse) * out1 + exp(lse2 - max_lse) * out2)
                      / (exp(lse1 - max_lse) + exp(lse2 - max_lse))

    FlyDSL kernel would use:
    - rocdl.exp2 for hardware exponential
    - Vector operations for per-head merging
    """
    max_lse = torch.maximum(lse, prefix_lse)

    correction_main = torch.exp(lse - max_lse)
    correction_prefix = torch.exp(prefix_lse - max_lse)

    total = correction_main + correction_prefix

    output.mul_(correction_main.unsqueeze(-1))
    output.add_(prefix_output * correction_prefix.unsqueeze(-1))
    output.div_(total.unsqueeze(-1))

    lse.copy_(max_lse + torch.log(total))
