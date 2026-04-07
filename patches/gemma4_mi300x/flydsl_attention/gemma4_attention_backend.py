"""
FlyDSL Attention Backend for vLLM

Integrates FlyDSL attention kernels as a vLLM attention backend,
handling Gemma4's dual head_dim dispatch.

Registration: --attention-backend FLYDSL_ATTN
"""

import dataclasses
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FlyDSLAttentionMetadata:
    """Metadata for FlyDSL attention dispatch."""
    is_prefill: bool = False
    seq_lens: Optional[torch.Tensor] = None
    req_to_tokens: Optional[torch.Tensor] = None
    max_seq_len: int = 0
    sliding_window: int = -1
    is_bidirectional: bool = False


class FlyDSLAttentionImpl:
    """
    FlyDSL attention implementation.

    Dispatches between decode (paged) and prefill (flash) attention
    based on the request phase. Handles Gemma4's dual head_dim by
    maintaining separate compiled kernels for each dimension.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 scale: float, num_kv_splits: int = 8, page_size: int = 16):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.kv_group_num = num_heads // num_kv_heads
        self.num_kv_splits = num_kv_splits
        self.page_size = page_size

        from .gemma4_decode_attention import FlyDSLDecodeAttention
        from .gemma4_prefill_attention import FlyDSLPrefillAttention

        self.decode_attn = FlyDSLDecodeAttention(
            head_dim=head_dim,
            num_kv_splits=num_kv_splits,
            page_size=page_size,
        )
        self.prefill_attn = FlyDSLPrefillAttention(
            head_dim=head_dim,
            is_causal=True,
        )

        logger.info(
            f"FlyDSL attention: heads={num_heads}, kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, scale={scale:.4f}, "
            f"kv_group={self.kv_group_num}, page_size={page_size}"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: FlyDSLAttentionMetadata,
    ) -> torch.Tensor:
        if attn_metadata.is_prefill:
            return self.prefill_attn.forward(
                query, key, value,
                seq_lens=attn_metadata.seq_lens,
                sm_scale=self.scale,
                sliding_window=attn_metadata.sliding_window,
                is_bidirectional=attn_metadata.is_bidirectional,
            )
        else:
            if kv_cache is None:
                raise ValueError("kv_cache required for decode attention")
            k_cache, v_cache = kv_cache.unbind(dim=0) if kv_cache.dim() == 6 else (kv_cache, kv_cache)
            return self.decode_attn.forward(
                query, k_cache, v_cache,
                req_to_tokens=attn_metadata.req_to_tokens,
                seq_lens=attn_metadata.seq_lens,
                sm_scale=self.scale,
                kv_group_num=self.kv_group_num,
                sliding_window=attn_metadata.sliding_window,
            )


def create_gemma4_attention_dispatch(model_config: dict):
    """
    Create attention implementations for all Gemma4 layer types.

    Gemma4 has two types of attention layers:
    - Sliding window: head_dim=256, num_kv_heads=16, window=1024 (50 layers)
    - Full attention: head_dim=512, num_kv_heads=4 (10 layers)

    Returns a dict mapping layer_type -> FlyDSLAttentionImpl
    """
    num_q_heads = model_config.get('num_attention_heads', 32)
    sliding_kv_heads = model_config.get('num_key_value_heads', 16)
    full_kv_heads = model_config.get('num_global_key_value_heads', 4)
    sliding_head_dim = model_config.get('head_dim', 256)
    full_head_dim = model_config.get('global_head_dim', 512)

    sliding_scale = 1.0 / (sliding_head_dim ** 0.5)
    full_scale = 1.0 / (full_head_dim ** 0.5)

    return {
        'sliding': FlyDSLAttentionImpl(
            num_heads=num_q_heads,
            num_kv_heads=sliding_kv_heads,
            head_dim=sliding_head_dim,
            scale=sliding_scale,
        ),
        'full': FlyDSLAttentionImpl(
            num_heads=num_q_heads,
            num_kv_heads=full_kv_heads,
            head_dim=full_head_dim,
            scale=full_scale,
        ),
    }
