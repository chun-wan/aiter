"""
FlyDSL Attention Kernels for Gemma4-31B on AMD MI300X

Replaces vLLM's Triton attention backend with FlyDSL-based kernels
using MFMA instructions and AMD buffer operations.

Supports:
- Decode paged attention (split-KV, 2-stage)
- Prefill flash attention (online softmax)
- Gemma4 dual head_dim (256 for sliding, 512 for full attention)
- Bidirectional attention for vision tokens
- Sliding window attention (window=1024)
"""

__version__ = "0.1.0"
