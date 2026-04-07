"""
FlyDSL Utility Kernel Operations for Gemma4-31B

Replaces:
  - vllm._C.rms_norm (C++ HIP) -> rmsnorm_kernel @flyc.kernel
  - vllm._C.silu_and_mul (C++ HIP) -> silu_mul_kernel @flyc.kernel
  - torch.compile inductor RoPE -> rope_kernel @flyc.kernel
  - inline Triton softmax -> softmax_kernel @flyc.kernel

FlyDSL primitives:
  - rocdl.exp2 for softmax
  - rocdl.rcp for reciprocal (1/x)
  - Vector ops for fused element-wise
  - buffer_ops for global memory access
"""

import torch
import math


def rmsnorm_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    RMSNorm: output = input * weight / sqrt(mean(input^2) + epsilon)

    FlyDSL kernel design:
    - Block reduction for variance (mean of squares)
    - rocdl.rcp + sqrt for rsqrt
    - Fused multiply with weight
    - Can fuse with add (residual connection) for fused_add_rms_norm
    """
    variance = input.float().pow(2).mean(-1, keepdim=True)
    input_norm = input * torch.rsqrt(variance + epsilon)
    return (input_norm * weight).to(input.dtype)


def silu_and_mul_reference(
    input: torch.Tensor,
) -> torch.Tensor:
    """
    SiLU×Mul (GeGLU activation): output = silu(input[:, :H]) * input[:, H:]

    FlyDSL kernel design:
    - Split input at hidden_dim midpoint
    - silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    - Use rocdl.exp2 for exp: exp(x) = exp2(x * LOG2E)
    - Fuse multiply in single kernel
    """
    d = input.shape[-1] // 2
    return torch.nn.functional.silu(input[..., :d]) * input[..., d:]


def rope_reference(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    theta: float = 10000.0,
    partial_rotary_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary Position Embedding (RoPE).

    Gemma4 has DUAL RoPE:
    - Sliding window layers: theta=10000, partial_rotary_factor=1.0
    - Full attention layers: theta=1000000, partial_rotary_factor=0.25

    FlyDSL kernel design:
    - Compute frequency bands: freq_i = 1 / (theta ^ (2i/d))
    - Apply rotation: [cos(pos*freq), -sin(pos*freq); sin(pos*freq), cos(pos*freq)]
    - Partial rotary: only rotate first (d * partial_rotary_factor) dimensions
    - Can fuse with QK-norm: fused_qk_norm_rope
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=positions.device).float() / rotary_dim))

    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    def apply_rope(x, cos, sin, rotary_dim):
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]
        x_rot_out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return torch.cat([x_rot_out, x_pass], dim=-1)

    query_out = apply_rope(query, cos, sin, rotary_dim)
    key_out = apply_rope(key, cos, sin, rotary_dim)
    return query_out, key_out


def softmax_reference(
    input: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Softmax: output = exp(input - max) / sum(exp(input - max))

    FlyDSL kernel design:
    - Block-level max reduction
    - rocdl.exp2 for exp: exp(x) = exp2(x * LOG2E)
    - Block-level sum reduction
    - rocdl.rcp for reciprocal
    - Fused in single kernel pass (or 2-pass for numerical stability)
    """
    return torch.softmax(input.float(), dim=dim).to(input.dtype)
