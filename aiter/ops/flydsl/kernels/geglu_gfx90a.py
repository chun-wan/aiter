# SPDX-License-Identifier: MIT
# FlyDSL GeGLU Activation Kernel for gfx90a (MI250)
# gelu_tanh(gate) * up -- element-wise fused kernel
#
# This kernel fuses the split + gelu + multiply into a single GPU kernel
# to avoid extra memory round-trips.
#
# Layout: input [M, 2*N] -> output [M, N]
#   gate = input[:, :N]
#   up   = input[:, N:]
#   output = gelu_tanh(gate) * up

import functools
import math

import flydsl.compiler as flyc
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, gpu, vector, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # ~0.7978845608


@functools.lru_cache(maxsize=32)
def compile_geglu_kernel(
    M: int,
    N: int,
    BLOCK_SIZE: int = 256,
):
    """Compile a FlyDSL GeGLU kernel for gfx90a.

    gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    output = gelu_tanh(gate) * up

    where input[:, :N] = gate, input[:, N:] = up
    """
    arch = get_rocm_arch()

    @flyc.kernel
    def geglu_fwd(
        input_ptr,     # [M, 2*N] bf16
        output_ptr,    # [M, N] bf16
        stride_m: int,
        stride_n: int,
        M_param: int,
        N_param: int,
    ):
        # This is a placeholder structure for the FlyDSL kernel.
        # Full implementation requires the FlyDSL expr/arith API
        # to express the gelu_tanh computation in MLIR.
        #
        # The actual kernel body would use:
        # - buffer_load for coalesced BF16 reads
        # - arith.mulf, arith.addf for FP32 gelu computation
        # - tanh approximation via polynomial
        # - buffer_store for BF16 output write
        pass

    return geglu_fwd


def flydsl_geglu(input_tensor, output_tensor=None):
    """FlyDSL GeGLU for gfx90a.

    Fallback: uses PyTorch for now until FlyDSL kernel body is complete.
    """
    M, N2 = input_tensor.shape
    N = N2 // 2
    if output_tensor is None:
        output_tensor = input_tensor.new_empty(M, N)

    gate = input_tensor[:, :N]
    up = input_tensor[:, N:]

    import torch
    import torch.nn.functional as F
    output_tensor.copy_(F.gelu(gate, approximate="tanh") * up)
    return output_tensor
