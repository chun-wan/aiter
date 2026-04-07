"""
AITER AllReduce Integration for Gemma4-31B TP Communication

Provides three allreduce strategies:
  A. quick_all_reduce (INT4 compressed) - 4x bandwidth reduction
  B. custom_all_reduce ASM - lower latency than RCCL
  C. fused_allreduce_rmsnorm - merge allreduce + normalization

Integration with vLLM:
  Set VLLM_USE_AITER_ALLREDUCE=qr|asm|fused to select strategy.
  Default: standard RCCL allreduce.

Usage:
  from flydsl_attention.integration.aiter_allreduce import AiterAllReduceManager

  manager = AiterAllReduceManager(strategy='qr', rank=0, world_size=8)
  manager.initialize()

  # In forward pass:
  output = manager.all_reduce(hidden_states)

  # Fused variant:
  output = manager.all_reduce_rmsnorm(hidden_states, residual, weight, epsilon)
"""

import os
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class AiterAllReduceManager:
    """
    Manages AITER-based allreduce for tensor parallelism.

    Strategies:
    - 'qr':    quick_all_reduce with INT4 compression (4x bandwidth reduction)
    - 'asm':   custom_all_reduce with hand-written ASM kernels
    - 'fused': fused allreduce + rmsnorm (single kernel launch)
    - 'rccl':  default RCCL/NCCL (fallback)
    """

    def __init__(self, strategy: str = 'rccl', rank: int = 0, world_size: int = 1):
        self.strategy = strategy
        self.rank = rank
        self.world_size = world_size
        self._initialized = False

    def initialize(self):
        if self.world_size <= 1:
            logger.info("TP=1, no allreduce needed")
            self._initialized = True
            return

        if self.strategy == 'qr':
            self._init_quick_allreduce()
        elif self.strategy == 'asm':
            self._init_asm_allreduce()
        elif self.strategy == 'fused':
            self._init_fused_allreduce()
        else:
            logger.info("Using default RCCL allreduce")

        self._initialized = True

    def _init_quick_allreduce(self):
        """Initialize AITER quick allreduce with INT4 compression."""
        try:
            from aiter.ops.quick_all_reduce import init_custom_qr, qr_get_handle, qr_open_handles
            handle = qr_get_handle()
            all_handles = [None] * self.world_size
            dist.all_gather_object(all_handles, handle)
            qr_open_handles(all_handles)
            init_custom_qr(self.rank, self.world_size, all_handles)
            logger.info(f"AITER quick_all_reduce (INT4) initialized: rank={self.rank}/{self.world_size}")
        except Exception as e:
            logger.warning(f"Failed to init quick_all_reduce: {e}, falling back to RCCL")
            self.strategy = 'rccl'

    def _init_asm_allreduce(self):
        """Initialize AITER custom allreduce with ASM kernels."""
        try:
            from aiter.ops.custom_all_reduce import set_custom_all_reduce
            set_custom_all_reduce(True)
            logger.info(f"AITER custom_all_reduce (ASM) initialized: rank={self.rank}/{self.world_size}")
        except Exception as e:
            logger.warning(f"Failed to init ASM allreduce: {e}, falling back to RCCL")
            self.strategy = 'rccl'

    def _init_fused_allreduce(self):
        """Initialize AITER fused allreduce+rmsnorm."""
        try:
            from aiter.ops.custom_all_reduce import all_reduce_rmsnorm_
            self._fused_fn = all_reduce_rmsnorm_
            logger.info(f"AITER fused allreduce+rmsnorm initialized")
        except Exception as e:
            logger.warning(f"Failed to init fused allreduce: {e}, falling back to RCCL")
            self.strategy = 'rccl'
            self._fused_fn = None

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform allreduce using the configured strategy."""
        if self.world_size <= 1:
            return tensor

        if self.strategy == 'qr':
            from aiter.ops.quick_all_reduce import qr_all_reduce
            return qr_all_reduce(tensor)
        elif self.strategy == 'asm':
            from aiter.ops.custom_all_reduce import all_reduce_asm_
            return all_reduce_asm_(tensor)
        else:
            dist.all_reduce(tensor)
            return tensor

    def all_reduce_rmsnorm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """
        Fused allreduce + RMSNorm in a single kernel launch.

        Equivalent to:
          all_reduce(input)
          output = rmsnorm(input + residual, weight, epsilon)

        But done in one kernel, saving:
          - 1 kernel launch overhead
          - 1 global memory read/write pass
        """
        if self.world_size <= 1:
            variance = (input + residual).float().pow(2).mean(-1, keepdim=True)
            return ((input + residual) * torch.rsqrt(variance + epsilon) * weight).to(input.dtype)

        if self.strategy == 'fused' and self._fused_fn is not None:
            return self._fused_fn(input, residual, weight, epsilon)
        else:
            self.all_reduce(input)
            combined = input + residual
            variance = combined.float().pow(2).mean(-1, keepdim=True)
            return (combined * torch.rsqrt(variance + epsilon) * weight).to(input.dtype)

    def destroy(self):
        """Clean up allreduce resources."""
        if self.strategy == 'qr':
            try:
                from aiter.ops.quick_all_reduce import qr_destroy
                qr_destroy()
            except:
                pass


def get_allreduce_manager() -> AiterAllReduceManager:
    """
    Factory function that creates the allreduce manager based on environment.

    Set VLLM_USE_AITER_ALLREDUCE to one of:
      'qr'    - quick allreduce INT4
      'asm'   - custom ASM allreduce
      'fused' - fused allreduce+rmsnorm
      'rccl'  - default RCCL (or unset)
    """
    strategy = os.environ.get('VLLM_USE_AITER_ALLREDUCE', 'rccl').lower()
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    return AiterAllReduceManager(strategy=strategy, rank=rank, world_size=world_size)
