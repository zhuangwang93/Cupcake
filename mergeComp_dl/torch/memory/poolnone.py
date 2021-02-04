import torch
import horovod.torch as hvd

from .memory_pool import MemoryPool


class PoolNoneMemory(MemoryPool):
    def __init__(self, named_parameters, fusion_num=2):
        self.world_size = hvd.size()
        super().__init__(named_parameters, fusion_num)

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        grad = self.get_grad(name)
        residual = self.get_velocity(name)
        residual.copy_(grad)

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

    def reduce(self, ctx, name):
        reduction = self.get_reduction(name)
        reduction.zero_()
        # TODO:for compression algorithms with allreduce, the received results have been averaged already.
        # Probably there is no need to divide c with self.world_size.
        for c in ctx:
            reduction.add_(c/self.world_size)