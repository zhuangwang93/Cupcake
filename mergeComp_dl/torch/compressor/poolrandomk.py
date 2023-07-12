import torch
from math import ceil
from mergeComp_dl.torch import Compressor


class PoolRandomKCompressor(Compressor):
    def __init__(self, compress_ratio):
        super().__init__()
        self.name = "PoolDGC"
        self.quantization = False
        self.sparsification = True
        self.compress_ratio = compress_ratio


    def _sparsify_randomk(self, tensor, numel=0, k=0):
        if numel == 0:
            numel = tensor.numel()
        if k == 0:
            k = ceil(numel * self.compress_ratio)
        indices = torch.randint(0, numel, (k,), device=tensor.device, dtype=torch.int64)
        values = tensor[indices]
        return values, indices.type(torch.int32)


    def _sparsify(self, tensor):
        numel = tensor.numel()
        num_selects = int(numel * self.compress_ratio)
        indices = torch.randint(0, numel, (num_selects,), device=tensor.device, dtype=torch.int64)
        values = tensor[indices]
        return values, indices.type(torch.int32)


    def compress(self, tensor, name):
        numel = tensor.numel()
        values, indices = self._sparsify(tensor)
        tensor_compressed = values, indices
        ctx = (name, numel)

        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx):
        name, numel = ctx
        values, indices = tensor_compressed
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)
        return tensor_decompressed
