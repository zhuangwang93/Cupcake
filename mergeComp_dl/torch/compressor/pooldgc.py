import torch
import random
from math import ceil


from mergeComp_dl.torch import Compressor


class PoolDgcCompressor(Compressor):
    def __init__(self, compress_ratio, sample_ratio=0.001, strided_sample=True,
                 compress_upper_bound=1.3, compress_lower_bound=0.7,
                 max_adaptation_iters=4, warmup_epochs=4, warmup_coeff=0.25,
                 fp16_values=False, int32_indices=False):
        super().__init__(tensors_size_are_same=False)
        self.name = "PoolDGC"
        self.quantization = False
        self.compress_ratio = compress_ratio
        self.fp16_values = fp16_values
        self.int32_indices = int32_indices
        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.sample_ratio = min(max(sample_ratio, 0.001), 1.0)
        self.strided_sample = strided_sample
        self.compress_upper_bound = compress_upper_bound
        self.compress_lower_bound = compress_lower_bound
        self.max_adaptation_iters = max_adaptation_iters
        self.warmup_epochs = warmup_epochs
        if self.warmup_epochs > 0:
            if warmup_coeff is None:
                self.warmup_coeff = self.base_compress_ratio \
                                    ** (1. / (self.warmup_epochs + 1))
            else:
                if isinstance(warmup_coeff, (tuple, list)):
                    assert len(warmup_coeff) >= self.warmup_epochs
                    for wc in warmup_coeff:
                        assert 0 < wc <= 1
                else:
                    assert 0 < warmup_coeff <= 1
                self.warmup_coeff = warmup_coeff
        else:
            self.warmup_coeff = 1
        self.attributes = (self.sample_ratio, self.base_compress_ratio)
        self.num_selects = 4 * 1024 * 1024
        self.indices = torch.zeros(self.num_selects, dtype=torch.int32).cuda()
        self.zeros = {}
        self.masks = {}
        self.flag = False


    def warmup_compress_ratio(self, epoch):
        compress_ratio = self.base_compress_ratio
        if self.warmup_epochs > 0:
            if epoch < self.warmup_epochs:
                if isinstance(self.warmup_coeff, (tuple, list)):
                    compress_ratio = self.warmup_coeff[epoch]
                else:
                    compress_ratio = max(self.warmup_coeff ** (epoch + 1),
                                        self.base_compress_ratio)

        self.attributes = (self.sample_ratio, compress_ratio)
        print(epoch, self.attributes)


    def set_compress_ratio(self, ratio):
        sample_ratio, compress_ratio = self.attributes
        self.attributes = sample_ratio, ratio


    def _sparsify(self, tensor):
        sample_ratio, compress_ratio = self.attributes
        numel = tensor.numel()

        if numel < 1024*16:
            k = ceil(numel * compress_ratio)
            _, indices = torch.topk(tensor.abs(), k)
            values = tensor[indices]
            return values, indices, numel, None

        if numel <= 1024*1024:
            sample_ratio = 0.01

        num_samples = int(numel * sample_ratio)
        num_selects = int(numel * compress_ratio)

        if self.strided_sample:
            sample_stride = int(1 // sample_ratio)
            sample_start = random.randint(0, min(sample_stride, numel-1))
            samples = tensor[sample_start::sample_stride]
        else:
            samples = tensor[torch.randint(0, numel, (num_samples, ), device=tensor.device)]

        k = ceil(num_samples * compress_ratio * 1.1)
        thr = torch.min(torch.topk(samples.abs(), k, 0, largest=True, sorted=False)[0])

        mask = tensor.abs() >= thr
        selected = mask.sum()

        for _ in range(self.max_adaptation_iters):
            if selected > self.compress_upper_bound * num_selects:
                thr = self.compress_upper_bound * thr
            elif selected < self.compress_lower_bound * num_selects:
                thr = self.compress_lower_bound * thr
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        indices, = torch.where(mask)
        if indices.numel() >= num_selects:
            indices = indices[:num_selects]  
        else:
            indices = torch.randint(0, numel, (num_selects,), device=tensor.device, dtype=torch.int64)

        values = tensor[indices]
        return values, indices.type(torch.int32), numel, num_selects


    def compress(self, tensor, name):
        sample_ratio, compress_ratio = self.attributes
        # if fused tensors are not ready, NOT call communicate()
        if tensor is None:
            return None, name
        values, indices, numel, num_selects = self._sparsify(tensor)
        ctx = (name, numel, num_selects, values.dtype, indices.dtype)
        return (values, indices), ctx


    def decompress(self, tensor_compressed, ctx):
        sample_ratio, compress_ratio = self.attributes
        name, numel, num_selects, vdtype, idtype = ctx
        # decompress
        values, indices = tensor_compressed
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)

        return tensor_decompressed