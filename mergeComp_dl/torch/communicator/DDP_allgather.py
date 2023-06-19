import torch
from mergeComp_dl.torch import Communicator
from mergeComp_dl.torch.communicator.DDPbackend import DDPBackend
import time
import horovod.torch as hvd
import sys


class DDPAllgather(Communicator):
    def __init__(self, compressor, memory):
        super().__init__(compressor, memory)
        DDPBackend.init_dist()
        self.ddp = DDPBackend()
        self.local_rank = hvd.local_rank()
        self.local_size = hvd.local_size()
        self.world_size = hvd.size()
        self.worker_id = self.ddp.worker_id
        self.worker_num = self.ddp.worker_num
        self.comm_stream = torch.cuda.Stream()
        self.handles = {}
        self.shapes = {}
        self.name = "DDPAllgather"


    def is_topk_like(self):
        return self.compressor.sparsification


    def is_signsgd_like(self):
        return self.compressor.quantization


    def allgather_decompress(self, tensors_compressed, ctx, numel):
        assert(len(tensors_compressed) == 2)
        # we decompresss compressed tensors from the GPUs with the same local ID on different nodes
        # Because they are compressed from the same position range, we can sum the decompressed tensors
        if self.is_topk_like():
            return self.compressor.decompress(tensors_compressed, ctx)

        tensors, metadata = tensors_compressed
        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.worker_num), metadata.chunk(self.worker_num)

        tensors_decompressed = []
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))
        return sum(tensors_decompressed)


    def async_send(self, tensor, name):    
        self.shapes[name] = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        with torch.cuda.stream(self.comm_stream):
            if self.local_size > 1:
                # intra-node communication with reduce-scatter
                intra_tensor = self.ddp.reduce_scatter(tensor, intra=True)
            else:
                intra_tensor = tensor
            compensated_tensor = self.memory.compensate(intra_tensor, name)
            tensor_compressed, ctx = self.compressor.compress(compensated_tensor, name)
            self.memory.update(tensor, name, self.compressor, tensor_compressed, ctx)
            
            # inter-node communication with allgather
            assert(len(tensor_compressed) == 2)
            tensor, meta = tensor_compressed
            inter_tensors_compressed = [self.ddp.allgather(tensor, intra=False), self.ddp.allgather(meta, intra=False)]

            tensors, metadata = inter_tensors_compressed
            if isinstance(tensors, list):
                tensors = torch.stack(tensors).reshape(-1)
                metadata = torch.stack(metadata).reshape(-1)

            inter_tensors_compressed = [tensors, metadata]
            self.handles[name] = inter_tensors_compressed, ctx, numel
            return [-1], (name,)


    def decompress_tensor(self, name):
        with torch.cuda.stream(self.comm_stream):
            tensors_compressed, ctx, numel = self.handles[name]
            if ctx is None:
                return tensors_compressed
            return self.allgather_decompress(tensors_compressed, ctx, numel)


    def wait_receive(self, handle, ctx):
        name = ctx[0]
        with torch.cuda.stream(self.comm_stream):
            tensors_compressed, ctx, numel = self.handles[name]
            if ctx is None:
                decompressed_tensor = tensors_compressed
            else:
                decompressed_tensor = self.allgather_decompress(tensors_compressed, ctx, numel)

            if self.local_size > 1:
                tensor = self.ddp.allgather(decompressed_tensor, intra=True)
            else:
                tensor = decompressed_tensor
            tensor = tensor.reshape(self.shapes[name])
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        return tensor