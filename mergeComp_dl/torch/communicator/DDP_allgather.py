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
        self.flat_comm = (self.local_size == 1) or (self.worker_num == 1)
        self.comm_stream = torch.cuda.Stream()
        self.handles = {}
        self.shapes = {}
        self.name = "DDPAllgather"
        self.is_signsgd_like = self.compressor.quantization
        self.is_topk_like = not self.compressor.quantization


    def allgather_decompress(self, tensors_compressed, ctx):
        assert(len(tensors_compressed) == 2)
        # we decompresss compressed tensors from the GPUs with the same local ID on different nodes
        # Because they are compressed from the same position range, we can sum the decompressed tensors
        if self.is_topk_like:
            tensors, metadata = tensors_compressed
            if isinstance(tensors, list):
                tensors, metadata = torch.stack(tensors).reshape(-1), torch.stack(metadata).reshape(-1)
            return self.compressor.decompress((tensors, metadata), ctx)

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

        with torch.cuda.stream(self.comm_stream):
            if self.flat_comm:
                intra_tensor = tensor
            else:
                # intra-node communication with reduce-scatter
                intra_tensor = self.ddp.reduce_scatter(tensor, intra=True)

            compensated_tensor, meta = self.memory.pool_compensate(intra_tensor, name)
            if compensated_tensor is None:
                self.handles[name] = None
                return None, (name,)
            
            tensor_compressed, ctx = self.compressor.compress(compensated_tensor, name)
            self.memory.pool_update(compensated_tensor, name, self.compressor, tensor_compressed, ctx)
            
            # inter-node communication with allgather
            assert(len(tensor_compressed) == 2)
            tensor, meta = tensor_compressed
            if self.flat_comm:
                inter_tensors_compressed = [self.ddp.allgather_global(tensor), self.ddp.allgather_global(meta)]
            else:
                inter_tensors_compressed = [self.ddp.allgather(tensor, intra=False), self.ddp.allgather(meta, intra=False)]

            self.handles[name] = inter_tensors_compressed, ctx
            return [-1], (name,)


    def wait_receive(self, handle, ctx):
        name = ctx[0]
        if self.handles[name] is None:
            return None
        
        with torch.cuda.stream(self.comm_stream):
            tensors_compressed, ctx = self.handles[name]
            if ctx is None:
                decompressed_tensor = tensors_compressed
            else:
                decompressed_tensor = self.allgather_decompress(tensors_compressed, ctx)

            if self.flat_comm:
                tensor = decompressed_tensor
            else:
                tensor = self.ddp.allgather(decompressed_tensor, intra=True)
                tensor = torch.stack(tensor).reshape(-1)
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        return tensor