import torch
import torch.distributed as dist
import horovod.torch as hvd


class DDPBackend():
    def __init__(self):
        self.world_size = hvd.size()
        self.rank = hvd.rank()
        self.local_size = hvd.local_size()
        self.local_rank = hvd.local_rank()
        if self.world_size == 1:
            print('WARNING: world size is 1')
        self.set_groups()

    @staticmethod
    def init_dist(backend='nccl', init='tcp://127.0.0.1:4000'):
        if hvd.size() >= 1:
            dist.init_process_group(backend=backend, init_method=init, world_size=hvd.size(), rank=hvd.rank())

    
    def set_groups(self):
        assert(self.world_size % self.local_size == 0)
        self.worker_id = self.rank // self.local_size
        self.worker_num = self.world_size // self.local_size

        # the communication group for GPUs with the same ID in different machines
        self.inter_comm_ranks_groups = []
        for i in range(self.local_size):
            ranks_group = [rank for rank in range(i, self.world_size, self.local_size)]
            assert(len(ranks_group) == self.worker_num)
            self.inter_comm_ranks_groups.append(ranks_group)
        self.inter_comm_groups = [dist.new_group(group) for group in self.inter_comm_ranks_groups]

        # the communication group for GPUs in the same machine
        self.intra_comm_ranks_groups = []
        for i in range(self.worker_num):
            base_rank = i * self.local_size
            local_ranks_group = [rank+base_rank for rank in range(0, self.local_size)]
            self.intra_comm_ranks_groups.append(local_ranks_group)
        self.intra_comm_groups = [dist.new_group(group) for group in self.intra_comm_ranks_groups]


    def get_worker_num(self):
        return self.worker_num


    def get_worker_id(self):
        return self.worker_id


    def get_inter_comm_group(self):
        return self.inter_comm_groups[self.local_rank], self.worker_num


    def get_intra_comm_group(self):
        return self.intra_comm_groups[self.worker_id], self.local_size


    def get_inter_ranks_group(self):
        return self.inter_comm_ranks_groups[self.local_rank]


    def get_intra_ranks_group(self):
        return self.intra_comm_ranks_groups[self.worker_id]


    def get_comm_group(self, intra):
        if intra:
            return self.get_intra_comm_group()
        else:
            return self.get_inter_comm_group()


    def get_ranks_group(self, intra):
        if intra:
            return self.get_intra_ranks_group()
        else:
            return self.get_inter_ranks_group()

    # in-place operation. Allreduce among all GPUs
    def allreduce(self, tensor, async_op=True):
        if self.world_size == 1:
            return tensor
            
        if async_op:
            handle = dist.all_reduce(tensor, async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.all_reduce(tensor)
        return tensor / self.world_size


    def reduce_scatter(self, tensor, async_op=True, intra=True): 
        comm_group, group_size = self.get_comm_group(intra)
        if group_size == 1:
            return tensor

        assert(tensor.numel() % group_size == 0)
        tensors = list(torch.chunk(tensor, group_size))
        ret = torch.empty_like(tensors[self.local_rank])
        if async_op:
            handle = dist.reduce_scatter(ret, tensors, group=comm_group, async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.reduce_scatter(ret, tensors, group=comm_group)
        return ret


    # when tensors on all GPUs have the same size
    def allgather(self, tensor, async_op=True, intra=False):
        comm_group, group_size = self.get_comm_group(intra)
        if group_size == 1:
            # print("Warning: the communication group size is 1", flush=True)
            return tensor

        ret = [torch.empty_like(tensor) for _ in range(group_size)]
        if async_op:
            handle = dist.all_gather(ret, tensor, group=comm_group, async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.all_gather(ret, tensor, group=comm_group)
        return ret
    

    def allgather_global(self, tensor, async_op=True):
        if self.world_size == 1:
            # print("Warning: the communication group size is 1", flush=True)
            return tensor
        
        ret = [torch.empty_like(tensor) for _ in range(self.world_size)]
        if async_op:
            handle = dist.all_gather(ret, tensor, async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.all_gather(ret, tensor)
        return ret
    

    def allgatherv(self, tensor, intra=False): 
        # tensor_sizes is the set of tensor sizes from all GPUs. The size can be different
        comm_group, group_size = self.get_comm_group(intra)
        comm_ranks_group = self.get_ranks_group(intra)
        if group_size == 1:
            # print("Warning: the communication group size is 1", flush=True)
            return tensor

        numel = torch.tensor(tensor.numel(), dtype=torch.int32, device=tensor.device)
        tensor_sizes = self.allgather(numel, intra=intra)
        ret = torch.empty(sum(tensor_sizes), dtype=tensor.dtype, device=tensor.device)
        ret = list(torch.split(ret, tensor_sizes))

        req = []
        idx = 0
        for rank in comm_ranks_group:
            if rank != self.rank:
                req.append(dist.isend(tensor=tensor, group=comm_group, dst=rank))
                req.append(dist.irecv(ret[idx], group=comm_group, src=rank))
            else:
                ret[idx] = tensor
            idx += 1
        
        for r in req:
            r.wait()
        return ret


    def alltoallv(self, tensors, splits, intra=False):
        # tensor_sizes is the set of sizes for each partition from each GPU 
        comm_group, group_size = self.get_comm_group(intra)
        comm_ranks_group = self.get_ranks_group(intra)
        if group_size == 1:
            # print("Warning: the communication group size is 1", flush=True)
            return tensors

        assert(len(tensors) == group_size)
        assert(len(tensors) == len(splits))

        tensors_splits = self.allgather(splits, intra=intra)
        size_idx = self.local_rank if intra else self.worker_id
        tensor_sizes = [s[size_idx] for s in tensors_splits]
        
        ret = torch.empty(sum(tensor_sizes), dtype=tensors[0].dtype, device=tensors[0].device)
        ret = list(torch.split(ret, tensor_sizes))

        req = []
        idx = 0
        for rank in comm_ranks_group:
            if rank != self.rank:
                req.append(dist.isend(tensor=tensors[idx], group=comm_group, dst=rank))
                req.append(dist.irecv(ret[idx], group=comm_group, src=rank))
            else:
                ret[idx] = tensors[idx]
            idx += 1
        
        for r in req:
            r.wait()
        return ret


    # only support intra-gather
    def gather(self, tensor, local_root=0, intra=True):
        if not intra:
            print("WARNING: only support intra-gather")
            return tensor

        comm_group, group_size = self.get_comm_group(intra)
        ranks_group = self.get_ranks_group(intra)
        if group_size == 1:
            return tensor

        req = []
        root = self.worker_id * self.local_size + local_root
        assert(root in ranks_group)
        if local_root == self.local_rank:
            ret = [torch.empty_like(tensor) for _ in range(group_size)]
            idx = 0
            for rank in ranks_group:
                if rank != self.rank:
                    req.append(dist.irecv(ret[idx], group=comm_group, src=rank))
                else:
                    ret[idx] = tensor
                idx += 1
        else:
            root = self.worker_id * self.local_size + local_root
            req.append(dist.isend(tensor=tensor, group=comm_group, dst=root))
            ret = [tensor]
        
        for r in req:
            r.wait()
        return ret


    # only support intra-reduce
    def reduce(self, tensor, local_root=0, async_op=True, intra=True):
        if not intra:
            print("WARNING: only support intra-reduce")
            return tensor

        comm_group, group_size = self.get_comm_group(intra)
        ranks_group = self.get_ranks_group(intra)
        if group_size == 1:
            return tensor

        root = self.worker_id * self.local_size + local_root
        assert(root in ranks_group)
        if async_op:
            handle = dist.reduce(tensor, dst=root, group=comm_group, async_op=async_op)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.reduce(tensor, dst=root, group=comm_group)
        return tensor


    # only support intra-broadcast
    def broadcast(self, tensor, local_root=0, async_op=True, intra=True):
        if not intra:
            print("WARNING: only support intra-broadcast")
            return tensor

        comm_group, group_size = self.get_comm_group(intra)
        ranks_group = self.get_ranks_group(intra)
        if group_size == 1:
            return tensor

        # if src is not in the ranks group, it may be the local rank id
        root = self.worker_id * self.local_size + local_root
        assert(root in ranks_group)
        if async_op:
            handle = dist.broadcast(tensor, src=root, group=comm_group, async_op=async_op)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.broadcast(tensor, src=root, group=comm_group)
        return tensor


    def gatherv(self, tensor, dst, intra=True):
        comm_group, group_size = self.get_comm_group(intra)
        ranks_group = self.get_ranks_group(intra)
        if group_size == 1:
            return tensor

        numel = torch.tensor(tensor.numel(), dtype=torch.int32, device=tensor.device)
        tensor_sizes = self.allgather(numel, intra=intra)

        req = []
        if dst == self.rank:
            ret = torch.empty(sum(tensor_sizes), dtype=tensor.dtype, device=tensor.device)
            ret = list(torch.split(ret, tensor_sizes))
            idx = 0
            for rank in ranks_group:
                if rank != self.rank:
                    req.append(dist.irecv(ret[idx], group=comm_group, src=rank))
                else:
                    ret[idx] = tensor
                idx += 1
        else:
            req.append(dist.isend(tensor=tensor, group=comm_group, dst=dst))
            ret = tensor
        
        for r in req:
            r.wait()
        return ret