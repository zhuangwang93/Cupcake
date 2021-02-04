from abc import ABC, abstractmethod
import time
import torch


class Memory(ABC):
    def initialize(self, named_parameters):
        pass

    def pool_compensate(self, tensor, name):
        pass

    def pool_update(self, tensor, name, compressor, tensor_compressed, ctx, advance_reduce=False):
        """Update the residuals."""
        pass

    def pool_reduce(self, compressed_tensor):
        pass


class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    @abstractmethod
    def compress(self, tensor, name, ctx=None, server=False):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx, server=False):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)

    def clean(self):
        pass


class Communicator(ABC):
    @abstractmethod
    def async_send(self, tensors, name):
        raise NotImplemented("async_send was not implemented.")


    @abstractmethod
    def wait_receive(self, handles, ctx):
        raise NotImplemented("wait_receive was not implemented.")

    def __init__(self, compressor, memory):
        self.compressor = compressor
        self.memory = memory


    def send_step(self, tensor, name):
        tensor, meta = self.memory.pool_compensate(tensor, name)
        if tensor is not None:
            tensor_compressed, ctx = self.compressor.compress(tensor, name, meta)
            self.memory.pool_update(tensor, name, self.compressor, tensor_compressed, ctx)
            handles = self.async_send(tensor_compressed, ctx)
            return handles, ctx

        return False, None


    def receive_step(self, handles, ctx):
        name = ctx[0]
        tensors = self.wait_receive(handles, ctx)
        if tensors is not None:
            self.memory.reduce(tensors, name)
        self.memory.pool_step(name)
