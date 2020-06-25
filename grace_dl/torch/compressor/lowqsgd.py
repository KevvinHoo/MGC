import torch

from grace_dl.torch import Compressor


class LowQSGDCompressor(Compressor):

    def __init__(self):
        super().__init__()
        # self.quantum_num = quantum_num

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        upperbound = torch.max(tensor.abs()) #Apply the MAX VALUE in the abs of the tensor instead of the norm of tensor
        upperbound = upperbound.flatten()

        abs_gradient = tensor.abs()

        level_float = 127 / upperbound * abs_gradient

        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8)
        tensor_compressed = tensor_compressed, upperbound

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / 127 * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
