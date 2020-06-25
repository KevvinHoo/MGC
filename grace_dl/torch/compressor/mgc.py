import torch

from grace_dl.torch import Compressor


class MGC(Compressor):
    #

    def __init__(self, compress_ratio):
        super().__init__(tensors_size_are_same=False)
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):

        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        sample_shape = [max(1, int(numel * 0.01))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long) # Sample
        sample_tensor = tensor[sample_index]

        thr = torch.mean(sample_tensor.abs())

        mask = tensor.abs() >= thr
        selected = mask.sum()

        for _ in range(10):
            if selected > 1.3 * numel * self.compress_ratio:
                thr = 1.3 * thr
            elif selected < 0.7 * numel * self.compress_ratio:
                thr = 0.7 * thr
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        indices, = torch.where(mask)
        values = tensor[indices]

        ################################################################################################################

        # upperbound = torch.max(values.abs())  # Apply the MAX VALUE in the abs of the tensor instead of the norm of tensor
        # lowwerbound = torch.min(values.abs())

        upperbound = torch.max(values.abs())
        lowwerbound = torch.min(values.abs())

        upperbound = upperbound.flatten()
        lowwerbound = lowwerbound.flatten()

        abs_gradient = values.abs()

        level_float = 127 / (upperbound - lowwerbound) * (abs_gradient - lowwerbound)

        previous_level = level_float.floor()
        prob = torch.empty_like(values).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = values.sign()

        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8)

        ctx = shape, numel, upperbound, lowwerbound

        return [tensor_compressed, indices], ctx

    def decompress(self, tensor_compressed, ctx):
        shape, numel, upperbound, lowwerbound = ctx
        values, indices = tensor_compressed

        # tensor_compressed, upperbound, lowwerbound = values

        decode_output = values.type(torch.float32)
        value = (upperbound - lowwerbound) / 127 * decode_output + lowwerbound

        ################################################################################################################

        tensor_decompressed = torch.zeros(numel, dtype=value.dtype, layout=value.layout, device=value.device)
        tensor_decompressed.scatter_(0, indices, value)

        return tensor_decompressed.view(shape)
