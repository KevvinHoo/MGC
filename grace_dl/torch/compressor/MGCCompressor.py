import torch

from grace_dl.torch import Compressor

class MGCCompressor(Compressor):

    def __init__(self, compress_ratio):
        super().__init__(tensors_size_are_same=True)
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):

        shape = tensor.size()
        numel = tensor.numel()
        tensor = tensor.flatten()


        sample_shape = [max(1, int(numel * 0.01))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
        sample_tensor = tensor[sample_index]

        thr = torch.mean(sample_tensor.abs())
        mask = tensor.abs() > thr
        selected = mask.sum()
        # k = max(2, int(numel * self.compress_ratio * 0.01))
        # _, indices = torch.topk(tensor.abs(), k)

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
        upperbound = torch.max(values.abs()).flatten()
        lowwerbound = torch.min(values.abs()).flatten()

        abs_gradient = values.abs()

        level_float = 127 / (upperbound - lowwerbound) * (abs_gradient - lowwerbound)

        previous_level = level_float.floor()
        prob = torch.empty_like(values).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = previous_level + is_next_level

        tensor_compressed = (new_level * values.sign())

        tensor_comp = torch.zeros(numel, dtype=tensor_compressed.dtype, layout=tensor_compressed.layout, device=tensor_compressed.device)
        tensor_comp.scatter_(0, indices, tensor_compressed)

        tensor_comp = tensor_comp.type(torch.int8)

        tensor_comp = tensor_comp, upperbound, lowwerbound

        return tensor_comp, shape

    def decompress(self, tensor_compressed, ctx):

        shape = ctx
        values, upperbound ,lowwerbound = tensor_compressed

        decode_output = values.type(torch.float32).abs()

        value = (upperbound - lowwerbound) / 127 * decode_output + lowwerbound

        value *= values.sign()

        # tensor_decompressed = torch.zeros(numel, dtype=decode_output.dtype, layout=decode_output.layout, device=decode_output.device)
        # tensor_decompressed.scatter_(0, indices, decode_output)

        return value.view(shape)