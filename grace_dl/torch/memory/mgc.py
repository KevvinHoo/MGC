import torch
from grace_dl.torch import Memory


class MgcMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor = self.beta * self.residuals[name] + self.gamma * tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        # just store residuals in SPARSIFICATION STEP to further speed up the algorithm

        numel, shape, upperbound = ctx
        _, indices = tensor_compressed

        value = tensor.flatten()[indices]

        tensor_decompressed = torch.zeros(numel, dtype=value.dtype, layout=value.layout, device=value.device)
        tensor_decompressed.scatter_(0, indices, value)

        self.residuals[name] = tensor - tensor_decompressed.view(shape)
