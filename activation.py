"""Custom activation functions used by the NeRF backbone."""

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class _TruncatedExp(Function):
    """Exp activation whose backward gradient is clamped to avoid fp16 overflow."""

    @staticmethod
    @custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return grad_output * torch.exp(x.clamp(max=15))


trunc_exp = _TruncatedExp.apply


def biased_softplus(x, bias: float = 0.0):
    """Softplus shifted by ``bias`` to keep early-iter densities small."""
    return torch.nn.functional.softplus(x - bias)
