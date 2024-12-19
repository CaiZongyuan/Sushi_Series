# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.compile import torch_compile_lazy


"""@torch_compile_lazy 装饰器：
这是一个 PyTorch 的特性(在较新版本中引入), 用于延迟编译函数。
它会在函数第一次被调用时将其编译成更优化的版本, 从而提高后续调用的性能"""


@torch_compile_lazy
def gating_forward_kernel(
    weight_in: torch.Tensor, weight_out: torch.Tensor, activation, x: torch.Tensor
):
    x = F.linear(x, weight_in)
    B, T, _ = x.shape
    """这是 GLU(Gated Linear Unit)机制的核心, 
    将输出分成两部分：一部分用于门控, 另一部分用于线性变换"""
    x = x.view(B, T, 2, -1)
    """对 reshape 后的张量的前半部分应用激活函数,然后与后半部分相乘。
    这就是门控操作, 前半部分控制后半部分的信息流动"""
    x = activation(x[..., 0, :]) * x[..., 1, :]
    x = F.linear(x, weight_out)
    return x


class ActivationGating(nn.Module):
    """
    实现了一个通用的门控 FFN 层，可以使用不同的激活函数
    Args:
        dim (int): dimension of the input and output of the transformer.
        activation (any callable Tensor to Tensor): activation function to use.
        **factory_kwargs: other kwargs passed to the linear layer, in particular device and dtype.
    """

    # 与PyTorch的Fully Sharded Data Parallel(FSDP)相关, 表示该模块是FSDP的最终模块
    _fsdp_final = True

    def __init__(self, dim: int, dim_feedforward: int, activation, **factory_kwargs):
        super().__init__()
        # We should have 8 d^2 param, instead we will have
        # 2 * h * d + h * d = 3 h * d = 8 d^2
        # so h = 8 d / 3 but following Hervé's advice we use 21 / 8 as an approx.
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)
        self.linear_out = nn.Linear(hidden, dim, bias=False, **factory_kwargs)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        return gating_forward_kernel(
            self.linear_in.weight, self.linear_out.weight, self.activation, x
        )


# 根据传入的名称字符串返回相应的激活函数
def _get_activation(name: str):
    if name in ["sigmoid", "tanh", "relu"]:
        return getattr(torch, name)
    elif name in ["leaky_relu", "elu", "gelu", "silu", "mish", "softsign"]:
        return getattr(torch.nn.functional, name)
    elif name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown activation {name}")


def _make_gating(
    name: str, dim: int, dim_feedforward: int, **factory_kwargs
) -> nn.Module:
    return ActivationGating(
        dim, dim_feedforward, _get_activation(name), **factory_kwargs
    )


def make_gating(
    name: str, dim: int, dim_feedforward: int, **factory_kwargs
) -> nn.Module:
    gating = _make_gating(name, dim, dim_feedforward, **factory_kwargs)
    max_params = 2 * dim * dim_feedforward
    params = sum(p.numel() for p in gating.parameters())
    assert (
        params <= max_params
    ), f"{name} gating has {params} params, max is {max_params}"
    return gating
