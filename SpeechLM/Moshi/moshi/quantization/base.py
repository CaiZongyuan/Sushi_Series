# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Base class for all quantizers.
"""

from dataclasses import dataclass, field
import typing as tp

import torch
from torch import nn


# 存储量化操作的结果
@dataclass
class QuantizedResult:
    x: torch.Tensor
    # 量化码，通常是整数张量，表示码本中的索引
    codes: torch.Tensor
    # 使用的带宽 (kb/s)，每个批次项一个值
    bandwidth: torch.Tensor
    # 可选的惩罚项，用于损失函数
    penalty: tp.Optional[torch.Tensor] = None
    # 用于记录指标的字典
    metrics: dict = field(default_factory=dict)


class BaseQuantizer(nn.Module):
    """Base class for quantizers."""

    def __init__(self):
        super().__init__()
        # 用于控制是否对码本应用指数移动平均 (EMA)
        self._ema_frozen = False

    # 执行量化操作。它接收输入张量 x 和帧率 frame_rate，并返回一个 QuantizedResult 对象
    def forward(self, x: torch.Tensor, frame_rate: int) -> QuantizedResult:
        """
        Given input tensor x, returns first the quantized (or approximately quantized)
        representation along with quantized codes, bandwidth, and any penalty term for the loss.
        Finally, this returns a dict of metrics to update logging etc.
        Frame rate must be passed so that the bandwidth is properly computed.
        """
        raise NotImplementedError()

    # 将输入张量 x 编码为量化码
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth."""
        raise NotImplementedError()

    # 将量化码 codes 解码为量化表示
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        raise NotImplementedError()

    # 返回每个码本的大小（基数）
    @property
    def cardinality(self) -> int:
        """Cardinality of each codebook."""
        raise NotImplementedError()

    # 返回码本的总数
    @property
    def total_codebooks(self) -> int:
        """Total number of codebooks."""
        raise NotImplementedError()

    # 返回激活码本的数量
    @property
    def num_codebooks(self) -> int:
        """Number of active codebooks."""
        raise NotImplementedError()

    # 返回语义量化器
    @property
    def semantic_quantizer(self) -> "BaseQuantizer":
        """This returns the quantizer that models the first level of the hierarchy (typically semantic).

        In this case, it's the quantizer itself.
        """
        return self

    # 返回声学量化器
    @property
    def acoustic_quantizer(self) -> "BaseQuantizer":
        """This returns the quantizer that models the higher levels of the hierarchy (typically acoustic).

        In this case, it's the quantizer itself.
        """
        return self

    # 设置激活码本的数量
    def set_num_codebooks(self, n: int) -> None:
        """Set the number of active codebooks."""
        raise NotImplementedError()

    # 返回 _ema_frozen 的值，指示是否冻结 EMA
    @property
    def ema_frozen(self) -> bool:
        """Whether to apply ema to the codebooks."""
        return self._ema_frozen

    # 设置 _ema_frozen 的值
    def ema_frozen_(self, ema_frozen: bool) -> None:
        """Set whether ema should be applied to the codebooks."""
        self._ema_frozen = ema_frozen


class DummyQuantizer(BaseQuantizer):
    """Fake quantizer that actually does not perform any quantization."""

    def __init__(
        self,
        dimension: int,
        input_dimension: tp.Optional[int] = None,
        output_dimension: tp.Optional[int] = None,
    ):
        super().__init__()
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.input_proj: torch.nn.Module
        self.output_proj: torch.nn.Module
        if self.input_dimension == self.dimension:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False
            )
        if self.input_dimension == self.dimension:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False
            )

    def forward(self, x: torch.Tensor, frame_rate: int):
        q = x.unsqueeze(1)
        x = self.output_proj(self.input_proj(x))
        return QuantizedResult(
            x, q, torch.tensor(q.numel() * 32 * frame_rate / 1000 / len(x)).to(x)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        """
        x = self.input_proj(x)
        return x.unsqueeze(1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        """
        y = codes.squeeze(1)
        return self.output_proj(y)

    @property
    def total_codebooks(self):
        """Total number of codebooks."""
        return 1

    @property
    def num_codebooks(self):
        """Total number of codebooks."""
        return self.total_codebooks

    def set_num_codebooks(self, n: int):
        """Set the number of active codebooks."""
        raise AttributeError(
            "Cannot override the number of codebooks for the dummy quantizer"
        )

    @property
    def cardinality(self) -> int:
        """Cardinality of each codebook."""
        return 1
