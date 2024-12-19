# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import typing as tp

import torch

from .base import BaseQuantizer, QuantizedResult
from .core_vq import ResidualVectorQuantization


class ResidualVectorQuantizer(BaseQuantizer):
    """残差向量量化器。

    参数:
        dimension (int): 代码本的维度。
        input_dimension (None 或 int): 输入的维度，默认为 `dimension`，如果未提供。
        output_dimension (None 或 int): 输出的维度，默认为 `dimension`，如果未提供。
        n_q (int): 使用的向量量化器数量。
        q_dropout (bool): 在训练时随机丢弃量化器。
        no_quantization_rate (float): 在训练时完全不应用量化的概率。RVQ 代码本仍将接收输入值以学习适当的代码本。
        bins (int): 代码本大小。
        decay (float): 代码本上指数移动平均的衰减率。
        threshold_usage_ratio (float): 定义了集群使用率低于该阈值时替换质心的标准。这表示为质心在均匀分布下使用率的一个分数，因此不依赖于批次大小等。
        replaced_usage_ratio (float): 替换质心时，使用此初始质心使用率，以避免质心被过快替换。
        codebook_offset (int): 用于代码本索引的偏移量。在使用多个量化器（如在 `SplitResidualVectorQuantizer` 中）时很有用。
        force_projection (bool): 是否强制进行输入和输出投影，即使维度是固定的。
        generator_seed (int 或 None): 用于初始化不量化时使用的随机数生成器的种子。
    """

    def __init__(
        self,
        dimension: int = 128,
        input_dimension: tp.Optional[int] = None,
        output_dimension: tp.Optional[int] = None,
        n_q: int = 8,
        q_dropout: bool = False,
        no_quantization_rate: float = 0.0,
        # 码本大小
        bins: int = 1024,
        # EMA 衰减率
        decay: float = 0.99,
        #  centroid 使用率阈值
        threshold_usage_ratio: float = 0.1,
        # 替换 centroid 时的初始使用率
        replaced_usage_ratio: float = 1.0,
        # 码本索引偏移量
        codebook_offset: int = 0,
        # 是否强制使用投影层
        force_projection: bool = False,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        self.decay = decay
        self.rng_dropout = random.Random(1234)
        self.input_proj: torch.nn.Module
        self.output_proj: torch.nn.Module
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False
            )
        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False
            )
        # 初始化 self.vq, ResidualVectorQuantization模块执行核心的量化操作
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            threshold_usage_ratio=threshold_usage_ratio,
            replaced_usage_ratio=replaced_usage_ratio,
            codebook_offset=codebook_offset,
        )

    def forward(self, x: torch.Tensor, frame_rate: int):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T] with `C` number of channels.
            frame_rate (int): frame rate of the input (e.g `T = frame_rate * duration`), used to compute
                the bandwidth.

        Returns:
            QuantizedResult: Quantized result with the following attributes:
                - `x` (torch.Tensor): Quantized tensor of shape [B, C, T].
                - `codes` (torch.Tensor): Quantized codes of shape [B, K, T] with `K` number of codebooks.
                - `bw` (torch.Tensor): Bandwidth of the quantized tensor in kbits per second.
                - `penalty` (torch.Tensor): Commitment loss.
                - `metrics` (dict): RVQ metrics, in particular rate of dead code replacement, and entropy.
        """
        # 量化器数量
        n_q = self.n_q
        x = self.input_proj(x)
        # 如果模型处于训练模式且 q_dropout 为 True, 则随机丢弃一些量化器
        if self.training and self.q_dropout:
            # 据q_dropout随机选择使用的量化器数量
            n_q = self.rng_dropout.randint(1, self.n_q)
        # 计算每个量化器的带宽 (kbps)
        bw_per_q = math.log2(self.bins) * frame_rate / 1000
        # 这是核心量化步骤：执行量化操作
        quantized, codes, commit_loss, metrics = self.vq(x, n_q=n_q)
        B, _, _ = quantized.shape
        # 如果模型处于训练模式且 no_quantization_rate 大于 0，则以一定概率跳过量化步骤
        if self.training and self.no_quantization_rate > 0:
            mask = (
                torch.rand(B, 1, 1, device=x.device) <= self.no_quantization_rate
            ).float()
            quantized = x * mask + (1 - mask) * quantized
        # 将其维度转换回原始维度或指定的输出维度
        quantized = self.output_proj(quantized)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        # 计算总带宽
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResult(
            quantized, codes, bw, penalty=torch.mean(commit_loss), metrics=metrics
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        n_q = self.n_q
        if x.shape[-1] == 0:
            return torch.empty((x.shape[0], n_q, 0), device=x.device, dtype=torch.int64)

        x = self.input_proj(x)
        # 使用 self.vq 获取码本索引
        codes = self.vq.encode(x, n_q=n_q)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks, vq.decode expects [K, B, T].
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        quantized = self.output_proj(quantized)
        return quantized

    @property
    def total_codebooks(self):
        return self.max_n_q

    @property
    def num_codebooks(self):
        return self.n_q

    def set_num_codebooks(self, n: int):
        assert n >= 0 and n <= self.max_n_q
        self.n_q = n

    @property
    def cardinality(self) -> int:
        return self.bins


class SplitResidualVectorQuantizer(BaseQuantizer):
    """Residual Vector Quantizer with separate projections for the first quantizer and the rest.

    Args:
        n_q (int): Number of residual vector quantizers used.
        n_semantic_q (int): Number of residual vector quantizers used for the semantic quantizer.
        **kwargs: Arguments to the constructor of `ResidualVectorQuantizer` that are shared between both.
    """

    def __init__(
        self,
        *,
        n_q: int = 8,
        n_q_semantic: int = 1,
        **kwargs,
    ):
        super().__init__()
        assert n_q > n_q_semantic, (
            f"Number of quantizers {n_q} must be larger "
            f"than the number of semantic quantizers {n_q_semantic}."
        )
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        q_dropout = kwargs.pop("q_dropout", False)
        # 用于语义量化，默认只用一个量化器
        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )
        # 用于声学量化，默认使用用8-1=7个量化器
        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            codebook_offset=1,
            force_projection=True,
            q_dropout=q_dropout,
            **kwargs,
        )

    """这个方法用于重新规范化 rvq_first 和 rvq_rest 的输出值，并将它们相加。
    由于量化器丢弃 (q_dropout) 的存在，实际使用的量化器数量可能与预设值不同。
    这个方法根据实际使用的量化器数量重新计算权重，确保结果的正确性。"""

    def _renorm_and_add(
        self,
        first_val: torch.Tensor,
        rest_val: torch.Tensor,
        n_q_semantic: int,
        n_q_acoustic: int,
    ):
        """Renormalizes values from `rvq_first` and `rvq_rest` and adds them.

        This allows correcting statistics that are normalized by the number of quantizers. To renormalize, we use the
        number of quantizers that are actually used, e.g. taking into account quantizer dropout.
        """
        n_q = n_q_semantic + n_q_acoustic
        renorm_first_val = first_val * n_q_semantic / n_q
        renorm_rest_val = rest_val * n_q_acoustic / n_q
        return renorm_first_val + renorm_rest_val

    def forward(self, x: torch.Tensor, frame_rate: int):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T] with `C` number of channels.
            frame_rate (int): frame rate of the input (e.g `T = frame_rate * duration`), used to compute
                the bandwidth.

        Returns:
            QuantizedResult: Quantized result with the following attributes:
                - `x` (torch.Tensor): Quantized tensor of shape [B, C, T].
                - `codes` (torch.Tensor): Quantized codes of shape [B, K, T] with `K` number of codebooks.
                - `bw` (torch.Tensor): Bandwidth of the quantized tensor in kbits per second.
                - `penalty` (torch.Tensor): Commitment loss.
                - `metrics` (dict): RVQ metrics, in particular rate of dead code replacement, and entropy.
        """
        # 使用语义量化器
        semantic_result = self.rvq_first(x, frame_rate)
        if self.n_q == self.n_q_semantic:
            return semantic_result
        # 使用声学量化器
        acoustic_result = self.rvq_rest(x, frame_rate)
        # 将语义量化结果和声学量化结果相加
        full_quantized_emb = semantic_result.x + acoustic_result.x
        # 将语义量化码和声学量化码拼接起来
        full_quantized_codes = torch.cat(
            [semantic_result.codes, acoustic_result.codes], dim=1
        )
        # 获取实际使用的量化器数量。由于量化器丢弃的存在，这可能与self.n_q_semantic或者n_q_acoustic不同
        n_q_semantic = semantic_result.codes.shape[1]
        n_q_acoustic = acoustic_result.codes.shape[1]
        # 计算总带宽，将语义带宽和声学带宽相加
        full_quantized_bandwidth = semantic_result.bandwidth + acoustic_result.bandwidth
        # 重新规范化语义惩罚项和声学惩罚项，并将它们相加，得到最终的惩罚项
        full_quantized_penalty = self._renorm_and_add(
            semantic_result.penalty, acoustic_result.penalty, n_q_semantic, n_q_acoustic
        )
        # 初始化最终的指标字典 full_quantized_metrics 为语义指标
        full_quantized_metrics = semantic_result.metrics
        """将声学指标添加到最终的指标字典中。 
        如果某个指标在语义指标中已经存在，
        则使用 _renorm_and_add 方法重新规范化并相加；
        否则，直接添加到字典中"""
        for key, value in acoustic_result.metrics.items():
            if key in full_quantized_metrics:
                full_quantized_metrics[key] = self._renorm_and_add(
                    full_quantized_metrics[key], value, n_q_semantic, n_q_acoustic
                )
            else:
                full_quantized_metrics[key] = value
        return QuantizedResult(
            full_quantized_emb,
            full_quantized_codes,
            full_quantized_bandwidth,
            penalty=full_quantized_penalty,
            metrics=full_quantized_metrics,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        codes = self.rvq_first.encode(x)
        if self.n_q > self.n_q_semantic:
            acoustic_codes = self.rvq_rest.encode(x)
            codes = torch.cat([codes, acoustic_codes], dim=1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks.
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized

    @property
    def total_codebooks(self):
        return self.rvq_first.max_n_q + self.rvq_rest.max_n_q

    @property
    def num_codebooks(self):
        return self.rvq_first.num_codebooks + self.rvq_rest.num_codebooks

    @property
    def n_q(self):
        return self.rvq_first.n_q + self.rvq_rest.n_q

    @property
    def dimension(self):
        return self.rvq_first.dimension

    @property
    def semantic_quantizer(self) -> ResidualVectorQuantizer:
        """This returns the quantizer that models the first level of the hierarchy (typically semantic)."""
        return self.rvq_first

    @property
    def acoustic_quantizer(self) -> ResidualVectorQuantizer:
        """This returns the quantizer that models the higher levels of the hierarchy (typically acoustic)."""
        return self.rvq_rest

    def set_num_codebooks(self, n: int):
        assert n >= self.n_q_semantic and n <= self.total_codebooks
        self.rvq_rest.set_num_codebooks(n - self.n_q_semantic)

    @property
    def cardinality(self) -> int:
        assert self.rvq_rest.cardinality == self.rvq_first.cardinality
        return self.rvq_first.cardinality
