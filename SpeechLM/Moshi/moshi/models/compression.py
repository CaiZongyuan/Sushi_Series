# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of this file is adapted from encodec.py in https://github.com/facebookresearch/audiocraft
# released under the following license.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Compression models or wrapper around existing models. In particular, provides the implementation
for Mimi. Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import typing as tp

import torch
from torch import nn


from ..quantization import (
    QuantizedResult,
    BaseQuantizer,
    SplitResidualVectorQuantizer,
    ResidualVectorQuantizer,
)
from ..modules.resample import ConvDownsample1d, ConvTrUpsample1d
from ..modules.streaming import StreamingModule, State
from ..utils.compile import no_compile, CUDAGraphed


logger = logging.getLogger()


class CompressionModel(StreamingModule[State]):
    """
    CompressionModel (抽象基类) 定义了音频压缩模型的通用接口，任何音频压缩模型都应该实现这些方法
    """

    # 模型的前向传播方法，输入音频张量 x，输出量化结果 QuantizedResult
    @abstractmethod
    def forward(self, x: torch.Tensor) -> QuantizedResult: ...

    # 将音频张量 x 编码为离散的码本索引
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.encode`."""
        ...

    # 将码本索引解码为重构的音频张量
    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.decode`."""
        ...

    # 将码本索引解码为连续的潜在空间表示
    @abstractmethod
    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        ...

    # 一系列属性定义
    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def cardinality(self) -> int: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int: ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...


@dataclass
class _MimiState:
    graphed_tr_enc: CUDAGraphed | None
    graphed_tr_dec: CUDAGraphed | None

    def reset(self):
        pass


class MimiModel(CompressionModel[_MimiState]):
    """Mimi model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (float): Final frame rate of the quantized representatiopn.
        encoder_frame_rate (float): frame rate of the encoder model. Note that if `frame_rate != encopder_frame_rate`,
            the latent will be resampled linearly to match the desired `frame_rate` before and after quantization.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        encoder_transformer (nn.Module or None): optional transformer for the encoder.
        decoder_transformer (nn.Module or None): optional transformer for the decoder.
        resample_method (str): method to use for resampling the latent space before the quantizer.
        upsample_channel_wise_bug (bool): controls whether the upsampling is channel wise.
            Defaults to true to reproduce bug in original implementation.
        freeze_encoder: whether to freeze the encoder weights.
        freeze_quantizer: whether to freeze the quantizer weights.
        freeze_quantizer_level: If positive, freeze the quantizer up to this level.
        torch_compile_encoder_decoder (bool): if True, uses torch.compile on the encoder / decoder.
            Deactivated by default for training as this is incompatible at the moment with weight norm.
            See https://github.com/pytorch/pytorch/issues/121902
            Also this seems to work well with 2.2.0, but completely fail with 2.4.0.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: BaseQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        encoder_transformer: tp.Optional[nn.Module] = None,
        decoder_transformer: tp.Optional[nn.Module] = None,
        resample_method: str = "interpolate",
        upsample_channel_wise_bug: bool = True,
        freeze_encoder: bool = False,
        freeze_quantizer: bool = False,
        freeze_quantizer_level: int = -1,
        torch_compile_encoder_decoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.encoder_frame_rate = encoder_frame_rate
        self.torch_compile_encoder_decoder = torch_compile_encoder_decoder

        if freeze_encoder:
            """遍历编码器self.encoder的所有参数,
            并将requires_grad属性设置为False"""
            for p in self.encoder.parameters():
                p.requires_grad = False
            # 如果存在self.encoder_transformer，则同样冻结它的所有参数。
            if self.encoder_transformer is not None:
                for p in self.encoder_transformer.parameters():
                    p.requires_grad = False
            """冻结量化器self.quantizer中名为input_proj.weight的参数。
            input_proj 是将编码器的输出投影到量化器输入空间的层，
            冻结它的权重意味着固定这个投影关系"""
            for name, p in self.quantizer.named_parameters():
                if name.endswith("input_proj.weight"):
                    p.requires_grad = False
        if freeze_quantizer:
            """调用量化器的ema_frozen_方法, 并传入 True。
            这通常意味着冻结量化器中使用的指数移动平均 (EMA) 的更新。
            很多量化器使用 EMA 来跟踪一些统计量，例如聚类中心。
            冻结 EMA 意味着这些统计量在训练过程中不会改变"""
            self.quantizer.ema_frozen_(True)
        self.freeze_quantizer = freeze_quantizer
        # 设置冻结量化器的级别.大于 0,则使用传入的值,否则，使用量化器的码本数量
        self.freeze_quantizer_level = (
            freeze_quantizer_level
            if freeze_quantizer_level > 0
            else self.quantizer.num_codebooks
        )

        # 编码器通常是SeanetEncoder，并且SeanetEncoder会暴露一个名为dimension的属性
        dimension = encoder.dimension
        assert isinstance(
            dimension, int
        ), f"Dimension should be int, got {dimension} of type {type(dimension)}."
        self.dimension = dimension

        # 处理音频信号的重采样操作：
        assert resample_method in [
            "interpolate",
            "conv",
            "avg_pool",
        ], f"Invalid resample_method {resample_method}"
        self.resample_method = resample_method
        # 如果编码器的帧率和模型的帧率不相同，则需要进行重采样
        if encoder_frame_rate != frame_rate:
            """如果模型是因果模型，则不能使用插值方法进行重采样。
            因为插值需要用到未来的信息，这与因果模型的定义相矛盾"""
            assert not (
                causal and resample_method == "interpolate"
            ), "Cannot interpolate with causal model."
            """使用卷积或平均池化进行重采样时，编码器的帧率必须大于模型的帧率。
            因为卷积和平均池化在这里是用于下采样的。不能用它们进行上采样。"""
            if resample_method in ["conv", "avg_pool"]:
                assert (
                    self.encoder_frame_rate > self.frame_rate
                ), "Cannot upsample with conv."
                # 计算下采样的步幅 downsample_stride
                downsample_stride = self.encoder_frame_rate / self.frame_rate
                # 下采样步幅必须是整数
                assert downsample_stride == int(
                    downsample_stride
                ), f"Only integer strides are supported, got {downsample_stride}"
                """如果重采样方法是 "conv"，则 learnt 为 True, 表示使用可学习的卷积层进行重采样。
                否则, learnt 为 False, 表示使用固定的平均池化层。"""
                learnt = resample_method == "conv"
                # 创建一个 ConvDownsample1d 对象，用于下采样
                self.downsample = ConvDownsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                )
                if freeze_encoder:
                    for p in self.downsample.parameters():
                        p.requires_grad = False
                # 创建一个 ConvTrUpsample1d 对象，用于上采样（在解码阶段使用）
                self.upsample = ConvTrUpsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                    channel_wise=upsample_channel_wise_bug,
                )

    # 初始化模型的流式处理状态，特别是为了利用 CUDA Graph 进行优化
    def _init_streaming_state(self, batch_size: int) -> _MimiState:
        # 获取模型参数所在的设备。这通常用于确定模型是在 CPU 还是 GPU 上运行
        device = next(self.parameters()).device
        # 如果设备类型不是 "cuda"（即不是 GPU），则将 disable 设置为 True。这意味着 CUDA Graph 将被禁用
        disable = device.type != "cuda"
        graphed_tr_dec = None
        graphed_tr_enc = None
        # 创建一个 CUDAGraphed 对象，并将self.encoder_transformer包装在其中。disable 参数控制是否禁用 CUDA Graph
        if self.encoder_transformer is not None:
            graphed_tr_enc = CUDAGraphed(self.encoder_transformer, disable=disable)
        if self.decoder_transformer is not None:
            graphed_tr_dec = CUDAGraphed(self.decoder_transformer, disable=disable)
        # 返回一个 _MimiState 对象，其中包含编码器和解码器转换器的 CUDA Graph（如果已创建）
        return _MimiState(graphed_tr_enc, graphed_tr_dec)

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.cardinality

    # 帧率转换：将输入张量 x 的帧率从编码器帧率转换为模型的整体帧率
    def _to_framerate(self, x: torch.Tensor):
        # Convert from the encoder frame rate to the overall framerate.
        # 获取输入张量的时间维度长度
        _, _, length = x.shape
        # 获取编码器帧率
        frame_rate = self.encoder_frame_rate
        # 获取模型的整体帧率
        new_frame_rate = self.frame_rate
        # 如果两个帧率相同，则无需转换，直接返回输入
        if frame_rate == new_frame_rate:
            return x
        # 如果重采样方法是插值
        if self.resample_method == "interpolate":
            # 计算目标长度
            target_length = int(length * new_frame_rate / frame_rate)
            # 使用线性插值进行重采样
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            # 否则使用 self.downsample(x) 进行重采样，这通常是卷积或平均池化下采样
            return self.downsample(x)

    """反向帧率转换：将输入张量 x 的帧率从模型的整体帧率转换回编码器帧率。
    逻辑与 _to_framerate 类似，
    只是在 else 分支中使用 self.upsample(x) 进行重采样，这通常是转置卷积上采样。 
    这在解码阶段使用，将潜在表示的帧率转换回编码器帧率，以便解码器能够正确处理"""

    def _to_encoder_framerate(self, x: torch.Tensor):
        # Convert from overall framerate to the encoder frame rate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.upsample(x)

    """torch.compile 上下文管理:这是一个属性，返回一个上下文管理器，
    用于控制是否使用 torch.compile 编译编码器和解码器。"""

    @property
    def _context_for_encoder_decoder(self):
        # 使用 torch.compile 编译编码器和解码器以提高性能
        if self.torch_compile_encoder_decoder:
            # 此时返回一个空上下文管理器，相当于什么都不做
            return nullcontext()
        else:
            # 这是一个自定义的上下文管理器，用于禁用 torch.compile
            return no_compile()

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        # 输入检查和初始化
        assert x.dim() == 3
        length = x.shape[-1]
        # 初始化一个字典 extra_metrics 用于存储额外的指标
        extra_metrics: tp.Dict[str, torch.Tensor] = {}

        # 冻结量化器 (可选)
        if self.freeze_quantizer:
            if isinstance(self.quantizer, SplitResidualVectorQuantizer):
                # 将相应层的 eval() 方法设置为评估模式，停止梯度计算和参数更新
                self.quantizer.rvq_first.eval()
                for i in range(
                    self.freeze_quantizer_level - self.quantizer.n_q_semantic
                ):
                    self.quantizer.rvq_rest.vq.layers[i].eval()
            elif isinstance(self.quantizer, ResidualVectorQuantizer):
                for i in range(self.freeze_quantizer_level):
                    self.quantizer.vq.layers[i].eval()
            else:
                raise ValueError(f"Unsupported quantizer type {type(self.quantizer)}")

        # 编码
        # 使用上下文管理器控制是否使用 torch.compile 编译编码器和解码器
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)
        # 将 emb 的帧率转换为模型的整体帧率
        emb = self._to_framerate(emb)
        # 计算预期长度，并断言 emb 的时间维度长度与预期长度相差小于 1。 这是为了确保帧率转换的正确性。
        expected_length = self.frame_rate * length / self.sample_rate
        assert abs(emb.shape[-1] - expected_length) < 1, (
            emb.shape[-1],
            expected_length,
        )

        # 使用量化器 self.quantizer 对嵌入 emb 进行量化
        q_res = self.quantizer(emb, self.frame_rate)
        emb = q_res.x
        # 将emb的帧率转换回编码器的帧率
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb)

        with self._context_for_encoder_decoder:
            out = self.decoder(emb)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = out
        q_res.metrics.update(extra_metrics)
        return q_res

    # 将音频波形 x 编码到未量化的潜在空间
    def _encode_to_unquantized_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a batch of waveforms to unquantized latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Unquantized embeddings.
        """
        assert (
            #  (Batch, Channels, Time)
            x.dim()
            == 3
        ), f"CompressionModel._encode_to_unquantized_latent expects audio of shape [B, C, T] but got {x.shape}"
        # 获取流式处理状态，这与 CUDA Graph 优化有关
        state = self._streaming_state
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        if self.encoder_transformer is not None:
            if state is None:
                (emb,) = self.encoder_transformer(emb)
            else:
                assert state.graphed_tr_enc is not None
                (emb,) = state.graphed_tr_enc(emb)
        # 将编码后的嵌入 emb 的帧率转换为模型的整体帧率
        emb = self._to_framerate(emb)
        return emb

    # 将音频波形 x 编码到量化的表示
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the given input tensor to quantized representation.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes (torch.Tensor): an int tensor of shape [B, K, T]
                with K the number of codebooks used and T the timestep.
        """
        emb = self._encode_to_unquantized_latent(x)
        # 使用量化器对未量化的嵌入进行量化
        codes = self.quantizer.encode(emb)
        return codes

    # 将音频波形 x 编码到潜在空间，可以控制是否进行量化
    # 提供了一个更灵活的接口，可以根据需要获取量化或未量化的潜在空间表示
    def encode_to_latent(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """Projects a batch of waveforms to latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Embeddings, either quantized or not.
        """
        emb = self._encode_to_unquantized_latent(x)
        if not quantize:
            return emb
        else:
            codes = self.quantizer.encode(emb)
            return self.decode_latent(codes)

    # 将量化码 codes 解码为重构的音频波形
    def decode(self, codes: torch.Tensor):
        """Decode the given codes to a reconstructed representation.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        state = self._streaming_state
        emb = self.decode_latent(codes)
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            if state is None:
                (emb,) = self.decoder_transformer(emb)
            else:
                assert state.graphed_tr_dec is not None
                (emb,) = state.graphed_tr_dec(emb)
        with self._context_for_encoder_decoder:
            out = self.decoder(emb)
        # out contains extra padding added by the encoder and decoder
        return out

    # 法将量化码 codes 解码为连续的潜在空间表示
    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)


class WrapperCompressionModel(CompressionModel[State]):
    """Base API for CompressionModel wrappers that do not depend on external frameworks."""

    def __init__(self, model: CompressionModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        return self.model.forward(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode(codes)

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode_latent(codes)

    def set_num_codebooks(self, n: int):
        self.model.set_num_codebooks(n)

    @property
    def quantizer(self):
        return self.model.quantizer

    @property
    def channels(self) -> int:
        return self.model.channels

    @property
    def frame_rate(self) -> float:
        return self.model.frame_rate

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def cardinality(self) -> int:
        return self.model.cardinality

    @property
    def num_codebooks(self) -> int:
        return self.model.num_codebooks

    @property
    def total_codebooks(self) -> int:
        return self.model.total_codebooks
