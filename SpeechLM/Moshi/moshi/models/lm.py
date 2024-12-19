# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
import logging
import typing as tp

import torch
from torch import nn

from ..utils.sampling import sample_token
from ..utils.compile import CUDAGraphed
from ..modules.streaming import StreamingContainer, StreamingModule
from ..modules.transformer import (
    StreamingTransformer,
    create_norm_fn,
)


logger = logging.getLogger(__name__)


class ScaledEmbedding(nn.Embedding):
    """在嵌入层之后可以选择性地应用层归一化(Layer Normalization).
    它还处理了一个特殊情况, 即当输入为特定值zero_idx时, 输出会被强制设置为零
    参数:
        norm (bool): 如果为 True, 则在嵌入层后使用层归一化。
        zero_idx (int): 特殊值，表示输出应精确为 0。
    """

    def __init__(self, *args, norm: bool = False, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        # 检查输入是否等于 zero_idx，并生成一个布尔掩码 is_zero
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        # 将输入钳制到非负范围
        input = input.clamp(min=0)
        # 调用父类的 forward 方法
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        # 将 is_zero 为 True 的位置的输出设置为零
        y = torch.where(is_zero[..., None], zero, y)
        return y


class LMModel(StreamingContainer):
    """基于 Transformer 的多code流语言模型。

    参数:
        n_q (int): 并行建模的输入流数量。
        dep_q (int): Depformer 中并行流的数量。
        card (int): 音频码本的词汇量大小。
        text_card (int): 文本词汇表大小。
        dim (int): Transformer 编码器的维度。
        num_heads (int): Transformer 编码器的头数。
        hidden_scale (int): Transformer 编码器隐藏前馈层的缩放因子。
        norm (str): 归一化方法。
        norm_emb (bool): 是否对嵌入进行归一化。
        bias_proj (bool): 是否在输出投影中使用偏置。
        depformer_*: 用于 Depformer Transformer 的参数，其他参数将共享。
        depformer_multi_linear (bool): 如果为 True, 则使用每个码本的一个线性层, 将主 Transformer 的输出投影到 Depformer 的潜在空间。
        depformer_dim_feedforward (int | list[int] | None): 如果为 None, 默认为 hidden_scale * depformer_dim。
        existing_text_padding_id (bool): 如果为 True, 将使用不同的标记作为初始文本标记和文本填充标记。
        same_initial (bool): 如果为 True, 使用相同的初始标记用于文本和音频模式。
        **kwargs: Transformer 编码器的其他参数"""

    def __init__(
        self,
        delays: tp.List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: tp.Optional[int] = None,
        context: tp.Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        assert len(delays) == self.num_codebooks, "unexpected number of delays"
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.context = context
        kwargs["context"] = context
        """functools.partial 函数来创建一个部分应用函数(partial function)。
        它接受一个可调用对象(如函数)和一些固定的参数，
        返回一个新的可调用对象, 这个新对象在调用时会自动使用固定的参数"""
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )
        """嵌入层: 
        一个 nn.ModuleList, 包含 n_q 个 ScaledEmbedding 层, 每个音频码本对应一个。
        ScaledEmbedding 允许缩放嵌入向量，并可以进行可选的归一化
        self.card + 1 表示除了音频码本中的 self.card 个词汇之外，还需要额外增加一个特殊的标记。
        特殊的标记是 self.initial_token_id, 它的值等于 self.card。
        在模型的初始化阶段, 会使用 self.initial_token_id 来填充输入序列的第一个位置，表示序列的开始"""
        self.emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]
        )
        # 文本词汇表 + 填充标记（如果原始分词器中没有）
        extra_text = self.existing_text_padding_id is None
        # 与音频不同，这里我们允许模型输出特殊标记
        self.text_emb = EmbeddingFactory(text_card + 1, dim)
        self.text_linear = nn.Linear(dim, text_card + extra_text, bias=bias_proj)
        depformer_prefix = "depformer_"
        # 筛选出不以 "depformer_" 开头的参数，用于主 Transformer
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }
        # Temporal Transformer 用于处理文本和音频嵌入的组合输入
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        self.depformer_multi_linear = depformer_multi_linear
        # 复制主 Transformer 的参数作为 Depformer 的基础参数
        kwargs_dep = main_kwargs.copy()
        # 将 kwargs 中以 "depformer_" 开头的参数去掉前缀后添加到 kwargs_dep 中，覆盖或新增相应的参数
        kwargs_dep.update(
            {
                # 字典推导式
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            # 创建 Depformer 的输入投影线性层
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False) for _ in range(dep_q)]
            )
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False)]
            )
        """创建 Depformer 的音频码本嵌入层
        在预测最后一个码本 dep_q - 1 的输出时, 我们已经有了之前所有码本的输出, 
        因此不需要再将最后一个码本的输入嵌入到 Depformer 中"""
        self.depformer_emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
        )
        # 创建 Depformer 的文本嵌入层
        self.depformer_text_emb = EmbeddingFactory(text_card + 1, depformer_dim)
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)
        # 创建 Depformer 的 StreamingTransformer 实例
        self.depformer = StreamingTransformer(
            d_model=depformer_dim,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            device=device,
            dtype=dtype,
            **kwargs_dep,
        )
        # 禁用 Depformer 的流式状态传播
        self.depformer.set_streaming_propagate(False)
        # 我们将直接将接下来的线性层应用于 Depformer 的输出
        dim = depformer_dim

        # 创建一个包含 dep_q 个线性层的模块列表，用于音频码本输出投影
        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )

    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        """Token id for the start of sequence (text)."""
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        """Token id for text padding."""
        if self.existing_text_padding_id is None:
            return self.text_card
        else:
            return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """Token id for optionally marking the last padding step for a word."""
        return 0

    @property
    def zero_token_id(self) -> int:
        """Special value in the input tokens, indicating that no sampling should
        happen for that value, and no input should be given to the model."""
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2

    @property
    def device(self):
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        # 返回将被输入模型以预测第一个时间步的初始标记。
        # 输出形状将是 [B, K, 1]。
        device = next(iter(self.parameters())).device
        zero = torch.full(
            [1, 1, 1], self.zero_token_id, device=device, dtype=torch.long
        )
        special = torch.full_like(zero, self.initial_token_id)

        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special
        text_token = text_special
        audio_token = audio_token.expand(-1, self.num_audio_codebooks, -1)
        token = torch.cat([text_token, audio_token], dim=1)
        return token

    def forward_text(
        self,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks."
        input_sequence = sequence
        input_ = None
        # 遍历每个音频 codebook, 计算音频嵌入并累加到 input_
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](
                input_sequence[:, cb_index + self.audio_offset]
            )
            input_ = audio_emb if input_ is None else input_ + audio_emb
        # 计算文本嵌入并累加到 input_
        text_emb = self.text_emb(input_sequence[:, 0])
        input_ = text_emb if input_ is None else input_ + text_emb
        # 将 input_ 传递给 Transformer 模型进行处理
        transformer_out = self.transformer(input_)

        # 如果配置了归一化层，则对 Transformer 输出进行归一化
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        # 通过线性层生成文本 logits 并调整形状
        text_logits = self.text_linear(transformer_out)
        text_logits = text_logits[:, None]
        return transformer_out, text_logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, K, S = sequence.shape
        # Depformer 每次只处理一个码本和一个时间步
        assert (
            K == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        assert (
            S == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        # 确保 transformer_out 的形状为 [B, 1, dim], 即 Temporal Transformer 的输出只包含一个时间步
        assert (
            transformer_out.shape[1] == 1
        ), "Transformer out should be a for a single step."
        last_token_input: tp.Optional[torch.Tensor] = None
        """获取上一个时间步的音频码本输出或文本嵌入:
        如果 depformer_cb_index 为 0, 表示当前正在预测第一个音频码本, 
        则使用 self.depformer_text_emb 将文本嵌入作为输入。
        否则, 使用 self.depformer_emb 将上一个时间步的音频码本输出作为输入"""
        depformer_input = transformer_out
        """depformer_in是一个线性层, 用于将 Transformer 输出转换为 Depformer 的输入"""
        if self.depformer_multi_linear:
            depformer_input = self.depformer_in[depformer_cb_index](depformer_input)
        else:
            depformer_input = self.depformer_in[0](depformer_input)
        """输入序列的sequence第一个是text embedding, 所以使用depformer_text_emb,
        之后的序列是audio embedding, 所以使用depformer_emb"""
        if depformer_cb_index == 0:
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        # 将投影后的 transformer_out 和上一个时间步的音频码本输出或文本嵌入相加，得到 Depformer 的输入
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1
        # depformer_input 的形状为 [B, 1, depformer_dim]。
        # depformer 的流状态确保运行正确的层。
        dep_output = self.depformer(depformer_input)
        # 使用 self.linears 中对应码本的线性层将 Depformer 的输出投影到音频码本的词汇空间，得到音频码本的 logits
        logits = self.linears[depformer_cb_index](dep_output)
        # 返回音频码本的 logits，形状为 [B, 1, 1, card]
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits


@dataclass
class _LMGenState:
    cache: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_depth: CUDAGraphed
    offset: int = 0

    def reset(self):
        self.offset = 0


# LMGen 类是用于流式生成音频和文本的模块。它使用 LMModel 进行推理, 并支持采样和贪婪解码
class LMGen(StreamingModule[_LMGenState]):
    def __init__(
        self,
        lm_model: LMModel,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        check: bool = False,
    ):
        assert not lm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = lm_model
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.check = check
        self.max_delay = max(
            lm_model.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            lm_model.delays, device=lm_model.device, dtype=torch.long
        )

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 2),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        )

        disable = lm_model.device.type != "cuda"
        graphed_main = CUDAGraphed(lm_model.forward_text, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        return _LMGenState(cache, initial, graphed_main, graphed_depth)

    # 输入 token，形状为 (B, Ki, 1)，其中 Ki 是用户提供的码本数量 (不包括 Depformer 生成的码本)
    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor) -> torch.Tensor | None:
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        #  input_tokens是用户提供的文本和音频标记, 形状为 (B, Ki, 1), Ki 是用户提供的码本数量(不包括 Depformer 生成的码本)
        assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
        B, Ki, S = input_tokens.shape
        # 只支持逐个时间步输入
        assert S == 1, "Only support being given steps one by one."
        """模型总共有 num_codebooks = n_q + 1 个码本需要预测, 
        其中 dep_q 个音频码本由 Depformer 自动生成,文本码本也需要用户提供, 
        所以用户需要提供的码本数量就是 num_codebooks - dep_q - 1"""
        needed_tokens = lm_model.num_codebooks - lm_model.dep_q - 1
        assert (
            Ki == needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

        # 获取缓存大小 CT
        CT = state.cache.shape[2]

        # 遍历用户提供的每个码本
        for q_other in range(input_tokens.shape[1]):
            # 计算码本的索引 k 和延迟 delay
            k = lm_model.dep_q + 1 + q_other
            delay = lm_model.delays[k]
            # 计算码本在缓存中的写入位置 write_position
            write_position = (state.offset + delay) % CT
            # 将用户提供的标记写入缓存的对应位置
            state.cache[:, k, write_position : write_position + 1] = input_tokens[
                :, q_other
            ]

        # 计算当前时间步在缓存中的位置 position
        position = state.offset % CT
        """对于每个码本：如果当前时间步小于等于码本的延迟, 则将初始标记写入缓存的对应位置。
        这是因为在序列的开头, 延迟的码本还没有足够的历史信息, 因此需要使用初始标记来填充"""
        for k, delay in enumerate(lm_model.delays):
            # 仅在最开始时，我们扩展初始令牌以供延迟的声学令牌使用，因为这些令牌没有合适的值可取。
            if state.offset <= delay:
                state.cache[:, k, position] = state.initial[:, k, 0]
        input_ = state.cache[:, :, position : position + 1]

        # 检查模型输入 (可选)
        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            # 断言 input_ 中不包含未生成的标记 lm_model.ungenerated_token_id
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            # 断言音频码本的标记值小于等于 lm_model.card
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            # 断言文本码本的标记值小于等于 lm_model.text_card
            assert (input_[:, :1] <= lm_model.text_card).all()

        """Temporal Transformer 预测文本标记:
            使用 state.graphed_main (即 LMModel.forward_text 方法的 CUDAGraphed 版本) 
        进行推理, 得到 transformer_out 和 text_logits"""
        transformer_out, text_logits = state.graphed_main(input_)
        """Shape of text_logits should be [B, K_text=1, T=1, Card_text]
        使用 sample_token 函数对 text_logits 进行采样，得到文本标记 text_token"""
        text_token = sample_token(
            text_logits.float(),
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )
        assert text_token.dim() == 3, text_token.shape
        assert text_token.shape[2] == 1
        assert text_token.shape[1] == 1, "Only one text stream supported."
        text_token = text_token[:, 0, 0]  # shape is [B]
        """用 Depformer 预测音频标记:使用 state.graphed_depth
        (即 LMGen.depformer_step 方法的 CUDAGraphed 版本)进行推理, 得到音频标记 audio_tokens"""
        audio_tokens = state.graphed_depth(text_token, transformer_out)

        # 将生成的标记写入缓存, 确保不会覆盖prompt tokens, 只覆盖未生成的tokens。

        # 将 state.offset 加 1，更新当前时间步
        state.offset += 1
        # 计算当前时间步在缓存中的位置 position
        position = state.offset % CT
        # 将生成的文本标记 text_token 写入缓存的对应位置
        state.cache[:, 0, position] = text_token
        # 将生成的音频标记 audio_tokens 写入缓存的对应位置
        state.cache[:, 1 : lm_model.dep_q + 1, position] = audio_tokens

        # 如果 state.offset 小于等于 self.max_delay, 则返回 None, 表示还没有生成足够多的标记来输出
        if state.offset <= self.max_delay:
            return None
        # 否则, 从缓存中提取生成的 tokens
        B = state.cache.shape[0]
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        index = (
            ((state.offset - self.max_delay + gen_delays_cuda) % CT)
            .view(1, -1, 1)
            .expand(B, -1, 1)
        )
        out = state.cache.gather(dim=2, index=index)
        return out

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        # 当前时间步的文本标记，形状为 (B,)
        (B,) = text_token.shape
        # 初始化上一个时间步的音频码本输出 prev_token 为当前时间步的文本标记 text_token
        prev_token = text_token
        lm_model = self.lm_model
        # 创建一个空列表 depformer_tokens，用于存储生成的音频标记
        depformer_tokens: list[torch.Tensor] = []
        # lm_model.depformer 不是流式模式
        assert not lm_model.depformer.is_streaming
        # 上下文管理器将 lm_model.depformer 设置为流式模式，并设置批量大小为 B
        with lm_model.depformer.streaming(B):
            for cb_index in range(lm_model.dep_q):
                # 将上一个时间步的音频码本输出 prev_token 转换为形状为 (B, 1, 1) 的张量
                input_ = prev_token[:, None, None]
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                )
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]  # shape is B
                depformer_tokens.append(next_token)
                prev_token = next_token

        assert len(depformer_tokens) == lm_model.dep_q, (
            len(depformer_tokens),
            lm_model.dep_q,
        )
        out = torch.stack(depformer_tokens, dim=1)
        assert out.shape == (B, lm_model.dep_q), out.shape
        return out
