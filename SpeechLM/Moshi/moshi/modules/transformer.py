# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer model, with streaming support, + CUDA Graphable.
Optimized for inference.

See `StreamingTransformer` for more information.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import typing as tp

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..utils.compile import no_compile
from .gating import make_gating
from .rope import RotaryEmbedding
from .streaming import StreamingModule, StreamingContainer


# 在进行归一化之前将输入转换为float32，计算后再转换回原始类型
class LayerNormF32(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_f32 = input.float()
        out_f32 = super().forward(x_f32)
        return out_f32.to(input.dtype)


# 实现均方根归一化 RMSNorm
def _rms_norm(
    x: torch.Tensor,
    alpha: torch.Tensor,
    dtype: tp.Optional[torch.dtype],
    eps: float,
):
    assert x.dim() == 3, f"RMSNorm expects 3D inputs but got {x.shape}"
    x_dtype = x.dtype
    if dtype is not None:
        x = x.to(dtype)
    var = eps + torch.mean(x**2, dim=2, keepdim=True)
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
    return y


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: tp.Optional[torch.dtype] = None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.alpha = nn.Parameter(
            torch.full((1, 1, dim), 1.0, requires_grad=True, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor):
        return _rms_norm(x, self.alpha, self.dtype, self.eps)


# 实现 Layer scale，用于缩放 残差连接 的输出
class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.

    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): If True, expect `[*, C]` shaped tensors, otherwise, `[*, C, T]`.
        device (torch.device or str, optional): Device on which to initialize the module.
        dtype (torch.dtype, optional): dtype to use to initialize the module.
    """

    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channel_last = channel_last
        # torch.full 用于创建一个形状为 (channels,) 的张量, 并将其所有元素初始化为 init 值
        self.scale = nn.Parameter(
            torch.full(
                (channels,), init, requires_grad=True, device=device, dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


#  工厂函数，根据给定的norm_type创建不同的归一化层
def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """为transformer编码器层创建归一化模块。

    参数:
        norm_type (str): 归一化方法。
        dim (int): 归一化层的维度。
        **kwargs (dict): 归一化层的其他参数。

    返回:
        nn.Module: 归一化模块。
    """
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "layer_norm_f32":
        kwargs.pop("dtype", None)
        return LayerNormF32(dim, eps=1e-8, **kwargs)
    elif norm_type in {"rms_norm"}:
        return RMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm_f32"}:
        kwargs.pop("dtype", None)
        return RMSNorm(dim, eps=1e-8, dtype=torch.float, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# 创建正弦位置 embedding
def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """创建正弦位置嵌入，形状为 `[B, T, C]`。

    参数:
        positions (torch.Tensor): 位置的 LongTensor。
        dim (int): 嵌入的维度。
        max_period (float): 余弦/正弦函数的最大周期。
        dtype (torch.dtype 或 str): 生成嵌入时使用的数据类型。
    返回:
        torch.Tensor: 正弦位置嵌入"""
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full(
        [], max_period, device=positions.device, dtype=dtype
    )  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


"""将多个线性层应用于给定的输入张量 x。每个时间步使用不同的权重集。
参数解析：
    num_linear: 线性层的数量, 即可能的时间步数。
    weight: 权重张量，形状为 [num_linear * chout, chin]。
    x: 输入张量，形状为 [B, T, C]，其中 B 是批次大小, T 是时间步数, C 是特征维度。
    offset: 当前时间步的偏移量, 用于解码时逐步提供时间步"""


def multi_linear(
    num_linear: int,
    weight: torch.Tensor,
    x: torch.Tensor,
    offset: int,
):
    B, T, C = x.shape
    ys = []
    chout, chin = weight.shape
    """ 把原来的weight的输出'chout'分成num_linear份
    即每个时间步使用不同的权重"""
    weight = weight.view(num_linear, -1, chin)
    """遍历每个时间步 t, 使用全连接层 F.linear(input, weight, bias=None)
    将当前时间步的输入 x[:, t] 与对应的权重 weight[t + offset] 进行线性变换
    """
    for t in range(T):
        # 对当前时间步的输入进行线性变换，结果形状为 (B, out_dim)
        y = F.linear(x[:, t], weight[t + offset])
        ys.append(y)
    # 使用 torch.stack 将所有时间步的结果堆叠成一个形状为 [B, T, chout] 的输出张量 out
    out = torch.stack(ys, 1)
    return out


# 设置或禁用模型中的注意力上下文范围（时间步数）。用于控制注意力机制的感受野
def set_attention_context(model: nn.Module, context: tp.Optional[int] = None) -> None:
    """停用或更改模型中的上下文跨度（以时间步为单位）
    参数:
        model (nn.Module): 需要查找注意力机制的模型。
        context (int 或 None): 新的临时上下文值。
    注意：这不是一个上下文管理器，而是一个永久更改上下文的普通函数。
    最初，它是一个上下文管理器，但在使用激活检查点时导致了一些有趣的问题，
    使得前向传播和反向传播之间的上下文不一致"""
    for module in model.modules():
        if isinstance(module, StreamingMultiheadAttention):
            module.context = context


# 命名元组，用于存储键 (keys)、值 (values) 和位置 (positions) 信息
class KVCacheResult(tp.NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T)
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions)


class RingKVCache:
    """高效的流式 KVCache, 兼容 Cuda Graph。
    它维护一个固定大小的缓存, 并在新的 KV 对到来时覆盖最旧的条目。
    参数:
        num_heads (int): 注意力机制中的头数。
        dim_per_head (int): 每个头的维度。
        device (torch.device): 初始化缓存的设备。
        dtype (torch.dtype): 缓存使用的数据类型"""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        # capacity: 缓存的最大容量，即可以存储多少个时间步的键值对
        self.capacity = capacity
        # 实际存储键值对的张量, 2 代表 K 和 V
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )
        # 指向下一个要写入缓存的位置的指针。它是一个标量张量，初始值为 0
        self.end_offset = torch.zeros(1, device=device, dtype=torch.long)

    """将 end_offset 重置为 0, 清空缓存
    zero_ 是一个张量的方法，用于将张量中的所有元素设置为零。
    这个方法是原地操作(in-place operation),
    即它会直接修改调用它的张量，而不会创建新的张量。"""

    def reset(self):
        self.end_offset.zero_()

    # 接收新的键 (k) 和值 (v)，并将它们添加到缓存中
    def complete(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape
        """计算新的键值对在缓存中的索引。 
        torch.arange(T) 生成从 0 到 T-1 的序列，
        加上 end_offset 得到在整个序列中的位置"""
        indexes = (
            torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype)
            + self.end_offset
        )
        # 对 capacity 取模，将其映射到环形缓存的有效范围内
        indexes = indexes % self.capacity
        """将新的键值对写入缓存
        tensor.index_copy_(dim, index, source)是 PyTorch 中的一个张量方法, 
            dim: 指定在哪个维度上进行索引复制。
            index: 一个一维张量, 包含要复制的索引位置。
            source: 源张量, 从中复制数据。
        用于将源张量中的数据复制到目标张量的指定索引位置。
        这个方法是原地操作(in-place operation), 
        即它会直接修改目标张量，而不会创建新的张量"""
        self.cache[0].index_copy_(2, indexes, k)
        self.cache[1].index_copy_(2, indexes, v)
        # 更新 end_offset 指针，指向下一个要写入的位置
        self.end_offset.add_(T)

        # 获取完整的键和值
        keys = self.cache[0]
        values = self.cache[1]

        # 生成一个从 0 到 capacity-1 的索引序列，用于计算位置信息
        indexes = torch.arange(
            self.capacity, device=self.end_offset.device, dtype=torch.long
        )
        # 当 end_offset 小于 capacity时，表示cache中还有没写入的数据，把他们索引标记为无效。
        # 当 end_offset 大于 capacity时，invalid 总为 True，也就是所有数据都可用。
        invalid = indexes >= self.end_offset
        # end_offset 表示当前最新的数据写入在环形缓存中的实际索引
        end_index = self.end_offset % self.capacity
        # delta 是每个索引相对于 end_offset 的偏移量
        delta = indexes - end_index

        """计算每个kv对应的时间步位置。
        positions 数组中的值表示缓存中每个位置对应原始序列中的时间步
        根据 delta 和 end_offset 计算每个键值对在原始序列中的位置。
        核心思想是将环形缓存中的索引映射回原始序列中的时间步。
        例如, 当self.end_offset为 5, capacity为 4 时,
        delta 为 [-1, 0, 1, 2], 
        positions 为 [4, 5, 2, 3], 
        表示当前的数据存在 cache 中的第2个位置, 
        而第三个和四个位置是之前的数据"""
        positions = torch.where(
            delta <= 0,
            # 当 delta <= 0 时，表示当前索引在 end_index 之前或等于 end_index
            self.end_offset + delta,
            # 当 delta > 0 时，表示当前索引在 end_index 之后，需要减去 capacity 来调整到正确的原始时间步
            self.end_offset + delta - self.capacity,
        )
        # 将无效条目的位置设置为 -1
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        return KVCacheResult(keys, values, positions)


# 存储多头注意力模块的流式状态，包括 KV 缓存和偏移量
@dataclass
class _MHAState:
    kv_cache: RingKVCache
    # 表示当前处理的时间步偏移量，存储在 GPU 上
    offset: torch.Tensor
    # 表示当前处理的时间步偏移量，存储在 CPU 上
    offset_cpu: int

    def reset(self):
        self.kv_cache.reset()
        self.offset.zero_()
        self.offset_cpu = 0


class StreamingMultiheadAttention(StreamingModule[_MHAState]):
    """实现了一个支持流式处理的 Multihead Attention 模块，
    它使用 RingKVCache 来存储过去的键和值，从而实现因果关系和高效的内存使用
    类似于 `nn.MultiheadAttention`，但支持流式处理和因果评估。
    参数:
        embed_dim (int): 投影的维度。
        num_heads (int): 头的数量。
        causal (bool): 自动应用因果掩码。
        context (int, 可选): 注意力可以访问的时间步数。
            当因果为真时，可以访问 `context` 个过去的时间步；当非因果时，
            可以访问 `context // 2` 个过去的时间步和相同数量的未来时间步。
        rope (`RotaryEmbedding`, 可选): 使用的旋转位置嵌入。
        weights_per_step (int): 每个时间步使用不同的权重。如果非零，对应于时间步数。
        device (torch.device, 可选): 初始化的设备。
        dtype (torch.dtype, 可选): 使用的数据类型"""

    _fsdp_final = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        weights_per_step: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.num_heads = num_heads

        out_dim = embed_dim
        out_dim = 3 * embed_dim
        # mult 用于调整输入和输出投影层的输出维度
        mult = 1
        """weights_per_step 表示每个时间步是否使用不同的权重。
        如果非零，表示每个时间步使用不同的权重"""
        self.weights_per_step = weights_per_step
        if weights_per_step:
            mult = weights_per_step
        # 创建输入投影层 in_proj
        in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False, **factory_kwargs)
        """保存输入投影层的权重和偏置 (mult * out_dim, embed_dim)
        在 PyTorch 中, nn.Linear 层的权重矩阵的形状
        遵循 (out_features, in_features) 的规则。
        这意味着权重矩阵的第一维表示输出特征的维度，第二维表示输入特征的维度"""
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias
        # 创建输出投影层 out_proj
        self.out_proj = nn.Linear(
            embed_dim, mult * embed_dim, bias=False, **factory_kwargs
        )

    # 初始化流式多头注意力机制的状态
    def _init_streaming_state(self, batch_size: int) -> _MHAState:
        # 确定缓存容量
        if self.context is None:
            if self.weights_per_step:
                capacity = self.weights_per_step
            else:
                raise RuntimeError(
                    "Cannot create a streaming KVCache without a context to estimate capacity."
                )
        else:
            capacity = self.context
        device = self.in_proj_weight.device
        # TODO: the following estimation will not work great with FSDP.
        dtype = self.in_proj_weight.dtype
        # 计算每个头的维度
        dim_per_head = self.embed_dim // self.num_heads
        # 创建 RingKVCache
        kv_cache = RingKVCache(
            batch_size, self.num_heads, dim_per_head, capacity, device, dtype
        )
        return _MHAState(
            kv_cache,
            offset=torch.zeros(1, device=device, dtype=torch.long),
            offset_cpu=0,
        )

    # 将新的 k 和 v 添加到 RingKVCache 中，并获取完整的 k, v 和 pos_k (键的位置信息)
    def _complete_kv(self, k, v) -> KVCacheResult:
        state = self._streaming_state
        if state is None:
            return KVCacheResult.from_kv(k, v)
        else:
            return state.kv_cache.complete(k, v)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        state = self._streaming_state
        T = query.shape[1]

        if state is None:
            offset = torch.zeros(1, device=query.device, dtype=torch.long)
            offset_cpu = 0
        else:
            assert self.causal, "Streaming only available for causal"
            offset = state.offset
            offset_cpu = state.offset_cpu

        # 将输入查询 query 投影到查询 q、键 k 和值 v 的高维空间中
        if self.weights_per_step:
            projected = multi_linear(
                self.weights_per_step, self.in_proj_weight, query, offset_cpu
            )
        else:
            projected = nn.functional.linear(query, self.in_proj_weight)
        q, k, v = rearrange(
            # p=3：表示有三个部分（查询、键、值）
            projected,
            "b t (p h d) -> p b h t d",
            p=3,
            h=self.num_heads,
        )

        # 应用旋转位置编码（RoPE）
        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)

        # 调用 _complete_kv 方法完成键和值的处理, pos_k是 position
        k, v, pos_k = self._complete_kv(k, v)
        """如果启用因果关系，则计算注意力偏置 attn_bias 实际上就是一个掩码矩阵(mask matrix)。
        这个掩码矩阵用于在多头注意力机制中控制哪些位置是可以关注的, 哪些位置是不能关注的。
        在因果模式下, 这个掩码矩阵确保了每个时间步的查询 query
        只能关注当前时间步及其之前的时间步, 而不能看到未来的信息"""
        if self.causal:
            pos_k = pos_k.view(1, -1)
            pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(
                -1, 1
            )
            # 计算查询位置和键位置之间的差值，表示查询相对于键的时间步差
            delta = pos_q - pos_k
            # 确保键的位置是有效的（即 pos_k >= 0），并且查询的位置不早于键的位置（即 delta >= 0）
            attn_bias = (pos_k >= 0) & (delta >= 0)
            """如果设置了上下文长度 self.context,
            则进一步限制 delta 必须小于 self.context,
            以确保查询只能关注有限的未来时间步"""
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
        else:
            attn_bias = None
        # 计算注意力
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
        # 恢复形状
        x = rearrange(x, "b h t d -> b t (h d)")
        # 线性层
        if self.weights_per_step:
            x = multi_linear(self.weights_per_step, self.out_proj.weight, x, offset_cpu)
        else:
            x = self.out_proj(x)
        # 更新状态
        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        return x


# 存储 Transformer 层的流式状态，包括偏移量
@dataclass
class _LayerState:
    offset_cpu: int

    def reset(self):
        self.offset_cpu = 0


# 实现了一个支持流式处理的 Transformer 层，包含了 StreamingMultiheadAttention、前馈网络 (FFN) 和归一化层
class StreamingTransformerLayer(StreamingModule[_LayerState]):
    """实现了一个支持流式处理和因果评估的  Transformer 层，
    包含了 StreamingMultiheadAttention、前馈网络 (FFN) 和归一化层
    参数:
        d_model (int): 数据的维度。
        num_heads (int): 头的数量。
        dim_feedforward (int): 前馈模块的中间维度。
        causal (bool): 自动应用因果掩码。
        context (int, 可选): 因果掩码的感受野，如果为 None 则为无限。
        custom (bool): 使用自定义的多头注意力实现，用于测试或基准测试。
        rope (`RotaryEmbedding`, 可选): 使用的旋转位置嵌入。
        norm (str): 使用的归一化方法。目前仅支持 'layer_norm'。
        layer_scale (float, 可选): 如果不为 None, 将使用给定初始值的 LayerScale。
        gating (str): 如果提供，将用特殊的门控机制（如 GLU、GSiGLU 等）替换前馈网络。
        weights_per_step (int): 每个时间步使用不同的权重。如果非零，应对应于可能的时间步数。
        skip_self_attn: 如果为 True, 跳过自注意力模块和归一化。
        device (torch.device, 可选): 初始化的设备。
        dtype (torch.dtype, 可选): 使用的数据类型"""

    _fsdp_final = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        norm: str = "layer_norm",
        layer_scale: tp.Optional[float] = None,
        gating: str = "none",
        weights_per_step: int = 0,
        activation=F.gelu,
        skip_self_attn: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Redefine self_attn to our streaming multi-head attention
        attn_kwargs: tp.Dict[str, tp.Any] = {
            "embed_dim": d_model,
            "num_heads": num_heads,
        }
        if not skip_self_attn:
            self.self_attn: StreamingMultiheadAttention = StreamingMultiheadAttention(
                causal=causal,
                context=context,
                rope=rope,
                weights_per_step=weights_per_step,
                **attn_kwargs,  # type: ignore
                **factory_kwargs,  # type: ignore
            )  # type: ignore
            self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        # Redefine feedforward layers to expose bias parameter
        self.weights_per_step = weights_per_step
        self.gating: tp.Optional[nn.Module] = None
        self.linear1: tp.Optional[nn.Module] = None
        self.linear2: tp.Optional[nn.Module] = None
        self.activation = activation
        self.skip_self_attn = skip_self_attn

        if isinstance(dim_feedforward, list):
            assert dim_feedforward
            assert len(dim_feedforward) == weights_per_step, (
                "Length of dim_feedforward must match weights_per_step,"
                f" got {len(dim_feedforward)} != {weights_per_step}"
            )
        if gating == "none":
            assert (
                not weights_per_step
            ), "weights_per_step without gating not supported for now."
            assert not isinstance(
                dim_feedforward, list
            ), "List dim_feedforward without gating not supported for now."
            self.linear1 = nn.Linear(
                d_model, dim_feedforward, bias=False, **factory_kwargs
            )
            self.linear2 = nn.Linear(
                dim_feedforward, d_model, bias=False, **factory_kwargs
            )
        else:
            self.linear1 = None
            self.linear2 = None
            if weights_per_step:
                if isinstance(dim_feedforward, int):
                    dim_feedforward = [dim_feedforward] * weights_per_step
                assert isinstance(dim_feedforward, list), dim_feedforward
                self.gating = nn.ModuleList(
                    [
                        make_gating(gating, d_model, dim, **factory_kwargs)
                        for dim in dim_feedforward
                    ]
                )
            else:
                assert isinstance(dim_feedforward, int)
                self.gating = make_gating(
                    gating, d_model, dim_feedforward, **factory_kwargs
                )

        self.layer_scale_1: nn.Module
        self.layer_scale_2: nn.Module
        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore

    def _init_streaming_state(self, batch_size: int) -> _LayerState:
        return _LayerState(offset_cpu=0)

    # 前馈网络
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        state = self._streaming_state
        offset = 0
        if state is not None:
            offset = state.offset_cpu
        x_orig = x
        x = self.norm2(x)
        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            update = self.linear2(self.activation(self.linear1(x)))
        else:
            if self.weights_per_step:
                assert isinstance(self.gating, nn.ModuleList)
                B, T, D = x.shape
                ys = []
                for t in range(T):
                    y = self.gating[offset + t](x[:, t : t + 1])
                    ys.append(y)
                update = torch.cat(ys, dim=1)
            else:
                update = self.gating(x)
        return x_orig + self.layer_scale_2(update)

    # 自注意力
    def _sa_block(self, x: torch.Tensor):
        if self.skip_self_attn:
            return x
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, x, x)
        return x_orig + self.layer_scale_1(update)

    def forward(self, x: torch.Tensor):
        with ExitStack() as stack:
            if x.device.type != "cuda":
                stack.enter_context(no_compile())
            x = self._sa_block(x)
            x = self._ff_block(x)
            state = self._streaming_state
            if state:
                state.offset_cpu += x.shape[1]
            return x


#  存储 Transformer 模型的流式状态，包括偏移量
@dataclass
class _TransformerState:
    offset: torch.Tensor

    def reset(self):
        self.offset.zero_()


# 支持流式处理
class StreamingTransformer(StreamingModule[_TransformerState]):
    """实现了一个由多个 StreamingTransformerLayer 堆叠而成的
    完整 Transformer 模型，支持流式处理和因果评估
    参数:
        d_model (int): 数据的维度。
        num_heads (int): 头的数量。
        dim_feedforward (int): 前馈模块的中间维度。
        causal (bool): 自动应用因果掩码。
        context (int, 可选): 因果掩码的感受野，如果为 None 则为无限。
        layer_scale (float, 可选): 如果不为 None，将使用给定初始值的 LayerScale。
        positional_embedding (str): 位置嵌入策略（sin, rope, sin_rope 或 none）。
        max_period (float): 时间嵌入的最大周期。
        positional_scale (float): 位置嵌入的缩放因子，设置为 0 以禁用。
        layer_class: (`StreamingTransformerLayer` 的子类): 用于初始化层的类，允许在 AudioCraft 之外进行进一步的定制。
        device (torch.device, 可选): 初始化的设备。
        dtype (torch.dtype, 可选): 使用的数据类型。
        **kwargs: 详见 `StreamingTransformerLayer`。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        betas: tp.Optional[tp.Tuple[float, float]] = None,
        layer_class: tp.Type[StreamingTransformerLayer] = StreamingTransformerLayer,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.betas = betas

        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope: tp.Optional[RotaryEmbedding] = None
        if self.positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                layer_class(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def _init_streaming_state(self, batch_size: int) -> _TransformerState:
        device = next(self.parameters()).device
        return _TransformerState(offset=torch.zeros(1, device=device, dtype=torch.long))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape

        state = self._streaming_state
        if state is None:
            offset = torch.zeros(1, dtype=torch.long, device=x.device)
        else:
            offset = state.offset

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offset.view(-1, 1, 1)
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period, dtype=x.dtype
            )
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        if state is not None:
            state.offset.add_(T)
        return x


class ProjectedTransformer(StreamingContainer):
    """这是一个包装器类，它在 Transformer 模型的输入和输出上添加了可选的投影层。
    这允许在输入和输出维度与 Transformer 的内部维度不同时进行适配，支持多个输出。
    参数:
        input_dimension (int): 输入的维度。
        output_dimensions (tuple[int]): 输出的维度。
        d_model (int): Transformer 的内部维度。
        conv_layout (bool): 如果为 True, 期望形状为 `[B, C, T]` 的张量，否则为[B, T, C]。
            同样，输出也将具有相同的布局。
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tp.Tuple[int, ...],
        d_model: int,
        *,
        conv_layout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(d_model=d_model, **kwargs)
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.conv_layout = conv_layout
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(
                    nn.Linear(d_model, output_dimension, bias=False)
                )

    def forward(self, x, *args, **kwargs):
        if self.conv_layout:
            x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, *args, **kwargs)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            if self.conv_layout:
                y = y.transpose(1, 2)
            ys.append(y)
        return ys
