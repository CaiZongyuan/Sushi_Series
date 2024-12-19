# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp

from einops import rearrange, repeat
import torch
from torch import nn
from torch import distributed
import torch.nn.functional as F


# 命名元组，用于组织前向传播的输出
class _CodebookForwardResult(tp.NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    metrics: tp.Dict[str, torch.Tensor]


class _VQForwardResult(tp.NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    loss: torch.Tensor
    metrics: tp.Dict[str, torch.Tensor]


# 用于原地更新指数移动平均 (EMA)
def _ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


# 从样本中随机采样指定数量的向量
def _sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


# 计算向量使用情况的熵
def _compute_entropy(usage: torch.Tensor) -> torch.Tensor:
    # Usage is some unnormalized distribution.
    proba = usage / usage.sum()
    p_log_p = torch.where(
        proba == 0, zero_scalar(usage.device), proba * torch.log(proba)
    )
    return -p_log_p.sum()


# 检查是否使用了分布式训练
def _is_distributed() -> bool:
    # Checks if we need to use distributed routines.
    return distributed.is_initialized() and distributed.get_world_size() > 1


# 运行 K-Means 算法初始化码本
def _run_kmeans(
    samples: torch.Tensor, num_clusters: int, num_iters: int = 50
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    # Kmeans algorithm used to initialize the codebooks.
    dim = samples.shape[-1]
    # 初始化聚类中心
    means = _sample_vectors(samples, num_clusters)
    bins = None

    for _ in range(num_iters):
        # 计算样本到聚类中心的距离
        dists = torch.cdist(samples[None], means[None], p=2)[0]
        # 将样本分配到最近的聚类
        buckets = dists.argmin(dim=-1)
        # 统计每个聚类的样本数量
        bins = torch.bincount(buckets, minlength=num_clusters)
        # 标记空聚类
        zero_mask = bins == 0
        # 避免除以零
        bins.clamp_(min=1)

        # 创建新的聚类中心
        new_means = torch.zeros_like(means)
        # 计算新的聚类中心
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        # 对新的聚类中心进行归一化
        new_means /= bins[..., None]
        # 重新采样一些样本
        resampled = _sample_vectors(samples, num_clusters)
        # 替换空聚类的中心
        means = torch.where(zero_mask[..., None], resampled, new_means)

    assert bins is not None
    return means, bins


# 返回一个零标量，避免同步点
def zero_scalar(device) -> torch.Tensor:
    """Returns a 0. value on the given device without introducing a synchronization point."""
    return torch.zeros([1], device=device)[0]


class EuclideanCodebook(nn.Module):
    """使用欧几里得距离的码本。

    参数:
        dim (int): 维度。
        codebook_size (int): 码本大小。
        decay (float): 码本上的指数移动平均衰减因子。
        epsilon (float): 用于数值稳定的 epsilon 值。
        threshold_usage_ratio (float): 定义了在均匀分布下，当某个聚类中心的使用率低于其期望使用率的某个比例时，该聚类中心将被替换。这个比例不依赖于批次大小等。
        replaced_usage_ratio (float): 当替换某个聚类中心时，使用此初始聚类中心使用率，以避免聚类中心被过快替换。
        check_unused_every (int): 每 `check_unused_every` 次迭代检查一次未使用的聚类中心。这是为了避免过多的同步点。

    缓冲区:
        cluster_usage (torch.Tensor): 每批次的聚类使用率的 EMA, 例如, 这将依赖于批次大小等。
        embedding_sum (torch.Tensor): 分配给每个聚类的点的和的 EMA。特别是, 可以通过 `cluster_usage` 归一化来获得实际的聚类中心。
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        replaced_usage_ratio: float = 1.0,
        check_unused_every: int = 5,
    ):
        super().__init__()
        self.decay = decay

        self.dim = dim
        self.codebook_size = codebook_size

        self.epsilon = epsilon
        self.threshold_usage_ratio = threshold_usage_ratio
        self.replaced_usage_ratio = replaced_usage_ratio
        self.check_unused_every = check_unused_every
        self._next_unused_check = check_unused_every
        self._cached_initialized = False

        # 创建 buffers
        # self.register_buffer 方法是在 PyTorch 的 nn.Module 类中定义的
        # 缓冲区不参与梯度计算，也不会被优化器更新
        # 指示码本是否已初始化
        self.register_buffer("_initialized", torch.tensor([False], dtype=torch.float))
        # 存储每个码字的使用次数 (EMA)
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        embedding = torch.zeros(codebook_size, dim)
        # 存储分配给每个码字的向量的和 (EMA)
        self.register_buffer("embedding_sum", embedding)
        # 缓存计算出的码本向量
        self.register_buffer("_embedding", None, persistent=False)

    #  处理旧版本模型的 state_dict，进行键名映射
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        mappings = {
            "inited": "_initialized",
            "cluster_size": "cluster_usage",
            "embed_avg": "embedding_sum",
            "embed_sum": "embedding_sum",
        }
        for old_name, new_name in mappings.items():
            old_name = prefix + old_name
            if old_name in state_dict:
                value = state_dict.pop(old_name)
                if new_name is not None:
                    state_dict[prefix + new_name] = value
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    """如果 _embedding 为 None,
    则根据 embedding_sum 和 cluster_usage 计算码本向量，
    并将其缓存到 _embedding 中"""

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            embedding = (
                self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            )
            self.register_buffer("_embedding", embedding, persistent=False)
            return embedding
        return self._embedding

    @property
    def initialized(self) -> bool:
        """返回码本是否已初始化。 使用缓存机制避免重复读取 _initialized buffer"""
        if not self._cached_initialized:
            self._cached_initialized = self._initialized.item()
        return self._cached_initialized

    # 使用 K-Means 算法初始化码本
    def _init_embedding(self, data: torch.Tensor) -> None:
        # Initialize the codebook, e.g. using kmeans.
        if self.initialized:
            return

        # 支持分布式训练，在 rank 0 的主进程上运行 K-Means，然后将结果广播到其他进程
        rank = 0
        if _is_distributed():
            """distributed.get_rank() 是 PyTorch 分布式训练中的一个方法，
            用于获取当前进程的排名(rank)。
            在分布式训练中，多个进程协同工作以加速模型训练，
            每个进程都有一个唯一的排名，从 0 开始编号。
            通常, rank 0 被视为主进程，负责执行一些特定的任务，
            如模型参数的初始化和日志记录"""
            rank = distributed.get_rank()
            # 首先收集形状，以防不同 GPU 的有效批量大小不同
            # 然后收集实际内容
            if rank == 0:
                """distributed.get_world_size() 获取分布式训练中所有进程的数量。
                [None] * distributed.get_world_size() 是 Python 中创建列表的一种方式，
                用于初始化一个长度为distributed.get_world_size()的列表, 其中每个元素都是None"""
                other_shapes: tp.List[torch.Size] = [None] * distributed.get_world_size()  # type: ignore
                """data.shape 是当前进程的数据张量的形状
                将所有进程的数据形状收集到主进程的other_shapes列表中"""
                distributed.gather_object(data.shape, other_shapes)
                other_data: tp.List[torch.Tensor] = [
                    torch.empty(shape, device=data.device, dtype=data.dtype)
                    for shape in other_shapes
                ]
                # 将所有进程的数据收集到主进程中，并拼接成一个完整的数据张量
                distributed.gather(data, other_data)
                data = torch.cat(other_data, dim=0)
            else:
                """如果只传入一个参数，表示当前进程将对象发送到主进程，
                但不接收任何返回值。这是因为在非主进程中，
                我们通常不需要存储收集的结果，只需要将数据发送出去即可"""
                distributed.gather_object(data.shape)
                distributed.gather(data)

        if rank == 0:
            embedding, cluster_usage = _run_kmeans(data, self.codebook_size)
            self.embedding_sum.data.copy_(embedding * cluster_usage[:, None])
            self.cluster_usage.data.copy_(cluster_usage)
            self._initialized.data.fill_(1)
        # Make sure all buffers across workers are in sync after initialization
        self._broadcast_buffers()

    # 在分布式环境中同步 buffers
    def _broadcast_buffers(self) -> None:
        if _is_distributed():
            for buffer in self.buffers():
                distributed.broadcast(buffer, 0)

    """替换不常用的码字（即质心）
    具体来说, 它根据传入的布尔掩码 mask, 确定哪些质心需要被替换, 
    并从输入样本 samples 中采样新的质心来替换它们"""

    def _replace_expired_codes(self, samples: torch.Tensor, mask: torch.Tensor) -> None:
        # 采样新的质心
        new_vectors = _sample_vectors(samples, self.codebook_size)
        # 计算替换使用的阈值
        replace_cluster_usage = (
            self.replaced_usage_ratio * self.cluster_usage.sum() / self.codebook_size
        )
        """更新质心的累加和
        mask 是一个布尔张量, 表示哪些质心需要被替换。
        mask[:, None] 在 mask 的最后一个维度上增加一个新的维度, 使其形状与 new_vectors 匹配。
        对于需要替换的质心(mask 为 True), 用 replace_cluster_usage * new_vectors 更新 self.embedding_sum。
        对于不需要替换的质心(mask 为 False), 保持 self.embedding_sum 不变。"""
        self.embedding_sum[:] = torch.where(
            mask[:, None], replace_cluster_usage * new_vectors, self.embedding_sum
        )
        # 更新质心的使用次数
        self.cluster_usage[:] = torch.where(
            mask, replace_cluster_usage, self.cluster_usage
        )
        """举例：
        1.采样新的质心：
            new_vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        2.计算替换使用的阈值：
            self.cluster_usage.sum() = 10 + 2 + 5 + 1 = 18
            replace_cluster_usage = 0.1 * 18 / 4 = 0.45
        3.更新质心的累加和：
            mask = [False, True, False, True]
            mask[:, None] = [[False, False], [True, True], [False, False], [True, True]]
            replace_cluster_usage * new_vectors = [[0.45, 0.9], [1.35, 1.8], [2.25, 2.7], [3.15, 3.6]]
            self.embedding_sum 更新后：
                [[0.0, 0.0],
                [1.35, 1.8],
                [4.0, 4.0],
                [3.15, 3.6]]
        4.更新质心的使用次数：
            self.cluster_usage 更新后 [10.0, 0.45, 5.0, 0.45]
        """

    # 检查不常用的码字，并进行替换
    def _check_expired_codes(self, batch_samples: torch.Tensor) -> torch.Tensor:
        # 如果 self.initialized 为 False，表示模型尚未初始化，直接返回一个零标量
        if not self.initialized:
            return zero_scalar(batch_samples.device)
        # self._next_unused_check是一个计数器，用于控制检查频率, 每次调用此方法时, 计数器减 1
        self._next_unused_check -= 1
        # 如果计数器大于 0, 表示当前不需要进行检查，直接返回一个零标量
        if self._next_unused_check > 0:
            return zero_scalar(batch_samples.device)
        """不会检查每次迭代以避免有太多的同步点。
        如果计数器小于等于 0, 重置计数器为self.check_unused_every, 
        表示下次检查的时间间隔"""
        self._next_unused_check = self.check_unused_every
        """计算阈值
        self.threshold_usage_ratio 是一个预定义的比例值。
        self.cluster_usage 是一个张量，记录了每个质心的使用次数。
        self.codebook_size 是码本（即质心集合）的大小。
        计算每个质心的使用次数的平均值, 并乘以 self.threshold_usage_ratio, 
        得到阈值 threshold_cluster_usage"""
        threshold_cluster_usage = (
            self.threshold_usage_ratio * self.cluster_usage.sum() / self.codebook_size
        )
        # 找出不常用的质心
        expired_codes = self.cluster_usage < threshold_cluster_usage

        assert batch_samples.dim() == 2
        # 替换不常用的质心
        self._replace_expired_codes(batch_samples, mask=expired_codes)
        # 调用 _broadcast_buffers 方法，同步更新模型的缓冲区
        self._broadcast_buffers()

        return expired_codes.float().mean()

    # 展平所有维度，但保留最后一个维度，例如返回一个形状为 `[N, D]` 的向量
    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... d -> (...) d")
        return x

    # 将码字索引张量调整回原始形状
    """* 解包操作符将一个列表或元组中的元素作为单独的参数传递给函数。
    例如，如果有 shape = (2, 3, 4),
    那么 *shape[:-1] 会将 2 和 3 作为单独的参数传递给 view 方法，即 view(2, 3)"""

    # 恢复x原来的形状，但是去掉维度
    def _reshape_codes(self, codes: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return codes.view(*shape[:-1])

    # 将输入向量量化到最近的码字，返回码字索引
    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        # 将 x 中的每个向量投影到最近的质心，并返回其索引。
        # x 应该是形状为[N, D]的张量，其中 N 是输入向量的数量，D是维度。
        assert x.dim() == 2
        """计算两个张量中每对向量之间的欧几里得距离, p=2 表示欧几里得距离
        x[None] 是一种常用的操作，用于在张量的前面增加一个新的维度。
        假设 x 的形状为 [N, D],那么x[None] 的形状将变为 [1, N, D]"""
        dists = torch.cdist(x[None], self.embedding[None], p=2)[0]
        """dists.argmin(dim=-1) 找到最后一个维度的最小值索引
        即在最后一个维度（即码字维度）上找到最小距离的索引"""
        codes = dists.argmin(dim=-1)
        return codes

    #  将输入张量编码为码字索引
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """给定一个形状为 [*, D] 的张量 x, 返回一个形状为 [*] 的整数编码张量。
        编码定义为 x 中每个向量最近的质心的索引"""
        assert x.dtype.is_floating_point, f"Input should be floats, got {x.dtype}"
        shape = x.shape
        x = self._reshape_input(x)
        codes = self._quantize(x)
        codes = self._reshape_codes(codes, shape)
        return codes

    # 将码字索引解码为量化向量
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """给定一个形状为 `[*]` 的编码张量，返回一个形状为 `[*, D]` 的张量，
        该张量对应于每个编码索引关联的质心。
        """
        assert (
            not codes.dtype.is_floating_point
        ), f"Codes should be integers, got {codes.dtype}"
        """F.embedding 是 PyTorch 中的一个函数，用于根据索引查找嵌入向量。
        例如: codes 是 [1, 2, 0],
        self.embedding 是 [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]]。
        F.embedding(codes, self.embedding) 根据 codes 中的索引查找 self.embedding 中对应的质心：
        codes[0] = 1 对应 self.embedding[1] = [2.0, 2.0]
        codes[1] = 2 对应 self.embedding[2] = [4.0, 4.0]
        codes[2] = 0 对应 self.embedding[0] = [0.0, 0.0]
        最终返回的 quantized_vectors 是 [[2.0, 2.0], [4.0, 4.0], [0.0, 0.0]],
                                        形状为 [3, 2]"""
        quantized = F.embedding(codes, self.embedding)
        return quantized

    def forward(
        self, x: torch.Tensor, initialize: bool = True
    ) -> _CodebookForwardResult:
        shape = x.shape
        # 将输入张量 x 展平为二维张量 (*, dim)
        x = self._reshape_input(x)

        """如果模型处于训练模式且 initialize 为 True, 则初始化码本。
        使用 K-Means 算法初始化码本。 
        x.detach() 创建一个不参与梯度计算的副本，避免影响 K-Means 的结果。 
        注意，只有当之前的量化层已经初始化后，才会进行初始化，
        避免在同一批数据上重复运行 K-Means。"""
        if self.training and initialize:
            """如果 initialize 为 False, 我们不允许初始化这一层,
            并且后续代码将在一个全零的代码本上运行。
            这是因为之前的层已经使用批次数据进行了 k-means 初始化
            因此，残差主要为 0"""
            self._init_embedding(x.detach())
        # 使用 _quantize 方法将展平后的输入 x 量化到最近的码字，得到码字索引 flat_codes
        flat_codes = self._quantize(x)
        # 将 flat_codes 的形状恢复为原始输入 x 的形状（除了最后一个维度 dim）
        codes = self._reshape_codes(flat_codes, shape)
        # 使用 decode 方法将码字索引 codes 解码为量化向量 quantized
        quantized = self.decode(codes)
        # 创建一个空字典，用于存储指标
        metrics: tp.Dict[str, torch.Tensor] = {}

        # 如果模型处于训练模式，则更新码本和指标
        if self.training:
            # 检查是否有不常用的码字，并进行替换。 将替换比例存储在 expired 中
            expired = self._check_expired_codes(x)
            # 将 expired 添加到指标字典中
            metrics["rvq_expired"] = expired
            # 计算每个码字的使用次数，并使用 EMA 更新 self.cluster_usage
            cluster_usage = torch.zeros_like(self.cluster_usage)
            cluster_usage.scatter_add_(
                0, flat_codes, torch.ones_like(flat_codes, dtype=cluster_usage.dtype)
            )
            _ema_inplace(self.cluster_usage, cluster_usage, self.decay)

            if self.initialized:
                # 我们报告的熵是经过均匀分布的熵归一化后的值，
                # 这意味着当熵为 1 时，代码本被最优地利用。
                # 计算码本使用情况的熵，并将其添加到指标字典中
                metrics["rvq_entropy"] = _compute_entropy(
                    self.cluster_usage
                ) / math.log(self.codebook_size)
            # 计算分配给每个码字的向量的和，并使用 EMA 更新 self.embedding_sum
            embedding_sum = torch.zeros_like(self.embedding_sum)
            embedding_sum.scatter_add_(0, repeat(flat_codes, "n -> n d", d=self.dim), x)
            _ema_inplace(self.embedding_sum, embedding_sum, self.decay)
            #  清除缓存的码本向量，因为 embedding_sum 已经更新
            self.register_buffer("_embedding", None)

        return _CodebookForwardResult(quantized, codes, metrics)


class VectorQuantization(nn.Module):
    """向量量化实现。
    目前仅支持欧几里得距离。

    参数:
        dim (int): 维度
        codebook_size (int): 代码本大小
        codebook_dim (int): 代码本维度。如果未定义，则使用在 `dim` 中指定的维度。
        decay (float): 代码本上指数移动平均的衰减率。
        epsilon (float): 用于数值稳定的 epsilon 值。
        threshold_usage_ratio (float): 定义了集群使用率低于该阈值时替换质心的标准。这表示为质心在均匀分布下使用率的一个分数，因此不依赖于批次大小等。
        replaced_usage_ratio (float): 替换质心时，使用此初始质心使用率，以避免质心被过快替换。
        check_unused_every (int): 每 `check_unused_every` 次迭代检查一次未使用的质心。这是为了避免过多的同步点。
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            decay=decay,
            epsilon=epsilon,
            threshold_usage_ratio=threshold_usage_ratio,
            **kwargs,
        )
        self.codebook_size = codebook_size

    @property
    def embedding(self):
        return self._codebook.embedding

    @property
    def initialized(self):
        return self._codebook.initialized

    def _rearrange_input(self, x):
        x = rearrange(x, "b d n -> b n d")
        return x

    def _rearrange_output(self, quantized):
        quantized = rearrange(quantized, "b n d -> b d n")
        return quantized

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes `x` into discrete integer codes."""
        x = self._rearrange_input(x)
        x = self.project_in(x)
        codes = self._codebook.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts integer codes into quantized vectors."""
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        quantized = self._rearrange_output(quantized)
        return quantized

    def forward(self, x: torch.Tensor, initialize: bool = True) -> _VQForwardResult:
        x = self._rearrange_input(x)
        quantized, codes, metrics = self._codebook(x, initialize=initialize)

        if self.training:
            quantized = x + (quantized - x).detach()
            loss = F.mse_loss(x, quantized.detach())
        else:
            loss = zero_scalar(x.device)

        quantized = self.project_out(quantized)
        quantized = self._rearrange_output(quantized)

        return _VQForwardResult(quantized, codes, loss, metrics)


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers: int, codebook_offset: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        # 码本偏移量，用于在输出指标中区分不同层级的码本
        self.codebook_offset = codebook_offset

    def forward(
        self, x: torch.Tensor, n_q: tp.Optional[int] = None
    ) -> _VQForwardResult:
        """
        Args:
            x: 输入张量, 形状为 [B, C, T].
            n_q: 如果提供, 表示使用的量化层数。默认为所有层
        """
        # 初始化为零标量，用于存储最终的量化输出
        quantized_out = zero_scalar(x.device)
        residual = x

        all_losses = []
        all_codes = []
        all_metrics: tp.Dict[str, torch.Tensor] = {}

        # n_q 如果未提供，则使用所有量化层
        n_q = n_q or len(self.layers)
        # 初始化为 True，表示前一层是否已初始化
        previous_layer_is_initialized = True

        # 迭代 n_q 个量化层
        for i, layer in enumerate(self.layers[:n_q]):  # type: ignore
            if self.training:
                this_layer_is_initialized = layer.initialized
            """我们只允许在前一层已经在之前的迭代中初始化的情况下进行k-means初始化,
            这是为了避免在同一批次上学习后续的 k-means,
            这最终会导致批次耗尽并在 0 值上运行 k-means"""
            # 对残差向量 residual 进行量化
            quantized, codes, loss, metrics = layer(
                residual, initialize=previous_layer_is_initialized
            )
            if self.training:
                previous_layer_is_initialized = this_layer_is_initialized  # type: ignore

            # 不计算量化输出的梯度，使用 STE 技巧
            quantized = quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_codes.append(codes)
            all_losses.append(loss)

            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key] += value / n_q
                else:
                    all_metrics[key] = value / n_q
                all_metrics[key + f"_{i + self.codebook_offset}"] = value

        if self.training:
            # Solving subtle bug with STE and RVQ: https://github.com/facebookresearch/encodec/issues/25
            # STE 技巧，确保梯度正确回传
            quantized_out = x + (quantized_out - x).detach()

        out_losses, out_codes = map(torch.stack, (all_losses, all_codes))
        return _VQForwardResult(quantized_out, out_codes, out_losses, all_metrics)

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        """Encodes `x` into discrete integer codes. If `n_q` is provided, only uses the first `n_q` codebook levels."""
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:  # type: ignore
            # 对残差向量进行编码，得到码字索引
            indices = layer.encode(residual)
            # 将码字索引解码为量化向量
            quantized = layer.decode(indices)
            #  计算新的残差向量
            residual = residual - quantized
            # 将码字索引添加到列表中
            all_indices.append(indices)
        # 将码字索引列表转换为张量
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts the integer codes into quantized vectors."""
        quantized = zero_scalar(codes.device)
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            quantized = quantized + layer.decode(layer_codes)
        return quantized
