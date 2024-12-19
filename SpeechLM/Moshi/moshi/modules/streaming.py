# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Streaming module API that should be implemented by all Streaming components,
"""

import abc
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import math
import typing as tp
from torch import nn
import torch


# 协议类，定义了一个名为 reset 的方法
class Resetable(tp.Protocol):
    def reset(self) -> None:
        pass


"""定义 State 为一个类型变量，并指定其边界为 Resetable, 
可以在泛型类或泛型函数中使用 State, 
并且确保传入的类型必须实现 Resetable 协议"""
State = tp.TypeVar("State", bound=Resetable)


class StreamingModule(abc.ABC, nn.Module, tp.Generic[State]):
    """流组件的通用 API。

    每个流组件都有一个流状态(streaming state), 该状态只是一个字典dict[str, Tensor]。
    按照约定，每个张量的第一个维度必须是批量大小。
    不要在键名中使用dots, 因为这会与子模块(如state_dict中的子模块)冲突。

    如果 self._is_streaming 为 True, 组件应在 self._streaming_state 中使用并记住适当的状态。

    要将流组件设置为流状态，使用

        with module.streaming():
            ...

    这将在退出上下文管理器时自动重置流状态。
    这也会自动传播到所有流子模块。

    一些模块还可能实现 StreamingModule.flush 方法，尽管这比较复杂，
    因为所有父模块必须是 StreamingModule 并且也实现该方法才能正常工作。参见后面的 StreamingSequential。
    """

    def __init__(self) -> None:
        super().__init__()
        """Python 3.10 引入的类型注解新语法 | 来表示联合类型(Union Types)。
        联合类型允许变量可以是多个不同类型的其中之一。
        在self._streaming_state 可以是 State 类型的对象，也可以是 None"""
        self._streaming_state: State | None = None
        self._streaming_propagate: bool = True

    # @property 装饰器用于将类的方法转换为属性，使得可以通过点操作符直接访问该方法，而不需要使用括号调用
    # 返回模型是否处于流式处理模式
    @property
    def is_streaming(self):
        return self._streaming_state is not None

    # 设置是否将流状态传播到子模块
    def set_streaming_propagate(self, streaming_propagate: bool):
        self._streaming_propagate = streaming_propagate

    """递归地将函数 fn 应用于自身和所有子模块。遍历所有子模块，
    并根据 _streaming_propagate 的值决定是否将 fn 应用于子模块。
    这是管理和同步自身和子模块状态的关键"""

    def _apply_named_streaming(self, fn: tp.Any):
        def _handle_module(prefix: str, module: nn.Module, recurse: bool = True):
            """参数：
            prefix: str: 模块名称的前缀, 用于构建完整的模块路径。
            module: nn.Module: 当前处理的模块。
            recurse: bool = True: 一个布尔值, 表示是否递归处理子模块"""
            propagate = True
            if isinstance(module, StreamingModule):
                if module._streaming_propagate:
                    fn(prefix, module)
                else:
                    propagate = False
            if not recurse:
                return
            if propagate:
                for name, child in module.named_children():
                    _handle_module(prefix + "." + name, child)

        _handle_module("", self, recurse=False)
        for name, child in self.named_children():
            _handle_module(name, child)

    # 开始流式处理，初始化流状态
    def _start_streaming(self, batch_size: int):
        def _start_streaming(name: str, module: StreamingModule):
            module._streaming_state = module._init_streaming_state(batch_size)

        self._apply_named_streaming(_start_streaming)

    # 停止流式处理，将流状态重置为 None
    def _stop_streaming(self):
        def _stop_streaming(name: str, module: StreamingModule):
            module._streaming_state = None

        self._apply_named_streaming(_stop_streaming)

    # 抽象方法，需要子类实现，用于创建初始流状态
    @abc.abstractmethod
    def _init_streaming_state(self, batch_size: int) -> State: ...

    #  永久进入流式处理模式
    def streaming_forever(self, batch_size: int):
        self._start_streaming(batch_size)

    """上下文管理器，用于临时进入流式处理模式。
    在 with 块内，模型处于流式处理状态；退出 with 块后，流状态会被自动重置。 
    这是推荐的用法，确保正确地进入和退出流模式"""

    @contextmanager
    def streaming(self, batch_size: int):
        self._start_streaming(batch_size)
        try:
            yield
        finally:
            self._stop_streaming()

    # 重置流状态。这通常在每个序列处理结束后调用
    def reset_streaming(self):
        def _reset(name: str, module: StreamingModule):
            state = module._streaming_state
            if state is None:
                raise ValueError(
                    f"Trying to reset streaming, but {name} wasn't streaming."
                )
            state.reset()

        self._apply_named_streaming(_reset)

    # 获取完整的流状态，包括子模块的流状态
    def get_streaming_state(self) -> dict[str, tp.Any]:
        state: dict[str, tp.Any] = {}

        def _add(name: str, module: StreamingModule):
            state[name] = module._streaming_state

        self._apply_named_streaming(_add)
        return state

    # 设置完整的流状态，包括子模块的流状态
    def set_streaming_state(self, state: dict[str, tp.Any]):
        state = dict(state)

        def _set(name: str, module: StreamingModule):
            if name in state:
                # 检查 state 字典中是否存在 name 对应的流状态
                module._streaming_state = state[name]
                # 如果存在，设置模块的流状态并从 state 字典中移除该条目
                state.pop(name)
            else:
                raise RuntimeError(f"Expected to find a streaming state for {name}.")

        self._apply_named_streaming(_set)
        if state:
            raise RuntimeError(f"Some states were not consumed: {list(state.keys())}")


"""简单的数据类, 用于表示空状态。
它没有任何成员变量, reset 方法也是空操作。 
这主要用于那些不需要维护流状态的 StreamingModule 的子类"""


@dataclass
class _NullState:
    pass

    def reset(self) -> None:
        pass


"""这个类继承自 StreamingModule[_NullState]。 
它的作用是为不需要维护复杂流状态的模块提供一个简单的基类。 
因为它使用 _NullState 作为泛型参数，
所以它的 _init_streaming_state 方法 simply 返回一个 _NullState 实例，
这意味着它实际上没有维护任何状态。 这对于构建模块的层次结构很有用，
即使某些模块不需要流状态，也可以保持一致的接口"""


class StreamingContainer(StreamingModule[_NullState]):
    def _init_streaming_state(self, batch_size: int) -> _NullState:
        return _NullState()


@dataclass
class _StreamingAddState:
    previous_x: torch.Tensor | None = None
    previous_y: torch.Tensor | None = None

    def reset(self):
        self.previous_x = None
        self.previous_y = None


class StreamingAdd(StreamingModule[_StreamingAddState]):
    def _init_streaming_state(self, batch_size: int) -> _StreamingAddState:
        return _StreamingAddState()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # 非流式模式, 直接返回 x + y
        if self._streaming_state is None:
            return x + y
        # 流式模式
        else:
            # 将当前输入 x 和 y 与前一个时间步的输入 prev_x 和 prev_y 在最后一个维度上拼接
            prev_x = self._streaming_state.previous_x
            prev_y = self._streaming_state.previous_y
            if prev_x is not None:
                x = torch.cat([prev_x, x], dim=-1)
            if prev_y is not None:
                y = torch.cat([prev_y, y], dim=-1)
            # 计算 x 和 y 在最后一个维度上的最小长度 m_l
            m_l = min(x.shape[-1], y.shape[-1])
            # 将 x 和 y 超出 m_l 的部分保存, 用于下一个时间步的计算
            self._streaming_state.previous_x = x[..., m_l:]
            self._streaming_state.previous_y = y[..., m_l:]
            # 返回 x 和 y 在 m_l 范围内的元素之和
            return x[..., :m_l] + y[..., :m_l]


@dataclass
class _StreamingConvState:
    previous: torch.Tensor | None = None

    def reset(self):
        self.previous = None


# 支持流式处理的一维卷积操作
class RawStreamingConv1d(nn.Conv1d, StreamingModule[_StreamingConvState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """确保 padding 为 0,
        这意味着 padding 需要在调用 RawStreamingConv1d 之前手动处理。
        这是因为流式处理中, padding 的处理方式与普通卷积不同"""
        assert self.padding[0] == 0, "Padding should be handled outside."
        """确保 stride 不大于 kernel_size。
        这保证了卷积操作在流式处理过程中不会跳过任何输入数据"""
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."

    # 回一个 _StreamingConvState 实例，用于存储前一个时间步的输入数据
    def _init_streaming_state(self, batch_size: int) -> _StreamingConvState:
        return _StreamingConvState()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        stride = self.stride[0]
        # 计算考虑了 dilation 的有效 kernel size
        kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        # 非流式模式, 直接调用父类 nn.Conv1d 的 forward 方法进行卷积操作
        if self._streaming_state is None:
            return super().forward(input)
        # 流式模式
        else:
            # 由于潜在的重叠，获取前一个时间步的输入数据
            previous = self._streaming_state.previous
            # 将前一个时间步的输入数据与当前输入数据在最后一个维度（时间维度）上拼接
            if previous is not None:
                input = torch.cat([previous, input], dim=-1)
            B, C, T = input.shape
            # 核心逻辑: 计算可以进行完整卷积操作的帧数, 它计算了当前有多少个完整的卷积窗口可以计算
            num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
            """计算用于更新 _streaming_state.previous 的偏移量 offset 
            offset 表示从输入数据的起始位置到可以进行完整卷积操作的最后一帧的末尾位置的距离.
            这个距离用于确定哪些部分的数据已经用于卷积计算,
            哪些部分的数据需要保存到 _streaming_state.previous 中,
            以便在下一个时间步继续使用"""
            offset = num_frames * stride
            """将拼接后的输入数据中, 
            input[..., offset:]：表示从 offset 位置到输入数据末尾的部分,
            也就是超出当前可以计算的帧数的部分保存到 _streaming_state.previous 中,
            用于下一个时间步的计算。 
            这确保了下一轮卷积可以利用之前步骤剩余的数据"""
            self._streaming_state.previous = input[..., offset:]
            # 如果可以进行完整卷积操作的帧数大于 0，则进行卷积计算
            if num_frames > 0:
                """计算进行卷积操作所需的输入数据的长度
                num_frames - 1: 表示可以进行完整卷积操作的帧数减去 1,
                因为最后一帧的末尾位置已经在 offset 中计算过了。
                    `* stride`: 表示这些帧之间的总滑动距离。
                    `+ kernel`: 表示最后一个卷积核的大小"""
                input_length = (num_frames - 1) * stride + kernel
                # 调用父类 nn.Conv1d 的 forward 方法进行卷积操作
                out = super().forward(input[..., :input_length])
            else:
                # 如果不能进行完整卷积操作，则返回一个空张量。 这表示当前输入不足以产生一个完整的卷积输出
                out = torch.empty(
                    B, self.out_channels, 0, device=input.device, dtype=input.dtype
                )
            return out


@dataclass
class _StreamingConvTrState:
    partial: torch.Tensor | None = None

    def reset(self):
        self.partial = None


# 支持流式处理的一维转置卷积操作
# 转置卷积，也称为反卷积，用于增大输入特征图的空间维度
class RawStreamingConvTranspose1d(
    nn.ConvTranspose1d, StreamingModule[_StreamingConvTrState]
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert self.dilation[0] == 1, "No dilation for now"
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."
        assert self.output_padding[0] == 0, "Output padding not supported."

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvTrState:
        return _StreamingConvTrState()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        B, C, T = x.shape
        stride = self.stride[0]
        kernel = self.kernel_size[0]
        if self._streaming_state is None:
            return super().forward(x)
        else:
            # 如果输入的时间维度 T 为 0，则返回一个空的输出张量
            if T == 0:
                return torch.empty(
                    B, self.out_channels, 0, device=x.device, dtype=x.dtype
                )
            out = super().forward(x)
            OT = out.shape[-1]
            partial = self._streaming_state.partial
            if partial is not None:
                """由于潜在的重叠，转置卷积的最右侧输出尚未准备好输出, 因为它将接收来自下一个输入帧的贡献。
                在这里我们恢复这些 `partial` 输出帧。我们知道 `partial` 张量的第一个时间步对应于 `out` 的第一个时间步，
                因为在 `out` 的第一个时间步之前的所有内容都已经清空了"""
                PT = partial.shape[-1]
                if self.bias is not None:
                    out[..., :PT] += partial - self.bias[:, None]
                else:
                    out[..., :PT] += partial
            """输入是 T, 输出是 S * (T - 1) + K。
            下一帧左侧的偏移量将是 S * T,
            因此从 0 到 S * T 之间的所有内容都已准备好输出，我们需要
            在内部状态中保留超出这部分的内容，即 S * (T - 1) + K - S * T = K - S"""
            invalid_steps = kernel - stride
            partial = out[..., OT - invalid_steps :]
            out = out[..., : OT - invalid_steps]
            self._streaming_state.partial = partial
            return out


# 测试函数 test，用于验证卷积层和反卷积层在不同参数设置下的表现
def test():
    torch.manual_seed(1234)
    device = "cpu"
    if torch.cuda.is_available():
        # 避免在单精度浮点数卷积中进行的 CUDA 优化。
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = "cuda:0"

    kernel_sizes = [1, 3, 4, 8, 15, 16]
    strides = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    chin = 6
    chout = 12

    """遍历卷积参数：对每一对 kernel 和 stride,
    itertools.product 是一个笛卡尔积函数,
    它会返回两个或多个输入迭代对象的所有可能组合"""
    for kernel, stride in itertools.product(kernel_sizes, strides):
        if stride > kernel:
            continue
        conv = RawStreamingConv1d(chin, chout, kernel, stride).to(device)
        convtr = RawStreamingConvTranspose1d(chout, chin, kernel, stride).to(device)

        # 遍历输入长度：对每个输入长度 length，生成随机输入数据 x，
        # 并进行前向传播得到 y，再进行反卷积得到 z
        for length in [4, 8, 32, 54, 65, 128, 1043]:
            print(f"ksize {kernel} strides {stride} len {length}")
            if length < kernel:
                continue
            batch_size = 3
            x = torch.randn(batch_size, chin, length).to(device)
            y = conv(x)
            z = convtr(y)
            # 分块处理：对不同的 chunk_size，将输入数据分成多个小块进行前向和反向传播，验证结果的一致性
            for chunk_size in [1, 3, 5, 8]:
                ys = []
                zs = []
                """使用 with 语句进入卷积层 conv 和反卷积层 convtr 的流处理模式。
                这通常用于处理连续的数据流，确保每次处理的上下文信息被正确传递"""
                with conv.streaming(batch_size), convtr.streaming(batch_size):
                    for offset in range(0, length, chunk_size):
                        chunk = x[..., offset : offset + chunk_size]
                        ys.append(conv(chunk))
                        zs.append(convtr(ys[-1]))
                # 将所有分块处理的结果拼接成完整的输出 y_stream 和 z_stream
                y_stream = torch.cat(ys, dim=-1)
                z_stream = torch.cat(zs, dim=-1)
                # 截取 y 和 z，使其长度与 y_stream 和 z_stream 一致。
                y = y[..., : y_stream.shape[-1]]
                z = z[..., : z_stream.shape[-1]]
                assert y.shape == y_stream.shape, (y.shape, y_stream.shape)
                delta = (y_stream - y).norm() / y.norm()
                # 计算 y 和 y_stream 之间的差异 delta，并确保 delta 小于 1e-6
                assert delta <= 1e-6, delta
                # 计算输出帧数 num_frames，并验证其与 y_stream 的形状是否一致
                num_frames = int((length - kernel) / stride) + 1
                assert num_frames == y_stream.shape[-1]

                assert z.shape == z_stream.shape, (z.shape, z_stream.shape)
                delta = (z_stream - z).norm() / z.norm()
                assert delta <= 1e-6, (delta, (z_stream - z).abs().mean(dim=(0, 1)))


if __name__ == "__main__":
    with torch.no_grad():
        test()
