# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch


from .client_utils import make_log
from .models import loaders, MimiModel, LMModel, LMGen


def log(level: str, msg: str):
    print(make_log(level, msg))


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


# 这个装饰器自动为类生成一些方法，例如 __init__、__repr__ 等，简化了代码。
@dataclass
class ServerState:
    # 模型实例，用于语音编码和解码
    mimi: MimiModel
    # SentencePiece 分词器，用于处理文本
    text_tokenizer: sentencepiece.SentencePieceProcessor
    # LLM Token 生成
    lm_gen: LMGen
    # 异步锁，用于同步对共享资源的访问，避免竞态条件
    lock: asyncio.Lock

    def __init__(
        self,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        device: str | torch.device,
    ):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = LMGen(lm)

        self.device = device

        # 计算音频帧大小：sample_rate 是音频采样率, frame_rate 是帧率
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        # 创建异步锁
        self.lock = asyncio.Lock()
        # 启动 Mimi 模型和语言模型的流式处理
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    # 预热模型，以减少首次请求的延迟
    def warmup(self):
        for chunk in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device
            )
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        torch.cuda.synchronize()

    async def handle_chat(self, request):
        # 创建一个 WebSocket 响应对象
        ws = web.WebSocketResponse()
        # 准备 WebSocket 连接
        await ws.prepare(request)

        # 接收循环，负责接收客户端发送的 WebSocket 消息
        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                log("info", "connection closed")

        # Opus 循环，负责处理接收到的音频数据，进行编码、解码和文本生成
        async def opus_loop():
            all_pcm_data = None

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    be = time.time()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size :]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            log("info", f"text token '{_text}'")
                            await ws.send_bytes(msg)
                    log("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")

        # 发送循环，负责将处理结果发送回客户端
        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        log("info", "accepted connection")
        # 初始化 close 变量，用于控制循环的结束
        close = False
        # 使用异步锁 self.lock 保护共享资源，确保同一时间只有一个连接可以使用 Mimi 和 LMGen
        async with self.lock:
            # 创建 Opus 写入流，用于将生成的音频数据编码为 Opus 格式
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            # 创建 Opus 读取流，用于解码客户端发送的 Opus 音频数据
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            # 重置 Mimi 模型和语言模型生成器的流式处理状态，为新的连接做准备
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            # 发送握手消息，表示连接建立成功
            await ws.send_bytes(b"\x00")
            # 并发运行三个异步循环：recv_loop、opus_loop 和 send_loop
            await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        log("info", "done with connection")
        return ws


def main():

    # ... 参数解析 ...

    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument(
        "--gradio-tunnel", action="store_true", help="Activate a gradio tunnel."
    )
    parser.add_argument(
        "--gradio-tunnel-token",
        help="Provide a custom (secret) token here to keep getting the same URL.",
    )

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument(
        "--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi."
    )
    parser.add_argument(
        "--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi."
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into, defaults Moshiko. "
        "Use this to select a different pre-trained model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )

    # 解析命令行参数
    args = parser.parse_args()
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ""
    # ... 设置 Gradio tunnel (可选)，方便用户通过公共 URL 访问服务器 ...
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            log(
                "error",
                "Cannot find gradio which is required to activate a tunnel. "
                "Please install with `pip install gradio`.",
            )
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # ... 加载 Mimi 模型 ...
    log("info", "loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    log("info", "mimi loaded")

    # ... 加载分词器 ...
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    # ... 加载 Moshi 语言模型 ...
    log("info", "loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, args.device)
    log("info", "moshi loaded")

    # ... 创建 Web 应用程序 ...
    # 创建 ServerState 实例，并将加载的模型、分词器和设备信息传入
    state = ServerState(mimi, text_tokenizer, lm, args.device)
    log("info", "warming up the model")
    state.warmup()
    # 创建一个 aiohttp Web 应用程序
    app = web.Application()
    """
    将 /api/chat 路由添加到应用程序，并将其绑定到 state.handle_chat 函数。 
    这意味着当客户端访问 /api/chat 路径时，会调用 handle_chat 函数处理 WebSocket 连接
    """
    app.router.add_get("/api/chat", state.handle_chat)

    # ... 设置静态文件服务(可选)，以便客户端可以访问 Web UI ...
    static_path: None | str = None
    if args.static is None:
        log("info", "retrieving the static content")
        dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        static_path = str(dist)
    elif args.static != "none":
        # When set to the "none" string, we don't serve any static content.
        static_path = args.static
    if static_path is not None:

        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        log("info", f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )

    # # ... 设置 SSL(可选)，以便服务器可以通过 HTTPS 协议提供服务 ...
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    # ... 启动 Web 服务器 ...
    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel("localhost", args.port, tunnel_token, None)
        log(
            "info",
            f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.",
        )
        log(
            "info",
            "Note that this tunnel goes through the US and you might experience high latency in Europe.",
        )
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
