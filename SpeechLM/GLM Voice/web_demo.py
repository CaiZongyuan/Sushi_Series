import json
import os.path
import tempfile  # 创建临时文件
import sys
import re
import uuid  # 生成唯一标识符
import requests  # 发送 HTTP 请求
from argparse import ArgumentParser

import torchaudio
from transformers import WhisperFeatureExtractor, AutoTokenizer
from speech_tokenizer.modeling_whisper import (
    WhisperVQEncoder,
)  # 自定义的 Whisper VQ 编码器

# 添加自定义模块路径
sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

from speech_tokenizer.utils import extract_speech_token  # 提取语音 token 的工具函数

import gradio as gr  # 创建 Web 界面
import torch

# 正则表达式，用于匹配音频 token
audio_token_pattern = re.compile(r"<\|audio_(\d+)\|>")

from flow_inference import AudioDecoder  # 音频解码器
from audio_process import AudioStreamProcessor  # 音频流处理器

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="8888")
    parser.add_argument("--flow-path", type=str, default="THUDM/glm-4-voice-decoder")
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument(
        "--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer"
    )
    args = parser.parse_args()

    flow_config = os.path.join(args.flow_path, "config.yaml")  # 音频解码器配置文件
    flow_checkpoint = os.path.join(args.flow_path, "flow.pt")
    hift_checkpoint = os.path.join(args.flow_path, "hift.pt")
    glm_tokenizer = None
    device = "cuda"
    audio_decoder: AudioDecoder = None
    whisper_model, feature_extractor = None, None

    # 加载语音 tokenizer, Decoder 和 WhisperVQEncoder
    def initialize_fn():
        global audio_decoder, feature_extractor, whisper_model, glm_model, glm_tokenizer
        if audio_decoder is not None:
            return

        # 加载 GLM 分词器
        glm_tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )

        # Flow & Hift
        audio_decoder = AudioDecoder(
            config_path=flow_config,
            flow_ckpt_path=flow_checkpoint,
            hift_ckpt_path=hift_checkpoint,
            device=device,
        )

        # 加载 Whisper 语音编码器
        whisper_model = (
            WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)

    # 清理界面和状态
    def clear_fn():
        return [], [], "", "", "", None, None

    def inference_fn(
        temperature: float,  # 温度参数，用于控制生成文本的随机性
        top_p: float,  # 核采样参数，用于控制生成文本的多样性
        max_new_token: int,  # 生成的最大新token数量
        input_mode,  # 输入模式，可以是 "audio" 或 "text"
        audio_path: str | None,  # 音频文件路径，如果输入模式是音频时使用
        input_text: str | None,  # 文本输入，如果输入模式是文本时使用
        history: list[dict],  # 对话历史记录
        previous_input_tokens: str,  # 上一次输入的token
        previous_completion_tokens: str,  # 上一次生成的回复token
    ):

        # 音频输入模式
        if input_mode == "audio":
            assert audio_path is not None
            history.append({"role": "user", "content": {"path": audio_path}})
            # 提取音频中的 token
            audio_tokens = extract_speech_token(
                whisper_model, feature_extractor, [audio_path]
            )[0]
            if len(audio_tokens) == 0:
                raise gr.Error("No audio tokens extracted")
            # token 格式化
            audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            # 添加开始和结束标记
            audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
            user_input = audio_tokens
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

        else:  # 处理文本输入
            assert input_text is not None
            history.append({"role": "user", "content": input_text})
            user_input = input_text
            system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."

        # Gather history
        inputs = previous_input_tokens + previous_completion_tokens
        inputs = inputs.strip()
        if "<|system|>" not in inputs:
            inputs += f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        # 发送请求并处理响应
        with torch.no_grad():
            response = requests.post(
                # 发送请求到生成器
                "http://localhost:10000/generate_stream",
                data=json.dumps(
                    {
                        "prompt": inputs,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_token,
                    }
                ),
                stream=True,
            )
            # 初始化文本和音频 token 列表
            text_tokens, audio_tokens = [], []
            # 获取音频 token 的偏移量
            audio_offset = glm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
            # 获取结束token的ID
            end_token_id = glm_tokenizer.convert_tokens_to_ids("<|user|>")
            # 初始化完整的token列表
            complete_tokens = []
            # 初始化prompt语音特征
            prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
            # 初始化prompt语音token
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
            # 生成唯一标识符
            this_uuid = str(uuid.uuid4())
            # 初始化语音生成列表
            tts_speechs = []
            # 初始化梅尔特征列表
            tts_mels = []
            # 初始化前一个梅尔特征
            prev_mel = None
            # 初始化结束标志
            is_finalize = False
            # 块大小列表
            block_size_list = [25, 50, 100, 150, 200]
            # 块大小索引
            block_size_idx = 0
            # 当前块大小
            block_size = block_size_list[block_size_idx]
            # 初始化音频流处理器
            audio_processor = AudioStreamProcessor()
            # 遍历响应流
            for chunk in response.iter_lines():
                # 解析 token ID
                token_id = json.loads(chunk)["token_id"]
                # 如果是结束 token，设置结束标志
                if token_id == end_token_id:
                    is_finalize = True
                # 如果音频 token 数量达到块大小或结束
                if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                    if block_size_idx < len(block_size_list) - 1:
                        block_size_idx += 1
                        # 更新块大小
                        block_size = block_size_list[block_size_idx]
                    # 转换为张量
                    tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)

                    if prev_mel is not None:
                        # 更新 prompt 语音特征
                        prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                    # 生成语音和梅尔特征
                    tts_speech, tts_mel = audio_decoder.token2wav(
                        tts_token,
                        uuid=this_uuid,
                        prompt_token=flow_prompt_speech_token.to(device),
                        prompt_feat=prompt_speech_feat.to(device),
                        finalize=is_finalize,
                    )
                    prev_mel = tts_mel
                    # 处理音频
                    audio_bytes = audio_processor.process(
                        tts_speech.clone().cpu().numpy()[0], last=is_finalize
                    )

                    tts_speechs.append(tts_speech.squeeze())
                    tts_mels.append(tts_mel)
                    if audio_bytes:
                        # 生成音频流
                        yield history, inputs, "", "", audio_bytes, None
                    # 更新 prompt 语音 token
                    flow_prompt_speech_token = torch.cat(
                        (flow_prompt_speech_token, tts_token), dim=-1
                    )
                    audio_tokens = []
                if not is_finalize:
                    # 添加到完整 token 列表
                    complete_tokens.append(token_id)
                    if token_id >= audio_offset:
                        # 如果是音频token，添加到音频token列表
                        audio_tokens.append(token_id - audio_offset)
                    else:
                        text_tokens.append(token_id)
        # 拼接生成的语音
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
        # 解码完整的文本
        complete_text = glm_tokenizer.decode(
            complete_tokens, spaces_between_special_tokens=False
        )
        # 创建临时文件保存音频
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # 保存音频
            torchaudio.save(f, tts_speech.unsqueeze(0), 22050, format="wav")
        # 将音频路径添加到对话历史
        history.append(
            {"role": "assistant", "content": {"path": f.name, "type": "audio/wav"}}
        )
        # 将文本回复添加到对话历史
        history.append(
            {
                "role": "assistant",
                "content": glm_tokenizer.decode(
                    text_tokens, ignore_special_tokens=False
                ),
            }
        )
        # 生成最终结果
        yield history, inputs, complete_text, "", None, (22050, tts_speech.numpy())

    def update_input_interface(input_mode):
        if input_mode == "audio":
            return [gr.update(visible=True), gr.update(visible=False)]
        else:
            return [gr.update(visible=False), gr.update(visible=True)]

    # Create the Gradio interface
    with gr.Blocks(title="GLM-4-Voice Demo", fill_height=True) as demo:
        with gr.Row():
            temperature = gr.Number(label="Temperature", value=0.2)

            top_p = gr.Number(label="Top p", value=0.8)

            max_new_token = gr.Number(
                label="Max new tokens",
                value=2000,
            )

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            type="messages",
            scale=1,
        )

        with gr.Row():
            with gr.Column():
                input_mode = gr.Radio(
                    ["audio", "text"], label="Input Mode", value="audio"
                )
                audio = gr.Audio(
                    label="Input audio",
                    type="filepath",
                    show_download_button=True,
                    visible=True,
                )
                text_input = gr.Textbox(
                    label="Input text",
                    placeholder="Enter your text here...",
                    lines=2,
                    visible=False,
                )

            with gr.Column():
                submit_btn = gr.Button("Submit")
                reset_btn = gr.Button("Clear")
                output_audio = gr.Audio(
                    label="Play",
                    streaming=True,
                    autoplay=True,
                    show_download_button=False,
                )
                complete_audio = gr.Audio(
                    label="Last Output Audio (If Any)", show_download_button=True
                )

        gr.Markdown("""## Debug Info""")
        with gr.Row():
            input_tokens = gr.Textbox(
                label=f"Input Tokens",
                interactive=False,
            )

            completion_tokens = gr.Textbox(
                label=f"Completion Tokens",
                interactive=False,
            )

        detailed_error = gr.Textbox(
            label=f"Detailed Error",
            interactive=False,
        )

        history_state = gr.State([])

        respond = submit_btn.click(
            inference_fn,
            inputs=[
                temperature,
                top_p,
                max_new_token,
                input_mode,
                audio,
                text_input,
                history_state,
                input_tokens,
                completion_tokens,
            ],
            outputs=[
                history_state,
                input_tokens,
                completion_tokens,
                detailed_error,
                output_audio,
                complete_audio,
            ],
        )

        respond.then(lambda s: s, [history_state], chatbot)

        reset_btn.click(
            clear_fn,
            outputs=[
                chatbot,
                history_state,
                input_tokens,
                completion_tokens,
                detailed_error,
                output_audio,
                complete_audio,
            ],
        )
        input_mode.input(
            clear_fn,
            outputs=[
                chatbot,
                history_state,
                input_tokens,
                completion_tokens,
                detailed_error,
                output_audio,
                complete_audio,
            ],
        ).then(update_input_interface, inputs=[input_mode], outputs=[audio, text_input])

    initialize_fn()
    # Launch the interface
    demo.launch(server_port=args.port, server_name=args.host)
