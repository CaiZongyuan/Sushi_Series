import sys
import os
sys.path.append('/root/LLMs/GLM-4-Voice')
sys.path.append('/root/LLMs/GLM-4-Voice/cosyvoice')
sys.path.append('/root/LLMs/CosyVoice/third_party/Matcha-TTS')

import torch
from flow_inference import AudioDecoder
import os.path

import onnx
import onnxruntime

flow_path="/root/LLMs/GLM-4-Voice/THUDM/glm-4-voice-decoder"
tokenizer_path="/root/LLMs/GLM-4-Voice/THUDM/glm-4-voice-tokenizer"

flow_config = os.path.join(flow_path, "config.yaml")
flow_checkpoint = os.path.join(flow_path, "flow.pt")
hift_checkpoint = os.path.join(flow_path, "hift.pt")
device = "cuda"
audio_decoder: AudioDecoder = None
whisper_model, feature_extractor = None, None


# Flow & Hift
audio_decoder = AudioDecoder(config_path=flow_config,
                             flow_ckpt_path=flow_checkpoint,
                             hift_ckpt_path=hift_checkpoint,
                             device=device,)


model = audio_decoder.flow

script = torch.jit.script(model.encoder)

# model.eval()

# # 导出onnx格式
# torch.onnx.export(model)


# onnx_model=onnx.load('model.onnx')

# onnx.checker.check_model(onnx_model)

# # 推理

# sess = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])


# def get_optimized_script(model, preserved_attrs=[]):
#     script = torch.jit.script(model)
#     if preserved_attrs != []:
#         script = torch.jit.freeze(script, preserved_attrs=preserved_attrs)
#     else:
#         script = torch.jit.freeze(script)
#     script = torch.jit.optimize_for_inference(script)
#     return script


# script = get_optimized_script(flow_encoder)
# script.save('{}/flow.encoder.fp32.zip'.format(flow_path))
# script = get_optimized_script(flow_encoder.half())
# script.save('{}/flow.encoder.fp16.zip'.format(flow_path))

