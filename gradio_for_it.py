import gradio as gr 
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers import LCMScheduler, DDPMScheduler
import torch

# 加载模型
vae = AutoencoderKL.from_pretrained("./VAE", torch_dtype=torch.float16, use_safetensors=False)

cache_dir = "/root/autodl-tmp/SDXL"

pipe = DiffusionPipeline.from_pretrained(
    "/root/autodl-tmp/SDXL/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
    vae=vae,
    torch_dtype=torch.float16,
    cache_dir=cache_dir
)

pipe.enable_model_cpu_offload()

# 加载 LoRA 权重（手绘动漫风格）
pipe.load_lora_weights("lora_models", weight_name="Hand-drawn Anime Style for LoRA.safetensors", adapter_name="draw")
pipe.load_lora_weights("lora_models", weight_name="lcm.safetensors", adapter_name="lcm")
# 定义生成图像的函数


def generate_image(prompt, negative_prompt, lora_scale=0.9, num_inference_steps=30, load_lcm=False):
    # 根据开关加载 lcm 权重
    if load_lcm:
        pipe.set_adapters(["draw", "lcm"], adapter_weights=[1.0, 1.0])
        cfg=1.8
        num_inference_steps=7
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    # 生成图像
    else :
        pipe.set_adapters(["draw", "lcm"], adapter_weights=[1.0, 0.0])
        cfg=6.0
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        cross_attention_kwargs={"scale": lora_scale},
        guidance_scale=cfg,
    ).images
    return images[0]

prompt = gr.Textbox(label="Prompt", value="an anime-style girl.")
negative_prompt = gr.Textbox(label="Negative Prompt", value="low quality, bad quality")
lora_scale = gr.Slider(minimum=0, maximum=2, step=0.1, label="LoRA Scale", value=1)
num_steps = gr.Slider(minimum=0, maximum=100, step=1, label="num_inference_steps", value=30)
load_lcm = gr.Checkbox(label="Load LCM", value=False)
# 创建 Gradio 接口
iface = gr.Interface(
    fn=generate_image,
    inputs=[prompt, negative_prompt, lora_scale, num_steps, load_lcm],
    outputs=gr.Image(label="Generated Image"),
    title="Hand-drawn-Anime-Style-LoRA",
    description="Hand-drawn Anime Style LoRA is a LoRA model designed to generate images in a specific hand-drawn anime style."
)
iface.launch(server_name="0.0.0.0", server_port=7860)
