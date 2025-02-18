import gradio as gr 
from diffusers import DiffusionPipeline, AutoencoderKL, AutoPipelineForText2Image, LCMScheduler, DDPMScheduler
from diffusers.utils import load_image
import torch
from PIL import Image
import io

# 加载 VAE
vae = AutoencoderKL.from_pretrained("./VAE", torch_dtype=torch.float16, use_safetensors=False)

# 定义缓存路径
cache_dir = "/root/autodl-tmp/SDXL"

# 加载 Stable Diffusion 模型
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

# 加载 IP-Adapter 权重
ip_adapter_path = "./ipadapter"  # 根据需要调整 IP-Adapter 权重路径
pipeline = AutoPipelineForText2Image.from_pretrained(
    "/root/autodl-tmp/SDXL/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", 
    torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter(ip_adapter_path, subfolder="sdxl", weight_name="ip-adapter_sdxl.safetensors", local_files_only=True)
pipeline.load_lora_weights("lora_models", weight_name="Hand-drawn Anime Style for LoRA.safetensors", adapter_name="draw2")
pipeline.set_ip_adapter_scale(0.6)

# 定义生成图像的函数
def generate_image(prompt, negative_prompt, lora_scale=0.9, num_inference_steps=30, load_lcm=False, load_ip_adapter=False, ip_adapter_image=None):
    # 根据开关加载 lcm 权重
    if load_lcm:
        pipe.set_adapters(["draw", "lcm"], adapter_weights=[1.0, 1.0])
        cfg = 1.8
        num_inference_steps = 7
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.set_adapters(["draw", "lcm"], adapter_weights=[1.0, 0.0])
        cfg = 6.0
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    # 如果启用了 IP-Adapter，加载 LoRA 权重并生成图像
    if load_ip_adapter:
        # 加载 LoRA 权重（如果启用 IP-Adapter）
        #pipeline.load_lora_weights("lora_models", weight_name="Hand-drawn Anime Style for LoRA.safetensors", adapter_name="draw2")
        
        # 确保只有在提供 IP-Adapter 图像时才进行图像生成
        if ip_adapter_image is not None:
            # 使用上传的文件路径读取图像内容
            with open(ip_adapter_image.name, "rb") as f:
                image = Image.open(f)  # 读取文件并加载为图像
                image = image.convert("RGB")  # 转换为 RGB 模式，以确保正确处理

            # 使用 `pipeline` 进行生成，并传递 IP-Adapter 图像和 LoRA 权重
            generator = torch.Generator(device="cpu").manual_seed(0)
            images = pipeline(
                prompt=prompt,
                ip_adapter_image=image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                cross_attention_kwargs={"scale": lora_scale},
                guidance_scale=cfg,
            ).images
        else:
            # 如果没有上传 IP-Adapter 图像，返回错误或默认行为
            return "Error: IP-Adapter image is required when IP-Adapter is enabled."
    else:
        # 如果没有启用 IP-Adapter，使用 LoRA 权重生成图像
        images = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            cross_attention_kwargs={"scale": lora_scale},
            guidance_scale=cfg,
        ).images

    return images[0]



# Gradio 界面
prompt = gr.Textbox(label="Prompt", value="an anime-style girl.")
negative_prompt = gr.Textbox(label="Negative Prompt", value="low quality, bad quality")
lora_scale = gr.Slider(minimum=0, maximum=2, step=0.1, label="LoRA Scale", value=1)
num_steps = gr.Slider(minimum=0, maximum=100, step=1, label="num_inference_steps", value=30)
load_lcm = gr.Checkbox(label="Load LCM", value=False)
load_ip_adapter = gr.Checkbox(label="Load IP Adapter", value=False)
ip_adapter_image = gr.File(label="Upload IP Adapter Image")  # 添加上传图片的功能

# 创建 Gradio 接口
iface = gr.Interface(
    fn=generate_image,
    inputs=[prompt, negative_prompt, lora_scale, num_steps, load_lcm, load_ip_adapter, ip_adapter_image],
    outputs=gr.Image(label="Generated Image"),
    title="Hand-drawn Anime Style LoRA with IP-Adapter",
    description="使用 LoRA 和可选的 IP-Adapter 生成动漫风格的图像，IP-Adapter 提供增强效果。上传图片作为 IP-Adapter 输入，LoRA 权重会被同时加载。"
)

iface.launch(server_name="0.0.0.0", server_port=7860)