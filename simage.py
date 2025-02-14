from diffusers import ControlNetModel, DiffusionPipeline, AutoencoderKL

from diffusers.utils import load_image

from PIL import Image

import peft

import torch

import numpy as np

import cv2

  

prompt = "a cute girl."

negative_prompt = 'low quality, bad quality'


vae = AutoencoderKL.from_pretrained("./VAE", torch_dtype=torch.float16, use_safetensors=False)

cache_dir = "/root/autodl-tmp/SDXL"

pipe = DiffusionPipeline.from_pretrained(

    "/root/autodl-tmp/SDXL/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",

    vae=vae,

    torch_dtype=torch.float16,

    cache_dir=cache_dir

  )

pipe.enable_model_cpu_offload()



pipe.load_lora_weights("lora_models", weight_name="Hand-drawn Anime Style for LoRA.safetensors", adapter_name="draw")
  
  

#images = pipe(

  #prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,

 #).images


lora_scale = 0.9
images = pipe(
    prompt, negative_prompt=negative_prompt,  num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images  

images[0].save(f"1.png")