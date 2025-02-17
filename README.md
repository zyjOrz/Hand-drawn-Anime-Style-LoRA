# Hand-drawn-Anime-Style-LoRA

![example.jpg](https://s2.loli.net/2025/02/14/Gc2Yevg37z5jPFm.jpg)

本项目是一个中文项目，由于作者编写相关文档经验不足，如有格式上的指导或建议欢迎在[Issues](https://github.com/zyjOrz/Hand-drawn-Anime-Style-LoRA/issues)中指出


---

## 目录

欢迎访问我的项目！本项目包括几个部分：
- [简介](#简介)
- [介绍](#介绍)
- [特性](#特性)
- [使用指南](#使用指南)

## 简介
Hand-drawn-Anime-Style-LoRA是一个基于SDXL的拥有生成特定手绘漫画画风的LoRA模型。

---

## 介绍
这个项目使用 **LoRA** 技术在 **SDXL** 的基础上进行微调，使其能够生成带有手绘漫画风格的图像。

模型利用了 LoRA（Low-Rank Adaptation）来进行微调，保留了原始模型的多种功能，并且能够生成更加精细的手绘风格。

同时使用gradio库编写webui, 成功将具有生成特定手绘动漫画风能力的AI上线网站

---

## 特性
- 基于 **SDXL**（Stability Diffusion XL）模型。
- 支持生成多种风格的手绘漫画图像。
- 可以根据输入的文本描述生成相应风格的图像。
- 采用 **LoRA** 技术进行优化，提升风格化效果。

---

## 使用指南

克隆本项目：
   ```bash
   git clone https://github.com/zyjOrz/Hand-drawn-Anime-Style-LoRA.git
   cd Hand-drawn-Anime-Style-LoRA
```

`train_text_to_image_lora_sdxl.py`用于训练得到lora

`simage.py`用于生图

`gradio_for_it.py`用于生成网站
