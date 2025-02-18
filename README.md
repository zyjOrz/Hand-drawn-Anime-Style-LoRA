# Hand-drawn-Anime-Style-LoRA-Expand

![example.jpg](https://s2.loli.net/2025/02/14/Gc2Yevg37z5jPFm.jpg)

---

## 目录

欢迎访问我的项目！本项目包括几个部分：
- [简介](#简介)
- [介绍](#介绍)
- [特性](#特性)
- [使用指南](#使用指南)

## 简介
Hand-drawn-Anime-Style-LoRA-Expand是一个基于SDXL的拥有生成特定手绘漫画画风的LoRA模型及其拓展。

---

## 介绍
这个项目使用 **LoRA** 方法在 **SDXL** 的基础上进行微调，使其能够生成带有手绘漫画风格的图像。

同时搭载了LCM_LoRA，可以加速推理，七步推理内生成图片。

搭载了IP_adapter，可以根据用户上传的图片进行手绘画风的风格迁移。

使用gradio库编写webui, 将具有生成特定手绘动漫画风能力的AI上线网站

---

## 使用指南

克隆本项目：
   ```bash
   git clone https://github.com/zyjOrz/Hand-drawn-Anime-Style-LoRA.git
   cd Hand-drawn-Anime-Style-LoRA
```
`dataset` 是用于训练lora的训练集

`train_text_to_image_lora_sdxl.py` 是使用的训练脚本

`gradio_for_it.py` ai应用后端，配备IP_adapter和LCM_lora的功能。

![1ac1acf2df7adb162cbd3ce7ac8e4bec.png](https://s2.loli.net/2025/02/18/2lxh79jFtPSkDdN.png)
