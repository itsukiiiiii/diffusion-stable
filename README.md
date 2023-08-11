# Stable_Diffusion
## 前言
- 支持单卡推理
- 支持 text2img、img2img、image inpainting 功能

## 版本信息
- StableDiffusion 官方代码：https://github.com/Stability-AI/stablediffusion.git (commit ID: cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf)
- 框架: PyTorch

## 使用 docker 环境
如果提供的有现成的 docker 镜像，可以跳过下面的 “环境配置” 部分。准备好数据集和模型后，可以直接加载使用。
- 加载 docker
```bash
docker load -i stable_diffusion_ubuntu18.04_py37_cntoolkit3.5.2.tar.gz
```
- 启动 docker
```bash
bash run_stable_diffusion_docker.sh
```

## 环境配置 (docker 内已经配置好)
```bash
pip3 install -r requirements.txt
```

## 准备模型
- Text2img、Img2img (./models/v2-1_768-nonema-pruned.ckpt) 
- OpenCLIP transformer encoder for text （CLIP-ViT-H-14-laion2B-s32B-b79K）
- Image Inpainting (./models/512-inpainting-ema.ckpt)
```bash
wget -P ./models https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-nonema-pruned.ckpt
wget -P ./models https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt
# OpenCLIP （CLIP-ViT-H-14-laion2B-s32B-b79K）
wget -p ./models https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
```
提示：可以使用提前下载好的 OpenCLIP 模型，避免在程序运行过程中重复下载。下载后的 OpenCLIP 模型可以在以下代码中进行修改并使用。  
```bash
# ./ldm/modules/encoders/modules.py +191
# 模型路径根据实际位置进行设置，以下路径仅为举例
pretrained_path = "/workspace/models/open_clip_pytorch_model.bin"
model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained_path)
```

## 推理
- Text2Img
```bash
# txt2img测试
python3 scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt ./models/v2-1_768-nonema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --device mlu
```
![推理结果1](./sample_imgs/text2img.jpg)

- Img2Img
```bash
# --init-img 的图片可以用户自己选择
python3 scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img ./outputs/txt2img-samples/samples/00003.png --strength 0.8 --ckpt ./models/v2-1_768-nonema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml
```
![推理结果2](./sample_imgs/img2img.jpg)

- Image Inpainting

```bash
# Image Inpainting with Stable Diffusion 测试 with gradio
# gradio 页面的 Adavanced Options 中的 Images 建议设为 1
python3 scripts/gradio/inpainting.py configs/stable-diffusion/v2-inpainting-inference.yaml ./models/512-inpainting-ema.ckpt
```
提示：
- gradio 库提供两种访问方式，在代码的 “launch(share=True)”部分，使能 share 即可开启 public url。local url（只可本地访问，速率快，使用体验好），public url（该链接可以提供给其他互联网用户使用，但是因网络原因，响应时间长，体验一般）。

程序运行成功后，通过点击或者在浏览器输入 public url 或者 local url 后，即可开始多轮对话。  
![推理结果3](./sample_imgs/image_inpainting.jpg)