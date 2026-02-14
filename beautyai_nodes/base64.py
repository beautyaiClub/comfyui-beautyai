"""
Base64 图片输入输出节点
"""

import base64
import io
import numpy as np
import torch
from PIL import Image


class BeautyAI_Base64ImageInput:
    """
    Base64 图片输入节点
    将 Base64 编码的图片字符串转换为 ComfyUI 的 IMAGE tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_string": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode_base64"
    CATEGORY = "BeautyAI"

    def decode_base64(self, base64_string):
        """
        将 Base64 字符串解码为图片 tensor

        Args:
            base64_string: Base64 编码的图片字符串

        Returns:
            tuple: (IMAGE tensor,)
        """
        try:
            # 移除可能存在的 data URI 前缀
            if "," in base64_string:
                base64_string = base64_string.split(",", 1)[1]

            # 解码 Base64 字符串
            image_data = base64.b64decode(base64_string.strip())

            # 从字节流创建 PIL Image
            image = Image.open(io.BytesIO(image_data))

            # 转换为 RGB 模式（如果不是的话）
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 转换为 numpy array
            image_np = np.array(image).astype(np.float32) / 255.0

            # 转换为 torch tensor，并添加 batch 维度
            # ComfyUI 的图片格式是 [batch, height, width, channels]
            image_tensor = torch.from_numpy(image_np)[None,]

            return (image_tensor,)

        except Exception as e:
            print(f"Base64ImageInput 错误: {str(e)}")
            # 返回一个小的空白图片作为错误处理
            error_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (error_image,)


class BeautyAI_Base64ImageOutput:
    """
    Base64 图片输出节点
    将 ComfyUI 的 IMAGE tensor 转换为 Base64 编码的字符串
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "format": (["PNG", "JPEG", "WEBP"], {
                    "default": "PNG"
                }),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base64_string",)
    FUNCTION = "encode_base64"
    OUTPUT_NODE = True
    CATEGORY = "BeautyAI"

    def encode_base64(self, images, format="PNG", quality=95):
        """
        将图片 tensor 编码为 Base64 字符串

        Args:
            images: IMAGE tensor (batch, height, width, channels)
            format: 输出图片格式 (PNG, JPEG, WEBP)
            quality: 图片质量 (1-100)，仅对 JPEG 和 WEBP 有效

        Returns:
            dict: {"ui": {"base64": [...]}, "result": (base64_string,)}
        """
        try:
            # 获取第一张图片（如果是 batch）
            image_tensor = images[0]

            # 转换为 numpy array，并将值从 [0, 1] 转换到 [0, 255]
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

            # 创建 PIL Image
            image = Image.fromarray(image_np)

            # 保存到字节流
            buffered = io.BytesIO()

            # 根据格式保存
            if format == "JPEG":
                image.save(buffered, format="JPEG", quality=quality)
            elif format == "WEBP":
                image.save(buffered, format="WEBP", quality=quality)
            else:  # PNG
                image.save(buffered, format="PNG")

            # 获取字节数据
            img_bytes = buffered.getvalue()

            # 编码为 Base64
            base64_string = base64.b64encode(img_bytes).decode('utf-8')

            # 添加 data URI 前缀
            mime_type = f"image/{format.lower()}"
            base64_with_prefix = f"data:{mime_type};base64,{base64_string}"

            # 同时返回 UI 数据和普通输出
            # UI 数据会出现在 History API 的 outputs 中
            return {
                "ui": {"base64": [base64_with_prefix]},
                "result": (base64_with_prefix,)
            }

        except Exception as e:
            print(f"Base64ImageOutput 错误: {str(e)}")
            return {
                "ui": {"base64": [""]},
                "result": ("",)
            }


# 节点映射
NODE_CLASS_MAPPINGS = {
    "BeautyAI_Base64ImageInput": BeautyAI_Base64ImageInput,
    "BeautyAI_Base64ImageOutput": BeautyAI_Base64ImageOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BeautyAI_Base64ImageInput": "Base64 图片输入",
    "BeautyAI_Base64ImageOutput": "Base64 图片输出",
}
