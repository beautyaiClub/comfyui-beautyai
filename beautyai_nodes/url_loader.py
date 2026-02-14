"""
URL 加载节点（图像和音频）
"""

import numpy as np
import torch
from PIL import Image
import folder_paths
from ..utils import download_file_from_url, load_audio_with_pyav


class BeautyAI_LoadImageFromURL:
    """
    从 URL 下载并加载图像
    自动生成带时间戳的文件名，直接返回 IMAGE tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "BeautyAI"

    def load_image(self, url):
        """从 URL 下载并加载图像"""
        filepath = None
        try:
            if not url.strip():
                raise ValueError("URL 不能为空")

            # 下载到 input 目录
            input_dir = folder_paths.get_input_directory()
            filepath = download_file_from_url(url, input_dir, file_extension='.png')

            # 加载图像
            image = Image.open(filepath)

            # 转换为 RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 转换为 numpy array
            image_np = np.array(image).astype(np.float32) / 255.0

            # 转换为 torch tensor [batch, height, width, channels]
            image_tensor = torch.from_numpy(image_np)[None,]

            print(f"BeautyAI LoadImageFromURL: 加载完成，形状: {image_tensor.shape}")

            return (image_tensor,)

        except Exception as e:
            print(f"BeautyAI LoadImageFromURL 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回空白图像
            error_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (error_image,)
        finally:
            # 可选：清理下载的文件（如果不需要缓存）
            # if filepath and os.path.exists(filepath):
            #     os.unlink(filepath)
            pass


class BeautyAI_LoadAudioFromURL:
    """
    从 URL 下载并加载音频
    自动生成带时间戳的文件名，直接返回 AUDIO dict
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"
    CATEGORY = "BeautyAI"

    def load_audio(self, url):
        """从 URL 下载并加载音频"""
        filepath = None
        try:
            if not url.strip():
                raise ValueError("URL 不能为空")

            # 下载到 input 目录
            input_dir = folder_paths.get_input_directory()
            filepath = download_file_from_url(url, input_dir, file_extension='.mp3')

            # 使用 PyAV 加载音频（ComfyUI 原生方法）
            waveform, sample_rate = load_audio_with_pyav(filepath)

            # 返回 ComfyUI AUDIO 格式
            audio_dict = {
                "waveform": waveform.unsqueeze(0),  # 添加 batch 维度
                "sample_rate": sample_rate
            }

            print(f"BeautyAI LoadAudioFromURL: 加载完成，采样率: {sample_rate}, 形状: {waveform.shape}")

            return (audio_dict,)

        except Exception as e:
            print(f"BeautyAI LoadAudioFromURL 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回空音频
            empty_audio = {
                "waveform": torch.zeros((1, 1, 1000)),
                "sample_rate": 44100
            }
            return (empty_audio,)
        finally:
            # 可选：清理下载的文件（如果不需要缓存）
            # if filepath and os.path.exists(filepath):
            #     os.unlink(filepath)
            pass


# 节点映射
NODE_CLASS_MAPPINGS = {
    "BeautyAI_LoadImageFromURL": BeautyAI_LoadImageFromURL,
    "BeautyAI_LoadAudioFromURL": BeautyAI_LoadAudioFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BeautyAI_LoadImageFromURL": "从 URL 加载图像",
    "BeautyAI_LoadAudioFromURL": "从 URL 加载音频",
}
