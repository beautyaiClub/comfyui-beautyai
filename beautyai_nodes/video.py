"""
视频合成节点
"""

import os
import shutil
import subprocess
import tempfile
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from scipy.io import wavfile
import folder_paths


class BeautyAI_VideoCombine:
    """
    视频合成节点
    将图像序列和音频合成为 MP4 视频文件
    保存到 output 目录，可通过 RunPod URL 访问
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 60,
                    "step": 1
                }),
                "filename_prefix": ("STRING", {
                    "default": "video_output",
                    "multiline": False
                }),
                "crf": ("INT", {
                    "default": 19,
                    "min": 0,
                    "max": 51,
                    "step": 1
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "combine_video"
    OUTPUT_NODE = True
    CATEGORY = "BeautyAI"

    def combine_video(self, images, frame_rate, filename_prefix, crf, audio=None):
        """
        合成视频

        Args:
            images: IMAGE tensor [batch, height, width, channels]
            frame_rate: 帧率
            filename_prefix: 文件名前缀
            crf: 视频质量 (0-51, 越小质量越好)
            audio: AUDIO dict (可选)

        Returns:
            tuple: (filename,)
        """
        temp_dir = None
        audio_file = None

        try:
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{filename_prefix}_{timestamp}.mp4"
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, output_filename)

            print(f"BeautyAI VideoCombine: 开始合成视频")
            print(f"BeautyAI VideoCombine: 图像数量: {len(images)}, 帧率: {frame_rate}")
            print(f"BeautyAI VideoCombine: 输出路径: {output_path}")

            # 创建临时目录保存帧
            temp_dir = tempfile.mkdtemp(prefix="beautyai_video_")
            print(f"BeautyAI VideoCombine: 临时目录: {temp_dir}")

            # 保存所有帧为图片（使用 JPEG 更快）
            for i, image_tensor in enumerate(images):
                # 转换为 numpy array [H, W, C]
                image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
                # 创建 PIL Image
                image = Image.fromarray(image_np)
                # 保存为 JPEG（比 PNG 快很多）
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
                image.save(frame_path, format="JPEG", quality=95)

            print(f"BeautyAI VideoCombine: 已保存 {len(images)} 帧")

            # 处理音频（如果有）
            if audio is not None:
                try:
                    waveform = audio["waveform"]
                    sample_rate = audio["sample_rate"]

                    # 移除 batch 维度 [batch, channels, samples] -> [channels, samples]
                    if waveform.dim() == 3:
                        waveform = waveform.squeeze(0)

                    print(f"BeautyAI VideoCombine: 音频形状: {waveform.shape}, 采样率: {sample_rate}")

                    # 保存音频为临时 WAV 文件
                    audio_file = os.path.join(temp_dir, "audio.wav")

                    # 转换为 int16 [samples, channels]
                    audio_data = (waveform.t() * 32767).clamp(-32768, 32767).to(torch.int16).cpu().numpy()

                    # 保存 WAV 文件
                    wavfile.write(audio_file, sample_rate, audio_data)

                    print(f"BeautyAI VideoCombine: 音频已保存到: {audio_file}")

                except Exception as e:
                    print(f"BeautyAI VideoCombine: 音频处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    audio_file = None

            # 构建 ffmpeg 命令
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                raise RuntimeError("未找到 ffmpeg")

            # 构建命令：先添加所有输入，再添加编码参数
            cmd = [
                ffmpeg_path,
                "-y",  # 覆盖输出文件
                "-framerate", str(frame_rate),
                "-i", os.path.join(temp_dir, "frame_%06d.jpg"),
            ]

            # 如果有音频，添加音频输入（必须在编码参数之前）
            if audio_file and os.path.exists(audio_file):
                cmd.extend(["-i", audio_file])

            # 添加视频编码参数
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "faster",  # 使用更快的编码预设
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
            ])

            # 如果有音频，添加音频编码参数
            if audio_file and os.path.exists(audio_file):
                cmd.extend([
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",  # 以最短的流为准
                ])

            # 输出文件
            cmd.append(output_path)

            print(f"BeautyAI VideoCombine: 执行 ffmpeg 命令")

            # 执行 ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip() or "未知错误"
                raise RuntimeError(f"ffmpeg 失败: {error_msg}")

            # 检查输出文件
            if not os.path.exists(output_path):
                raise RuntimeError("输出文件未生成")

            file_size = os.path.getsize(output_path)
            print(f"BeautyAI VideoCombine: 视频合成成功")
            print(f"BeautyAI VideoCombine: 文件大小: {file_size} bytes")
            print(f"BeautyAI VideoCombine: 文件名: {output_filename}")

            return (output_filename,)

        except Exception as e:
            error_msg = f"视频合成失败: {str(e)}"
            print(f"BeautyAI VideoCombine: {error_msg}")
            import traceback
            traceback.print_exc()
            return ("",)

        finally:
            # 清理临时文件
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"BeautyAI VideoCombine: 已清理临时目录")
                except Exception as e:
                    print(f"BeautyAI VideoCombine: 清理临时目录失败: {e}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "BeautyAI_VideoCombine": BeautyAI_VideoCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BeautyAI_VideoCombine": "视频合成",
}
