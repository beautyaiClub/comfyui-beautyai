"""
辅助函数模块
包含音频处理、文件下载等通用功能
"""

import os
import shutil
import subprocess
from datetime import datetime
import torch
import av


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def load_audio_with_pyav(filepath: str) -> tuple:
    """使用 PyAV 加载音频（ComfyUI 原生方法）"""
    with av.open(filepath) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)

        if not frames:
            raise ValueError("No audio frames decoded.")

        wav = torch.cat(frames, dim=1)
        wav = f32_pcm(wav)
        return wav, sr


def download_file_from_url(url: str, save_dir: str, file_extension: str = None) -> str:
    """
    从 URL 下载文件，自动生成带时间戳的文件名

    Args:
        url: 文件 URL
        save_dir: 保存目录
        file_extension: 文件扩展名（可选，从 URL 自动提取）

    Returns:
        str: 下载的文件完整路径
    """
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒

    # 从 URL 提取扩展名
    if not file_extension:
        url_path = url.split('?')[0]  # 移除 query 参数
        ext = os.path.splitext(url_path)[1]
        if not ext:
            ext = '.mp3'  # 默认扩展名
        file_extension = ext

    filename = f"download_{timestamp}{file_extension}"
    filepath = os.path.join(save_dir, filename)

    print(f"BeautyAI Download: 开始下载: {url}")
    print(f"BeautyAI Download: 保存到: {filepath}")

    # 使用 wget 下载
    wget_path = shutil.which("wget")
    if not wget_path:
        raise RuntimeError("未找到 wget 可执行文件")

    cmd = [
        wget_path,
        "--no-verbose",
        "--timeout=60",
        "--tries=3",
        "--output-document",
        filepath,
        url,
    ]

    # 使用系统环境变量中的代理设置（如果存在）
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "未知错误"
        raise RuntimeError(f"wget 下载失败: {msg}")

    size_bytes = os.path.getsize(filepath)
    print(f"BeautyAI Download: 下载成功，大小: {size_bytes} bytes")

    return filepath
