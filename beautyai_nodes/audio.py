"""
音频处理节点
"""

import math


class BeautyAI_AudioFrameCalculator:
    """
    音频帧数计算器
    根据音频时长和FPS自动计算需要的视频帧数
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.01
                }),
                "buffer_frames": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "round_up": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("num_frames", "audio_duration")
    FUNCTION = "calculate"
    CATEGORY = "BeautyAI/audio"

    def calculate(self, audio, fps, buffer_frames, round_up):
        """
        计算音频对应的视频帧数

        Args:
            audio: 音频数据 (包含waveform和sample_rate)
            fps: 视频帧率
            buffer_frames: 额外缓冲帧数
            round_up: 是否向上取整

        Returns:
            (num_frames, audio_duration): 帧数和音频时长(秒)
        """
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # 计算音频时长(秒)
        audio_length_samples = waveform.shape[-1]
        audio_duration = audio_length_samples / sample_rate

        # 计算需要的帧数
        exact_frames = audio_duration * fps

        if round_up:
            # 向上取整,确保帧数足够
            num_frames = math.ceil(exact_frames) + buffer_frames
        else:
            # 直接取整
            num_frames = int(exact_frames) + buffer_frames

        print(f"[BeautyAI_AudioFrameCalculator] Audio duration: {audio_duration:.2f}s, Sample rate: {sample_rate}Hz")
        print(f"[BeautyAI_AudioFrameCalculator] Exact frames needed: {exact_frames:.2f}")
        print(f"[BeautyAI_AudioFrameCalculator] Final frames: {num_frames} (fps={fps}, buffer={buffer_frames}, round_up={round_up})")

        return (num_frames, audio_duration)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "BeautyAI_AudioFrameCalculator": BeautyAI_AudioFrameCalculator,
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "BeautyAI_AudioFrameCalculator": "Audio Frame Calculator",
}
