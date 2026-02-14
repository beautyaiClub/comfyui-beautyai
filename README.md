# comfyui-beautyai

BeautyAI 自定义 ComfyUI 节点集合

## 节点列表

### 1. Base64 图片输入 (BeautyAI_Base64ImageInput)

将 Base64 编码的图片字符串转换为 ComfyUI 的 IMAGE tensor。

**输入：**
- `base64_string` (STRING): Base64 编码的图片字符串

**输出：**
- `image` (IMAGE): ComfyUI 标准图片格式

**特性：**
- 自动检测并移除 data URI 前缀
- 自动转换为 RGB 模式
- 错误处理：解码失败时返回空白图片

---

### 2. Base64 图片输出 (BeautyAI_Base64ImageOutput)

将 ComfyUI 的 IMAGE tensor 转换为 Base64 编码的字符串。

**输入：**
- `images` (IMAGE): ComfyUI 图片格式
- `format` (选择): PNG / JPEG / WEBP
- `quality` (INT): 图片质量 (1-100)

**输出：**
- `base64_string` (STRING): 带 data URI 前缀的 Base64 字符串

**特性：**
- 支持多种输出格式
- 可调节图片质量
- 自动添加 data URI 前缀

---

### 3. GRSAI NanoBanana API (GRSAI_NanoBanana)

对接 GRSAI NanoBanana API，处理图片生成请求。

**输入：**
- `api_key` (STRING): API 密钥
- `prompt` (STRING): 提示词
- `model` (选择): 模型选择
- `aspect_ratio` (选择): 宽高比
- `image_size` (选择): 图片大小
- `images` (IMAGE, 可选): 输入图片
- `image_urls` (STRING, 可选): 图片 URL 列表

**输出：**
- `images` (IMAGE): 生成的图片
- `status` (STRING): 请求状态
- `failure_reason` (STRING): 失败原因
- `error` (STRING): 错误信息

**特性：**
- 流式响应处理
- 进度条显示
- 支持多种输入方式

---

### 4. 从 URL 加载图像 (BeautyAI_LoadImageFromURL)

从 URL 下载并加载图像，自动生成带时间戳的文件名。

**输入：**
- `url` (STRING): 图像 URL

**输出：**
- `image` (IMAGE): ComfyUI 图像 tensor

**特性：**
- 自动生成时间戳文件名（`download_20240213_143025_123.png`）
- 保存到 `ComfyUI/input/` 目录（自动缓存）
- 支持系统代理（通过环境变量）
- 直接返回 IMAGE tensor，无需额外节点

---

### 5. 从 URL 加载音频 (BeautyAI_LoadAudioFromURL)

从 URL 下载并加载音频，使用 PyAV（ComfyUI 原生方法）。

**输入：**
- `url` (STRING): 音频 URL

**输出：**
- `audio` (AUDIO): ComfyUI 音频格式

**特性：**
- 自动生成时间戳文件名（`download_20240213_143025_456.mp3`）
- 保存到 `ComfyUI/input/` 目录（自动缓存）
- 使用 PyAV 加载（与 ComfyUI 原生 LoadAudio 相同）
- 支持所有 PyAV 支持的音频格式
- 支持系统代理（通过环境变量）

---

### 6. 视频合成 (BeautyAI_VideoCombine)

将图像序列和音频合成为 MP4 视频文件，保存到 output 目录。

**输入：**
- `images` (IMAGE): 图像序列
- `frame_rate` (INT): 帧率 (1-60, 默认 25)
- `filename_prefix` (STRING): 文件名前缀（默认 "video_output"）
- `crf` (INT): 视频质量 (0-51, 默认 19，越小质量越好)
- `audio` (AUDIO, 可选): 音频轨道

**输出：**
- `filename` (STRING): 生成的视频文件名

**特性：**
- 使用 ffmpeg 进行视频编码（H.264）
- 支持可选音频轨道（AAC 编码，192kbps）
- 自动生成带时间戳的文件名（`video_output_20240213_143025.mp4`）
- 保存到 `ComfyUI/output/` 目录（**RunPod 可通过 URL 访问**）
- 自动清理临时文件
- 如果有音频，视频长度以最短的流为准

**RunPod 使用说明：**
- 视频保存在 `output/` 目录
- RunPod 会自动将 output 目录的文件通过 URL 返回
- 可以通过 API 响应中的文件名访问生成的视频

---

## 安装

1. 将此文件夹放入 `ComfyUI/custom_nodes/` 目录
2. 安装依赖：
   ```bash
   pip install av scipy
   ```
3. 确保系统已安装 ffmpeg（视频合成节点需要）：
   ```bash
   # Ubuntu/Debian
   apt-get install ffmpeg

   # macOS
   brew install ffmpeg
   ```
4. 重启 ComfyUI

## 依赖

**基础节点（Base64、NanoBanana、LoadImageFromURL）：**
- torch
- numpy
- PIL (Pillow)
- requests

**音频和视频节点（LoadAudioFromURL、VideoCombine）：**
- av (PyAV) - 音频加载
- scipy - 音频保存
- ffmpeg - 系统依赖，视频编码

## 使用示例

### 示例 1：Base64 输入输出

```
[Base64 图片输入] → [图片处理] → [Base64 图片输出]
```

### 示例 2：URL 加载工作流

```
[从 URL 加载图像] → [ImageScaleByAspectRatio V2] → [InfiniteTalk]
                                                        ↑
[从 URL 加载音频] → [音频处理节点] ────────────────────┘
```

### 示例 3：完整视频生成工作流（替代 VHS_VideoCombine）

```
[从 URL 加载图像] → [图像处理] → [视频生成模型] → [视频合成] → 输出 MP4
                                                    ↑
[从 URL 加载音频] → [音频处理] ──────────────────────┘
```

**说明：**
- `视频合成` 节点接收图像序列和音频
- 自动合成为 MP4 视频并保存到 output 目录
- RunPod 会自动通过 URL 返回生成的视频文件
- 可以完全替代 `VHS_VideoCombine` 节点

### 示例 4：代理配置（可选）

如果需要使用代理下载：

```bash
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
python main.py
```

## 注意事项

1. **音频和视频节点**需要安装 PyAV：`pip install av`
2. **视频合成节点**需要系统安装 ffmpeg
3. URL 加载节点会将文件缓存到 `input/` 目录
4. 视频文件保存到 `output/` 目录，RunPod 可通过 URL 访问
5. 文件名使用时间戳，避免冲突
6. 支持通过环境变量配置代理

## RunPod 部署说明

在 RunPod 上使用时：

1. **Dockerfile 中确保安装依赖：**
   ```dockerfile
   RUN apt-get install -y ffmpeg
   RUN pip install av
   ```

2. **视频输出自动可访问：**
   - 视频保存在 `ComfyUI/output/` 目录
   - RunPod worker 会自动将 output 文件通过 URL 返回
   - 无需额外配置

3. **替代 VideoHelperSuite：**
   - 如果 `VHS_VideoCombine` 节点安装失败
   - 可以直接使用 `BeautyAI_VideoCombine` 替代
   - 功能相同，更简单可靠

## 版本

- v1.0.0 - 初始版本（Base64 节点）
- v2.0.0 - 添加 NanoBanana API 节点
- v3.0.0 - 添加 URL 加载节点（图像和音频）
- v4.0.0 - 添加视频合成节点（可替代 VHS_VideoCombine）
