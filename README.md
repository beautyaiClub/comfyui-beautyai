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

## 安装

1. 将此文件夹放入 `ComfyUI/custom_nodes/` 目录
2. 安装依赖（仅音频节点需要）：
   ```bash
   pip install av
   ```
3. 重启 ComfyUI

## 依赖

**基础节点（Base64、NanoBanana、LoadImageFromURL）：**
- torch
- numpy
- PIL (Pillow)
- requests

**音频节点（LoadAudioFromURL）：**
- av (PyAV) - 需要手动安装

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

### 示例 3：代理配置（可选）

如果需要使用代理下载：

```bash
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
python main.py
```

## 注意事项

1. **音频节点**需要安装 PyAV：`pip install av`
2. URL 加载节点会将文件缓存到 `input/` 目录
3. 文件名使用时间戳，避免冲突
4. 支持通过环境变量配置代理

## 版本

- v1.0.0 - 初始版本（Base64 节点）
- v2.0.0 - 添加 NanoBanana API 节点
- v3.0.0 - 添加 URL 加载节点（图像和音频）
