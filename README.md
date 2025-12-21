# comfyui-beautyai

BeautyAI 自定义 Custom Nodes

## 节点列表

### 1. Base64 图片输入 (BeautyAI_Base64ImageInput)

将 Base64 编码的图片字符串转换为 ComfyUI 的 IMAGE tensor。

**输入参数：**
- `base64_string` (STRING): Base64 编码的图片字符串
  - 支持带 data URI 前缀的格式：`data:image/png;base64,iVBORw0KG...`
  - 也支持纯 Base64 字符串

**输出：**
- `image` (IMAGE): ComfyUI 标准图片格式

**特性：**
- 自动检测并移除 data URI 前缀
- 自动转换为 RGB 模式
- 错误处理：如果解码失败，返回空白图片

### 2. Base64 图片输出 (BeautyAI_Base64ImageOutput)

将 ComfyUI 的 IMAGE tensor 转换为 Base64 编码的字符串。

**输入参数：**
- `images` (IMAGE): ComfyUI 图片格式
- `format` (选择): 输出图片格式
  - PNG (默认)
  - JPEG
  - WEBP
- `quality` (INT): 图片质量 (1-100)
  - 默认：95
  - 仅对 JPEG 和 WEBP 格式有效

**输出：**
- `base64_string` (STRING): 带 data URI 前缀的 Base64 字符串
  - 格式：`data:image/[format];base64,[base64_data]`

**特性：**
- 支持多种输出格式
- 可调节图片质量
- 自动添加 data URI 前缀，方便在网页中直接使用

## 安装

1. 将此文件夹放入 `ComfyUI/custom_nodes/` 目录
2. 重启 ComfyUI

## 使用示例

### 示例 1：Base64 输入 → 处理 → Base64 输出

```
BeautyAI_Base64ImageInput → [图片处理节点] → BeautyAI_Base64ImageOutput
```

### 示例 2：API 集成

这些节点特别适合通过 API 使用 ComfyUI：
- 将图片以 Base64 格式发送到 ComfyUI
- 处理后以 Base64 格式返回结果
- 无需处理文件上传和下载

## 依赖

这些节点使用 ComfyUI 已包含的标准库：
- torch
- numpy
- PIL (Pillow)

不需要额外安装依赖。