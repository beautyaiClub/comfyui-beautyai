"""
API 节点（GRSAI NanoBanana）
"""

import base64
import io
import json
import numpy as np
import requests
import torch
from PIL import Image
import comfy.utils


class GRSAI_NanoBanana:
    """
    GRSAI NanoBanana API 节点
    对接 GRSAI NanoBanana API，处理图片分析请求
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "model": (["nano-banana-fast", "nano-banana", "nano-banana-pro", "nano-banana-pro-vt"], {
                    "default": "nano-banana-fast"
                }),
                "aspect_ratio": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"], {
                    "default": "auto"
                }),
                "image_size": (["1K", "2K", "4K"], {
                    "default": "1K"
                }),
                "shut_progress": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "image_urls": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "web_hook": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "endpoint": ("STRING", {
                    "multiline": False,
                    "default": "https://api.grsai.com"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "status", "failure_reason", "error")
    FUNCTION = "process_api"
    OUTPUT_NODE = True
    CATEGORY = "BeautyAI"

    def process_api(self, api_key, prompt, model, aspect_ratio, image_size, shut_progress,
                    images=None, image_urls="", web_hook="", endpoint="https://api.grsai.com"):
        """
        调用 GRSAI NanoBanana API

        Args:
            api_key: API 密钥
            model: 模型选择
            aspect_ratio: 宽高比
            image_size: 图片大小
            shut_progress: 是否关闭进度
            prompt: 提示词（可选，支持连接文本节点）
            images: 输入图片（IMAGE tensor）
            image_urls: 图片 URL 列表（每行一个 URL）
            web_hook: Webhook 回调地址
            endpoint: API 端点

        Returns:
            tuple: (images_list, status, failure_reason, error)
        """
        try:
            # 准备请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            # 处理图片 URL 列表
            urls = []

            # 如果有图片输入，转换为 base64 并添加到 data URI
            if images is not None:
                for i in range(len(images)):
                    image_tensor = images[i]
                    # 转换为 numpy array
                    image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
                    # 创建 PIL Image
                    image = Image.fromarray(image_np)
                    # 保存到字节流
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_bytes = buffered.getvalue()
                    # 编码为 Base64
                    base64_string = base64.b64encode(img_bytes).decode('utf-8')
                    # 添加 data URI
                    data_uri = f"data:image/png;base64,{base64_string}"
                    urls.append(data_uri)

            # 如果有 URL 字符串输入，也添加进来
            if image_urls.strip():
                url_list = [url.strip() for url in image_urls.strip().split("\n") if url.strip()]
                urls.extend(url_list)

            # 准备请求体
            payload = {
                "model": model,
                "prompt": prompt,
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
                "urls": urls,
                "shutProgress": shut_progress
            }

            # 添加可选的 webhook
            if web_hook.strip():
                payload["webHook"] = web_hook.strip()

            # 构建完整的 API URL
            api_url = f"{endpoint.rstrip('/')}/v1/draw/nano-banana"

            print(f"GRSAI NanoBanana: 发送请求到 {api_url}")
            print(f"GRSAI NanoBanana: Payload keys: {list(payload.keys())}")

            # 发送 POST 请求（流式请求）
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=300  # 5分钟超时
            )

            # 检查响应状态
            response.raise_for_status()

            # 创建进度条（总进度为100）
            pbar = comfy.utils.ProgressBar(100)

            # 处理流式响应（SSE 格式）
            result_data = None
            last_progress = 0
            for line in response.iter_lines():
                if line:
                    try:
                        # 解码行数据
                        line_str = line.decode('utf-8').strip()

                        # SSE 格式：行以 "data: " 开头
                        if line_str.startswith('data: '):
                            # 去掉 "data: " 前缀
                            json_str = line_str[6:]  # 跳过 "data: "
                            # 解析 JSON
                            result_data = json.loads(json_str)
                            status = result_data.get('status', '')
                            progress = result_data.get('progress', 0)

                            # 更新进度条
                            if progress > last_progress:
                                pbar.update_absolute(progress)
                                last_progress = progress

                            print(f"GRSAI NanoBanana: 收到数据，状态: {status}, 进度: {progress}%")
                        elif line_str:
                            # 尝试直接解析 JSON（兼容非 SSE 格式）
                            result_data = json.loads(line_str)
                            status = result_data.get('status', '')
                            progress = result_data.get('progress', 0)

                            # 更新进度条
                            if progress > last_progress:
                                pbar.update_absolute(progress)
                                last_progress = progress

                            print(f"GRSAI NanoBanana: 收到数据，状态: {status}, 进度: {progress}%")
                    except json.JSONDecodeError as e:
                        print(f"GRSAI NanoBanana: JSON 解析错误: {e}, 原始数据: {line}")
                        continue

            # 如果没有收到任何数据
            if result_data is None:
                print("GRSAI NanoBanana: 未收到任何响应数据")
                # 返回空白图片和错误信息
                error_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (error_image, "failed", "", "未收到任何响应数据")

            # 提取返回字段
            results = result_data.get("results", [])
            status = result_data.get("status", "")
            progress = result_data.get("progress", 0)
            failure_reason = result_data.get("failure_reason", "")
            error_msg = result_data.get("error", "")

            # 检查 results 是否为 None
            if results is None:
                results = []

            print(f"GRSAI NanoBanana: 请求完成，状态: {status}, 进度: {progress}%, 结果数: {len(results)}")
            print(f"GRSAI NanoBanana: 失败原因: {failure_reason}, 错误: {error_msg}")

            # 下载并转换结果图片
            output_images = []
            total_images = len(results)

            for idx, result in enumerate(results):
                url = result.get("url", "")
                content = result.get("content", "")

                if not url:
                    print(f"GRSAI NanoBanana: 结果 {idx} 没有 URL")
                    continue

                try:
                    print(f"GRSAI NanoBanana: 开始下载图片 {idx+1}/{total_images}: {url}")

                    # 下载图片
                    img_response = requests.get(url, timeout=60)
                    img_response.raise_for_status()

                    print(f"GRSAI NanoBanana: 下载完成，大小: {len(img_response.content)} bytes")

                    # 从字节流创建 PIL Image
                    image = Image.open(io.BytesIO(img_response.content))
                    print(f"GRSAI NanoBanana: 图片模式: {image.mode}, 尺寸: {image.size}")

                    # 转换为 RGB 模式
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                        print(f"GRSAI NanoBanana: 已转换为 RGB 模式")

                    # 转换为 numpy array
                    image_np = np.array(image).astype(np.float32) / 255.0
                    print(f"GRSAI NanoBanana: numpy array 形状: {image_np.shape}")

                    # 转换为 torch tensor，添加 batch 维度 [1, H, W, C]
                    image_tensor = torch.from_numpy(image_np)[None,]
                    print(f"GRSAI NanoBanana: tensor 形状: {image_tensor.shape}")

                    output_images.append(image_tensor)
                    print(f"GRSAI NanoBanana: 成功处理图片 {idx}, 描述: {content if content else '(无描述)'}")

                except Exception as e:
                    print(f"GRSAI NanoBanana: 下载/处理图片 {idx} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # 如果没有成功下载任何图片，返回空白图片
            if len(output_images) == 0:
                print("GRSAI NanoBanana: 没有成功下载任何图片")
                error_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (error_image, status, failure_reason, error_msg)

            # 将所有图片合并成一个 batch tensor [N, H, W, C]
            # 注意：如果图片尺寸不同，需要单独处理
            try:
                result_images = torch.cat(output_images, dim=0)
                print(f"GRSAI NanoBanana: 合并后的 tensor 形状: {result_images.shape}")
            except Exception as e:
                print(f"GRSAI NanoBanana: 合并图片失败 (可能是尺寸不一致): {e}")
                # 如果合并失败（尺寸不一致），只返回第一张
                result_images = output_images[0]
                print(f"GRSAI NanoBanana: 只返回第一张图片，形状: {result_images.shape}")

            print(f"GRSAI NanoBanana: 返回 {len(output_images)} 张图片")
            return (result_images, status, failure_reason, error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"API 请求错误: {str(e)}"
            print(f"GRSAI NanoBanana: {error_msg}")
            error_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (error_image, "failed", "", error_msg)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            print(f"GRSAI NanoBanana: {error_msg}")
            import traceback
            traceback.print_exc()
            error_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (error_image, "failed", "", error_msg)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "GRSAI_NanoBanana": GRSAI_NanoBanana,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRSAI_NanoBanana": "GRSAI NanoBanana API",
}
