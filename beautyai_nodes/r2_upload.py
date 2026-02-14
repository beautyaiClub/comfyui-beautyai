"""
R2 上传节点
将视频文件上传到 Cloudflare R2 并返回公开 URL
"""

import os
import boto3
from botocore.exceptions import ClientError


class BeautyAI_UploadVideoToR2:
    """
    上传视频到 Cloudflare R2
    返回公开访问的 URL
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            },
            "optional": {
                "object_prefix": ("STRING", {
                    "default": "videos/",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("r2_url",)
    FUNCTION = "upload_to_r2"
    OUTPUT_NODE = True
    CATEGORY = "BeautyAI"

    def upload_to_r2(self, video_path, object_prefix="videos/"):
        """
        上传视频到 R2

        Args:
            video_path: 视频文件的完整路径
            object_prefix: R2 对象前缀（目录）

        Returns:
            tuple: (r2_url,)
        """
        try:
            # 从环境变量读取 R2 配置
            r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
            r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
            r2_bucket_name = os.environ.get("R2_BUCKET_NAME")
            r2_endpoint = os.environ.get("R2_ENDPOINT")
            r2_public_base_url = os.environ.get("R2_PUBLIC_BASE_URL")

            # 验证必需的环境变量
            missing_vars = []
            if not r2_access_key:
                missing_vars.append("R2_ACCESS_KEY_ID")
            if not r2_secret_key:
                missing_vars.append("R2_SECRET_ACCESS_KEY")
            if not r2_bucket_name:
                missing_vars.append("R2_BUCKET_NAME")
            if not r2_endpoint:
                missing_vars.append("R2_ENDPOINT")
            if not r2_public_base_url:
                missing_vars.append("R2_PUBLIC_BASE_URL")

            if missing_vars:
                raise ValueError(f"缺少必需的环境变量: {', '.join(missing_vars)}")

            # 验证视频文件存在
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")

            # 获取文件名
            filename = os.path.basename(video_path)

            # 构建 R2 对象键
            object_key = f"{object_prefix}{filename}"

            print(f"BeautyAI R2 Upload: 开始上传")
            print(f"BeautyAI R2 Upload: 本地文件: {video_path}")
            print(f"BeautyAI R2 Upload: R2 对象键: {object_key}")

            # 创建 S3 客户端（R2 兼容 S3 API）
            s3_client = boto3.client(
                's3',
                endpoint_url=r2_endpoint,
                aws_access_key_id=r2_access_key,
                aws_secret_access_key=r2_secret_key,
                region_name='auto'  # R2 使用 'auto'
            )

            # 上传文件
            file_size = os.path.getsize(video_path)
            print(f"BeautyAI R2 Upload: 文件大小: {file_size} bytes")

            s3_client.upload_file(
                video_path,
                r2_bucket_name,
                object_key,
                ExtraArgs={
                    'ContentType': 'video/mp4'
                }
            )

            # 构建公开 URL
            r2_url = f"{r2_public_base_url}/{object_key}"

            print(f"BeautyAI R2 Upload: 上传成功")
            print(f"BeautyAI R2 Upload: 公开 URL: {r2_url}")

            # 返回格式：handler 会提取这个 r2_url
            return {
                "ui": {
                    "r2_url": r2_url
                },
                "result": (r2_url,)
            }

        except ClientError as e:
            error_msg = f"R2 上传失败: {str(e)}"
            print(f"BeautyAI R2 Upload: {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = f"上传失败: {str(e)}"
            print(f"BeautyAI R2 Upload: {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "BeautyAI_UploadVideoToR2": BeautyAI_UploadVideoToR2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BeautyAI_UploadVideoToR2": "上传视频到 R2",
}
