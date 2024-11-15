import os
import numpy as np
import torch
from PIL import Image
import io
from datetime import datetime
import folder_paths

class ImageCompressorNode:
    """图片压缩节点"""
    
    def __init__(self):
        # 使用ComfyUI的输出目录
        self.base_output_dir = folder_paths.get_output_directory()
        self.output_dir = os.path.join(self.base_output_dir, 'compressed')
        self.counter = 0
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                
                "format": (["PNG", "WEBP", "JPEG"],),
                "quality": ("INT", {
                    "default": 85,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "resize_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "compression_level": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 9,
                    "step": 1,
                    "display": "slider"
                }),
                
                "save_image": ("BOOLEAN", {"default": True}),
                "output_prefix": ("STRING", {"default": "compressed_"}),
            },
            "optional": {
                "output_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("image", "compressed_size", "original_size", "save_path",)
    OUTPUT_NODE = True
    FUNCTION = "compress_image"
    CATEGORY = "image/processing"

    def compress_image(self, image, format, quality, resize_factor, compression_level,
                      save_image, output_prefix, output_path=""):
        # 设置输出路径
        if output_path:
            self.output_dir = output_path
        else:
            self.output_dir = os.path.join(self.base_output_dir, 'compressed')
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 确保输入是 torch.Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        
        # 将 Tensor 转换为 numpy 进行处理
        input_image = image.cpu().numpy()
        
        # 确保是3D格式 (H, W, C)
        if len(input_image.shape) == 4:
            input_image = input_image[0]
        
        # 调整通道顺序，确保是RGB格式
        if input_image.shape[-1] != 3:
            input_image = np.transpose(input_image, (1, 2, 0))
        
        # 将numpy数组转换为PIL图像
        img = Image.fromarray((input_image * 255).astype(np.uint8))
        
        # 获取原始图像大小
        original_buffer = io.BytesIO()
        img.save(original_buffer, format='PNG')
        original_size = original_buffer.tell()
        if original_size >= 1024 * 1024:
            original_size_str = f"{original_size / (1024 * 1024):.2f}MB"
        else:
            original_size_str = f"{original_size / 1024:.2f}KB"

        # 如果需要调整尺寸
        if resize_factor < 1.0:
            new_size = tuple(int(dim * resize_factor) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 保存到内存缓冲区并压缩
        buffer = io.BytesIO()
        
        # 根据不同格式优化保存参数
        save_options = {}
        if format == "PNG":
            save_options.update({
                'optimize': True,
                'compression_level': compression_level
            })
        elif format == "JPEG":
            save_options.update({
                'quality': quality,
                'optimize': True,
                'subsampling': 1
            })
        elif format == "WEBP":
            save_options.update({
                'quality': quality,
                'method': 6,
                'lossless': False
            })
        
        # 保存压缩后的图像
        img.save(buffer, format=format, **save_options)
        
        # 获取压缩后的文件大小
        compressed_size = buffer.tell()
        if compressed_size >= 1024 * 1024:
            size_str = f"{compressed_size / (1024 * 1024):.2f}MB"
        else:
            size_str = f"{compressed_size / 1024:.2f}KB"
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_prefix}{timestamp}_{self.counter:04d}.{format.lower()}"
        save_path = os.path.join(self.output_dir, filename)
        
        # 如果需要保存图片
        if save_image:
            # 保存压缩后的图片到文件
            with open(save_path, 'wb') as f:
                buffer.seek(0)
                f.write(buffer.getvalue())
            self.counter += 1
            save_path_str = f"已保存到: {save_path}"
        else:
            save_path_str = "未保存文件"
        
        # 从压缩后的数据创建新的图像
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # 如果是JPEG格式，转换为RGB（避免RGBA问题）
        if format == "JPEG" and compressed_img.mode != 'RGB':
            compressed_img = compressed_img.convert('RGB')
        
        # 转换为numpy数组
        compressed_array = np.array(compressed_img).astype(np.float32) / 255.0
        
        # 确保格式正确
        if len(compressed_array.shape) == 2:
            compressed_array = np.stack([compressed_array] * 3, axis=-1)
        
        # 转换回原始格式
        if len(image.shape) == 4:
            compressed_array = np.expand_dims(compressed_array, 0)
        
        # 转换为tensor并返回压缩后的图像
        result = torch.from_numpy(compressed_array).to(image.device)
        
        # 返回压缩后的图像和信息
        return (result, size_str, original_size_str, save_path_str)