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
        self.output_dir = output_path or os.path.join(self.base_output_dir, 'compressed')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 确保输入是 torch.Tensor 并转换为 numpy
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        input_image = image.cpu().numpy()
        
        # 处理特殊的 (1, 1, C) 形状
        if len(input_image.shape) == 3 and input_image.shape[0] == 1 and input_image.shape[1] == 1:
            channels = input_image.shape[2]
            side_length = int(np.sqrt(channels / 4)) if channels % 4 == 0 else int(np.sqrt(channels / 3))
            target_channels = 4 if channels % 4 == 0 else 3
            if channels % target_channels != 0:
                side_length = int(channels / target_channels) + 1
            
            new_image = np.zeros((side_length, side_length, target_channels), dtype=np.float32)
            flat_data = input_image[0, 0, :channels-(channels % target_channels)].reshape(-1, target_channels)
            new_image[:flat_data.shape[0]//side_length, :side_length] = flat_data.reshape(-1, side_length, target_channels)
            input_image = new_image
        elif len(input_image.shape) == 4:
            input_image = input_image[0]
        
        # 调整通道顺序和值域
        if len(input_image.shape) == 3 and input_image.shape[-1] not in [3, 4]:
            if input_image.shape[0] in [3, 4]:
                input_image = np.transpose(input_image, (1, 2, 0))
        
        if input_image.max() > 1.0:
            input_image = input_image / 255.0
        
        # 转换为PIL图像
        img = Image.fromarray((input_image * 255).astype(np.uint8), 
                            'RGBA' if input_image.shape[-1] == 4 else 'RGB')
        
        # 获取原始大小
        original_buffer = io.BytesIO()
        img.save(original_buffer, format='PNG')
        original_size = original_buffer.tell()
        original_size_str = f"{original_size / (1024 * 1024):.2f}MB" if original_size >= 1024 * 1024 else f"{original_size / 1024:.2f}KB"

        # 调整尺寸
        if resize_factor < 1.0:
            new_size = tuple(int(dim * resize_factor) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 设置保存选项
        save_options = {}
        if format == "PNG":
            save_options.update({'optimize': True, 'compression_level': compression_level})
        elif format == "JPEG":
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            save_options.update({'quality': quality, 'optimize': True, 'subsampling': 1})
        elif format == "WEBP":
            save_options.update({'quality': quality, 'method': 6, 'lossless': False, 'alpha_quality': quality})
        
        # 压缩并保存
        buffer = io.BytesIO()
        img.save(buffer, format=format, **save_options)
        compressed_size = buffer.tell()
        size_str = f"{compressed_size / (1024 * 1024):.2f}MB" if compressed_size >= 1024 * 1024 else f"{compressed_size / 1024:.2f}KB"
        
        # 处理文件保存和UI信息
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_prefix}{timestamp}_{self.counter:04d}.{format.lower()}"
        save_path = os.path.join(self.output_dir, filename)
        
        if save_image:
            with open(save_path, 'wb') as f:
                buffer.seek(0)
                f.write(buffer.getvalue())
            self.counter += 1
            save_path_str = f"Saved to: {save_path}"
            ui_info = {"ui": {"images": [{"filename": filename, "subfolder": self.output_dir, "type": 'output'}]}}
        else:
            save_path_str = "File not saved"
            ui_info = {"ui": {"images": []}}

        # 处理压缩后的图像
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        if format == "JPEG":
            compressed_array = np.array(compressed_img).astype(np.float32) / 255.0
        else:
            if compressed_img.mode == 'RGBA':
                compressed_array = np.array(compressed_img).astype(np.float32) / 255.0
            else:
                if input_image.shape[-1] == 4:
                    rgb_img = compressed_img.convert('RGB')
                    compressed_array = np.array(rgb_img).astype(np.float32) / 255.0
                else:
                    compressed_array = np.array(compressed_img.convert('RGB')).astype(np.float32) / 255.0
                    if len(compressed_array.shape) == 2:
                        compressed_array = np.stack([compressed_array] * 3, axis=-1)
        
        # 确保输出格式正确
        if len(image.shape) == 4:
            compressed_array = np.expand_dims(compressed_array, 0)
        
        result = torch.from_numpy(compressed_array).to(image.device)
        
        return {
            "result": (result, size_str, original_size_str, save_path_str),
            **ui_info
        }
