import os
import numpy as np
from PIL import Image
import folder_paths
import io
import torch

class BaseImageCompressor:
    """Base class for image compression nodes"""
    
    def __init__(self):
        self.base_output_dir = folder_paths.get_output_directory()
        self.output_dir = os.path.join(self.base_output_dir, 'compressed')
        self.counter = 0

    def setup_output_path(self, output_path=""):
        """Setup output directory"""
        self.output_dir = output_path or os.path.join(self.base_output_dir, 'compressed')
        os.makedirs(self.output_dir, exist_ok=True)

    def get_save_options(self, format, quality, compression_level):
        """Get save options based on format"""
        save_options = {}
        if format == "PNG":
            save_options.update({'optimize': True, 'compression_level': compression_level})
        elif format == "JPEG":
            save_options.update({'quality': quality, 'optimize': True, 'subsampling': 1})
        elif format == "WEBP":
            save_options.update({
                'quality': quality,
                'method': 6,
                'lossless': False,
                'alpha_quality': quality
            })
        return save_options

    def process_image(self, img, format, resize_factor, save_options):
        """Process single image with resize and format conversion"""
        if resize_factor < 1.0:
            new_size = tuple(int(dim * resize_factor) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        if format == "JPEG" and img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background

        return img

    def save_image_to_buffer(self, img, format, save_options):
        """Save image to buffer and return size info"""
        buffer = io.BytesIO()
        img.save(buffer, format=format, **save_options)
        size = buffer.tell()
        size_str = f"{size / (1024 * 1024):.2f}MB" if size >= 1024 * 1024 else f"{size / 1024:.2f}KB"
        return buffer, size_str

    @staticmethod
    def get_compression_params():
        """Get common compression parameters"""
        params = {
            "required": {
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
                })
            }
        }
        
        return params

    def convert_to_tensor(self, img, original_shape, device):
        """Convert PIL Image back to tensor with proper alpha channel handling"""
        if img.mode == 'RGBA':
            compressed_array = np.array(img).astype(np.float32) / 255.0
        else:
            # 如果原图有 alpha 通道但压缩后没有（比如 JPEG），使用 RGB 模式
            if len(original_shape) > 2 and original_shape[-1] == 4:
                rgb_img = img.convert('RGB')
                compressed_array = np.array(rgb_img).astype(np.float32) / 255.0
            else:
                compressed_array = np.array(img.convert('RGB')).astype(np.float32) / 255.0
                if len(compressed_array.shape) == 2:
                    compressed_array = np.stack([compressed_array] * 3, axis=-1)

        # 处理批处理维度
        if len(original_shape) == 4:
            compressed_array = np.expand_dims(compressed_array, 0)

        return torch.from_numpy(compressed_array).to(device) 

    def preprocess_input(self, input_image):
        """Preprocess input array to handle special shapes and channel order"""
        # Handle special (1, 1, C) shape
        if len(input_image.shape) == 3 and input_image.shape[0] == 1 and input_image.shape[1] == 1:
            channels = input_image.shape[2]
            target_channels = 4 if channels % 4 == 0 else 3
            side_length = int(np.sqrt(channels / target_channels))
            if channels % target_channels:
                side_length += 1
            
            flat_data = input_image[0, 0, :channels-(channels % target_channels)].reshape(-1, target_channels)
            input_image = np.zeros((side_length, side_length, target_channels), dtype=np.float32)
            input_image[:flat_data.shape[0]//side_length, :side_length] = flat_data.reshape(-1, side_length, target_channels)
        
        # Handle 4D tensor (batch) case
        elif len(input_image.shape) == 4:
            input_image = input_image[0]
        
        # Handle channel order
        if len(input_image.shape) == 3 and input_image.shape[-1] not in [3, 4]:
            if input_image.shape[0] in [3, 4]:
                input_image = np.transpose(input_image, (1, 2, 0))
        
        # Normalize values to [0, 1] range if needed
        if input_image.max() > 1.0:
            input_image = input_image / 255.0
        
        return input_image