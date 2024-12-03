from .base_compressor import BaseImageCompressor
import torch
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

class ImageCompressorNode(BaseImageCompressor):
    """Single image compression node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        params = cls.get_compression_params()
        params["required"]["image"] = ("IMAGE",)
        
        return params

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("image", "compressed_size", "original_size", "save_path",)
    OUTPUT_NODE = True
    FUNCTION = "compress_image"
    CATEGORY = "image/processing"

    def compress_image(self, image, format, quality, resize_factor, compression_level,
                      save_image, output_prefix, output_path=""):
        # Setup output path using base class method
        self.setup_output_path(output_path)

        # Convert input tensor to numpy array
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        input_image = image.cpu().numpy()
        
        # Preprocess input image
        input_image = self.preprocess_input(input_image)
        
        # Convert to PIL Image with proper alpha channel handling
        if input_image.shape[-1] == 4:
            # RGBA image
            img = Image.fromarray((input_image * 255).astype(np.uint8), 'RGBA')
        else:
            # RGB image
            img = Image.fromarray((input_image * 255).astype(np.uint8), 'RGB')

        # Get original size info
        original_buffer = io.BytesIO()
        img.save(original_buffer, format='PNG')
        original_size = original_buffer.tell()
        original_size_str = f"{original_size / (1024 * 1024):.2f}MB" if original_size >= 1024 * 1024 else f"{original_size / 1024:.2f}KB"

        # Get save options from base class
        save_options = self.get_save_options(format, quality, compression_level)
        
        # Process image using base class method
        img = self.process_image(img, format, resize_factor, save_options)
        
        # Save to buffer and get size info
        buffer, size_str = self.save_image_to_buffer(img, format, save_options)
        
        # Handle file saving and UI info
        filename = f"{output_prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.counter:04d}.{format.lower()}"
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

        # Convert back to tensor with proper alpha channel handling
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        result = self.convert_to_tensor(compressed_img, image.shape, image.device)
        
        return {
            "result": (result, size_str, original_size_str, save_path_str),
            **ui_info
        }
