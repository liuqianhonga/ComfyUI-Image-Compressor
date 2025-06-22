from .base_compressor import BaseImageCompressor
import torch
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import random

class ImageCompressorNode(BaseImageCompressor):
    """Node for compressing images in ComfyUI"""
    
    @classmethod
    def INPUT_TYPES(cls):
        params = cls.get_compression_params()
        params["required"]["images"] = ("IMAGE",)
        
        return params

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("compression_info", "images")
    OUTPUT_NODE = True
    FUNCTION = "compress_image"
    CATEGORY = "image"

    def compress_image(self, images, format, quality=85, resize_factor=1.0,
                      compression_level=6, save_image=True, output_prefix="compressed_",
                      output_path=""):
        """Compress input images and return compression information"""
        self.setup_output_path(output_path)
        
        ui_images = []
        compressed_images = []
        compressed_infos = []
        
        # Check if output path is within ComfyUI output directory
        try:
            base_path = os.path.abspath(self.base_output_dir)
            output_path = os.path.abspath(self.output_dir)
            is_within_comfyui = output_path.startswith(base_path)
        except Exception as e:
            is_within_comfyui = False
        
        # Process each image in the batch
        for batch_number, img_tensor in enumerate(images):
            # Convert input tensor to numpy array
            if not isinstance(img_tensor, torch.Tensor):
                img_tensor = torch.from_numpy(img_tensor)
            input_image = img_tensor.cpu().numpy()
            
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
            original_size_str = self.get_original_size(img)

            # Get save options from base class
            save_options = self.get_save_options(format, quality, compression_level)
            
            # Process image using base class method
            img = self.process_image(img, format, resize_factor, save_options)
            
            # Save to buffer and get size info
            buffer, size_str = self.save_image_to_buffer(img, format, save_options)
            
            # Handle file saving and UI info
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # Add milliseconds
            random_suffix = random.randint(1000, 9999)  # 4-digit random number
            filename = f"{output_prefix}{timestamp}_{self.counter:04d}_{random_suffix}.{format.lower()}"
            save_path = os.path.join(self.output_dir, filename)
            
            if save_image:
                with open(save_path, 'wb') as f:
                    buffer.seek(0)
                    f.write(buffer.getvalue())
                self.counter += 1
                save_path_str = f"{save_path}"
                # Only add to UI images if within ComfyUI output directory
                if is_within_comfyui:
                    ui_images.append({
                        "filename": filename,
                        "subfolder": self.output_dir,
                        "type": 'output'
                    })
            else:
                save_path_str = "File not saved"
            
            # Load the compressed image from buffer
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            compressed_img.load() # Make sure the image is fully loaded
            
            # Convert compressed image to tensor for output
            if compressed_img.mode == 'RGBA':
                img_np = np.array(compressed_img).astype(np.float32) / 255.0
            else:
                # 如果原图有 alpha 通道但压缩后没有（比如 JPEG），使用 RGB 模式
                if len(img_tensor.shape) > 2 and img_tensor.shape[-1] == 4:
                    rgb_img = compressed_img.convert('RGB')
                    img_np = np.array(rgb_img).astype(np.float32) / 255.0
                else:
                    img_np = np.array(compressed_img.convert('RGB')).astype(np.float32) / 255.0
                    if len(img_np.shape) == 2:
                        img_np = np.stack([img_np] * 3, axis=-1)

            if len(img_tensor.shape) == 4:
                img_np = np.expand_dims(img_np, 0)

            # Convert numpy array to torch tensor for ComfyUI compatibility
            img_to_tensor = torch.from_numpy(img_np).to(img_tensor.device)
            # Add processed image to list
            compressed_images.append(img_to_tensor)
            
            # Collect compression info
            compressed_infos.append(f"{save_path_str}: {original_size_str} -> {size_str}")
        
        # Only include UI images if within ComfyUI output directory
        result = {"result": ("Compression results:\n\n" + "\n".join(compressed_infos), compressed_images)}
        if is_within_comfyui and ui_images:
            result["ui"] = {"images": ui_images}
        return result
