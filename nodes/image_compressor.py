from .base_compressor import BaseImageCompressor
import torch
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

class ImageCompressorNode(BaseImageCompressor):
    """Node for compressing images in ComfyUI"""
    
    @classmethod
    def INPUT_TYPES(cls):
        params = cls.get_compression_params()
        params["required"]["images"] = ("IMAGE",)
        
        return params

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("compression_info",)
    OUTPUT_NODE = True
    FUNCTION = "compress_image"
    CATEGORY = "image"

    def compress_image(self, images, format, quality=85, resize_factor=1.0,
                      compression_level=6, save_image=True, output_prefix="compressed_",
                      output_path=""):
        """Compress input images and return compression information"""
        self.setup_output_path(output_path)
        
        ui_images = []
        compressed_sizes = []
        original_sizes = []
        save_paths = []
        
        # Check if output path is within ComfyUI output directory
        try:
            base_path = os.path.abspath(self.base_output_dir)
            output_path = os.path.abspath(self.output_dir)
            is_within_comfyui = output_path.startswith(base_path)
        except:
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
            original_sizes.append(original_size_str)

            # Get save options from base class
            save_options = self.get_save_options(format, quality, compression_level)
            
            # Process image using base class method
            img = self.process_image(img, format, resize_factor, save_options)
            
            # Save to buffer and get size info
            buffer, size_str = self.save_image_to_buffer(img, format, save_options)
            compressed_sizes.append(size_str)
            
            # Handle file saving and UI info
            filename = f"{output_prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.counter:04d}.{format.lower()}"
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
            save_paths.append(save_path_str)
        
        # For batch processing, we'll join the size and path information with newlines
        info_lines = []
        for path, orig, comp in zip(save_paths, original_sizes, compressed_sizes):
            info_lines.append(f"{path}: {orig} -> {comp}")
        compression_info = "Compression results:\n\n" + "\n".join(info_lines)
        
        print(f"Compression info: {compression_info}")


        # Only include UI images if within ComfyUI output directory
        result = {"result": (compression_info,)}
        if is_within_comfyui and ui_images:
            result["ui"] = {"images": ui_images}
        return result
