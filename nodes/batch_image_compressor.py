from .base_compressor import BaseImageCompressor
import os
from PIL import Image
import io

class BatchImageCompressorNode(BaseImageCompressor):
    """Batch image compression node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        params = cls.get_compression_params()
        params["required"]["input_path"] = ("STRING", {
            "default": "",
            "multiline": False
        })
        
        return params

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("compression_info",)
    OUTPUT_NODE = True
    FUNCTION = "compress_images"
    CATEGORY = "image"

    def compress_images(self, input_path, format, quality=85, resize_factor=1.0,
                       compression_level=6, save_image=True, output_prefix="compressed_",
                       output_path=""):
        """Compress input images and return compression information"""
        self.setup_output_path(output_path)

        # Validate input path
        if not os.path.exists(input_path):
            return {"result": (f"Input path does not exist: {input_path}",)}

        # Get save options from base class
        save_options = self.get_save_options(format, quality, compression_level)

        # Supported image formats
        supported_formats = {'.png', '.jpg', '.jpeg', '.webp'}
        compressed_sizes = []
        original_sizes = []
        save_paths = []
        
        # Process all images in directory
        for root, _, files in os.walk(input_path):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    input_file = os.path.join(root, file)
                    try:
                        # Read image
                        img = Image.open(input_file)
                        
                        # Get original size info
                        original_size_str = self.get_original_size(img)
                        original_sizes.append(original_size_str)
                        
                        # Process image using base class method
                        img = self.process_image(img, format, resize_factor, save_options)
                        
                        # Save to buffer and get size info
                        buffer, size_str = self.save_image_to_buffer(img, format, save_options)
                        compressed_sizes.append(size_str)
                        
                        # Handle file saving
                        filename = f"{output_prefix}{os.path.splitext(file)[0]}.{format.lower()}"
                        save_path = os.path.join(self.output_dir, filename)
                        
                        if save_image:
                            with open(save_path, 'wb') as f:
                                buffer.seek(0)
                                f.write(buffer.getvalue())
                            self.counter += 1
                            save_path_str = f"{save_path}"
                        else:
                            save_path_str = "File not saved"
                        save_paths.append(save_path_str)
                        
                    except Exception as e:
                        print(f"Failed to process {file}: {str(e)}")
                        continue

        # For batch processing, we'll join the size and path information with newlines
        info_lines = []
        for path, orig, comp in zip(save_paths, original_sizes, compressed_sizes):
            info_lines.append(f"{path}: {orig} -> {comp}")
        compression_info = "Compression results:\n\n" + "\n".join(info_lines)
        
        return {
            "result": (compression_info,)
        }