from .base_compressor import BaseImageCompressor
import os
from PIL import Image

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

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("processed_files", "save_path",)
    OUTPUT_NODE = True
    FUNCTION = "compress_images"
    CATEGORY = "image"

    def compress_images(self, input_path, format, quality, resize_factor, compression_level,
                       save_image, output_prefix, output_path=""):
        # Setup output path using base class method
        self.setup_output_path(output_path)

        # Validate input path
        if not os.path.exists(input_path):
            return {"result": (f"Input path does not exist: {input_path}", "")}

        # Get save options from base class
        save_options = self.get_save_options(format, quality, compression_level)

        # Supported image formats
        supported_formats = {'.png', '.jpg', '.jpeg', '.webp'}
        processed_files = []
        
        # Process all images in directory
        for root, _, files in os.walk(input_path):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    input_file = os.path.join(root, file)
                    try:
                        # Read image
                        img = Image.open(input_file)
                        
                        # Process image using base class method
                        img = self.process_image(img, format, resize_factor, save_options)
                        
                        if save_image:
                            # Generate output filename
                            filename = f"{output_prefix}{os.path.splitext(file)[0]}.{format.lower()}"
                            save_path = os.path.join(self.output_dir, filename)
                            
                            # Save compressed image
                            img.save(save_path, format=format, **save_options)
                            processed_files.append(f"Processed: {file} -> {filename}")
                            self.counter += 1
                        
                    except Exception as e:
                        processed_files.append(f"Failed to process {file}: {str(e)}")
                        continue

        # Generate results
        processed_summary = "\n".join(processed_files)
        save_path_str = f"Saved to: {self.output_dir}" if save_image else "Files not saved"
        
        return {
            "result": (processed_summary, save_path_str)
        }