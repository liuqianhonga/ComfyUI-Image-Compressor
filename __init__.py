from .nodes.image_compressor import ImageCompressorNode
from .nodes.batch_image_compressor import BatchImageCompressorNode

NODE_CLASS_MAPPINGS = {
    "ImageCompressor": ImageCompressorNode,
    "BatchImageCompressor": BatchImageCompressorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCompressor": "Image Compressor",
    "BatchImageCompressor": "Batch Image Compressor"
}