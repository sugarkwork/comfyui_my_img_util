import os
import numpy as np
import torch
from PIL import Image, ImageChops


def convert_to_pil(image: torch.Tensor) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(np.clip(255. * image.squeeze(), 0, 255).astype(np.uint8))
    if isinstance(image, torch.Tensor):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    raise ValueError(f"Unknown image type: {type(image)}")


def convert_to_tensor(image: Image.Image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class SimpleImageRotate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask")

    FUNCTION = "rotate"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def rotate(self, image:torch.Tensor, angle:float=0.0):
        image = convert_to_pil(image).convert("RGBA")

        rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True)
        rotated.save("rotated.png")

        alpha = rotated.split()[-1]
        reverse_alpha = ImageChops.invert(alpha).convert("RGB")
        reverse_alpha.save("reverse.png")

        return (convert_to_tensor(rotated), convert_to_tensor(reverse_alpha))



NODE_CLASS_MAPPINGS = {
    "Simple Image Rotate": SimpleImageRotate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Simple Image Rotate": "Simple Image Rotate",
}


def simple_test():
    mir = SimpleImageRotate()
    image = Image.open("test.png")
    mir.rotate(image, 3555.1)


#if __name__ == "__main__":
#    simple_test()
