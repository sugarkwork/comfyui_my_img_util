import sys
import numpy as np
import torch
from PIL import Image, ImageChops
import cv2


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



class AutoImageSelector:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image1": ("IMAGE",),
                "rank1": ("INT", {"default": 1, "min": 1, "step": 1}),
                "image2": ("IMAGE",),
                "rank2": ("INT", {"default": 2, "min": 1, "step": 1}),
                "image3": ("IMAGE",),
                "rank3": ("INT", {"default": 3, "min": 1, "step": 1}),
                "image4": ("IMAGE",),
                "rank4": ("INT", {"default": 4, "min": 1, "step": 1}),
                "ng_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE","INT")
    RETURN_NAMES = ("image","rank")

    FUNCTION = "select"

    CATEGORY = "image"

    class ImageRank:
        def __init__(self, rank: int, image: torch.Tensor):
            self.rank = rank
            self.image = image

        def is_valid(self):
            return self.rank >= 0 and self.image is not None

    def select(self, image1: torch.Tensor = None, rank1: int = 1,
                    image2: torch.Tensor = None, rank2: int = 2,
                    image3: torch.Tensor = None, rank3: int = 3,
                    image4: torch.Tensor = None, rank4: int = 4,
                    ng_image: torch.Tensor = None) -> torch.Tensor:
        
        images = [self.ImageRank(rank1, image1), self.ImageRank(rank2, image2),
                self.ImageRank(rank3, image3), self.ImageRank(rank4, image4)]

        top_rank = sys.maxsize
        top_rank_image = None

        for img in images:
            if not img.is_valid():
                continue
            
            if ng_image is not None:
                # 完全一致ならスキップ
                if torch.equal(img.image, ng_image):
                    continue
            
            if img.rank < top_rank:
                top_rank = img.rank
                top_rank_image = img.image

        return (top_rank_image, top_rank)


class OpenCVDenoiseColored:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "h_luminance": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "h_color": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "search_window": ("INT", {"default": 5, "min": 5, "max": 50, "step": 2}),
                "template_window": ("INT", {"default": 3, "min": 3, "max": 20, "step": 2}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "Image/Post-Processing"

    def denoise(self, image, h_luminance, h_color, search_window, template_window):
        output_images = []
        
        for img in image:
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            denoised_bgr = cv2.fastNlMeansDenoisingColored(
                img_bgr,
                None,
                h_luminance,
                h_color,
                template_window,
                search_window
            )
            
            denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(denoised_rgb.astype(np.float32) / 255.0)
            output_images.append(img_tensor)

        return (torch.stack(output_images),)


NODE_CLASS_MAPPINGS = {
    "Simple Image Rotate": SimpleImageRotate,
    "Auto Image Selector": AutoImageSelector,
    "OpenCVDenoiseColored": OpenCVDenoiseColored,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Simple Image Rotate": "Simple Image Rotate",
    "Auto Image Selector": "Auto Image Selector",
    "OpenCVDenoiseColored": "OpenCV Denoise (Luma/Chroma)",
}


def simple_test():
    mir = SimpleImageRotate()
    image = Image.open("test.png")
    mir.rotate(image, 3555.1)

    ais = AutoImageSelector()
    result = ais.select(image1="a", rank1=2, image2="b", rank2=2, image3="c", rank3=2, image4="d", rank4=0)
    print(result)


#if __name__ == "__main__":
#    simple_test()
