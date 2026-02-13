import sys
import numpy as np
import torch
import torch.nn.functional as F
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


class ImageResizeAndCrop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": 24576, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 24576, "step": 1}),
                "method": (["lanczos", "bicubic", "bilinear", "area", "nearest"],),
                "h_align": (["left", "center", "right"], {"default": "center"}),
                "v_align": (["top", "center", "bottom"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_and_crop"
    CATEGORY = "image"

    def resize_and_crop(self, image, width, height, method="lanczos", h_align="center", v_align="center"):
        # image layout is (B, H, W, C)
        B, H, W, C = image.shape
        
        # Determine strict resize to cover the target area
        scale_w = width / W
        scale_h = height / H
        scale = max(scale_w, scale_h)
        
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))
        
        # execute resize
        if method == "lanczos":
            resized_list = []
            for i in range(B):
                img_tensor = image[i]
                pil_img = convert_to_pil(img_tensor)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                resized_list.append(convert_to_tensor(pil_img))
            resized = torch.cat(resized_list, dim=0)
        else:
            # torch expects (B, C, H, W)
            permuted = image.permute(0, 3, 1, 2)
            
            align_corners = False
            if method in ["bicubic", "bilinear"]:
                align_corners = False
            
            resized_permuted = F.interpolate(permuted, size=(new_h, new_w), mode=method, align_corners=align_corners)
            resized = resized_permuted.permute(0, 2, 3, 1)
            
        # execute crop
        curr_h, curr_w = resized.shape[1], resized.shape[2]
        
        if h_align == "left":
            x = 0
        elif h_align == "right":
            x = curr_w - width
        else:
            x = (curr_w - width) // 2
            
        if v_align == "top":
            y = 0
        elif v_align == "bottom":
            y = curr_h - height
        else:
            y = (curr_h - height) // 2
            
        # bound check
        x = max(0, min(x, curr_w - width))
        y = max(0, min(y, curr_h - height))
        
        cropped = resized[:, y:y+height, x:x+width, :]
        return (cropped,)


class ImageTrimEdges:
    """画像の上下左右を指定量（ピクセルまたはパーセント）でトリミングするノード"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "top": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
                "bottom": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
                "left": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
                "right": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
                "is_pixel": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "trim"
    CATEGORY = "image"

    def trim(self, image: torch.Tensor, top: float, bottom: float, left: float, right: float, is_pixel: bool):
        # image layout: (B, H, W, C)
        B, H, W, C = image.shape

        if is_pixel:
            t = int(top)
            b = int(bottom)
            l = int(left)
            r = int(right)
        else:
            # パーセント → ピクセルに変換
            t = int(H * min(top, 99.0) / 100.0)
            b = int(H * min(bottom, 99.0) / 100.0)
            l = int(W * min(left, 99.0) / 100.0)
            r = int(W * min(right, 99.0) / 100.0)

        # 切り取り後のサイズが最低1ピクセル残るようにクランプ
        if t + b >= H:
            t = 0
            b = 0
        if l + r >= W:
            l = 0
            r = 0

        y_start = t
        y_end = H - b
        x_start = l
        x_end = W - r

        cropped = image[:, y_start:y_end, x_start:x_end, :]
        return (cropped,)


NODE_CLASS_MAPPINGS = {
    "Simple Image Rotate": SimpleImageRotate,
    "Auto Image Selector": AutoImageSelector,
    "OpenCVDenoiseColored": OpenCVDenoiseColored,
    "Image Resize And Crop": ImageResizeAndCrop,
    "Image Trim Edges": ImageTrimEdges,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Simple Image Rotate": "Simple Image Rotate",
    "Auto Image Selector": "Auto Image Selector",
    "OpenCVDenoiseColored": "OpenCV Denoise (Luma/Chroma)",
    "Image Resize And Crop": "Image Resize And Crop",
    "Image Trim Edges": "Image Trim Edges",
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
