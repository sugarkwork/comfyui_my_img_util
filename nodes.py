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


class MyImageRotate:
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
        img = convert_to_pil(image).convert("RGBA")

        # 元画像を読み込み
        image = Image.open("your_image.png").convert("RGBA")

        # 回転後の画像（expand=Trueで余白あり）
        rotated = image.rotate(15, resample=Image.BICUBIC, expand=True)

        # 回転前画像を新しいサイズで中央に貼り付ける（比較用）
        bg = Image.new("RGBA", rotated.size, (0, 0, 0, 0))
        offset = (
            (rotated.width - image.width) // 2,
            (rotated.height - image.height) // 2
        )
        bg.paste(image, offset)

        # 差分を計算（余白のある部分だけ残る）
        diff = ImageChops.difference(rotated, bg)

        # 差分画像を白黒マスクに変換（透明部分だけ抽出）
        mask = diff.convert("L").point(lambda x: 255 if x > 0 else 0, mode='1')

        # 白い背景にマスクを適用して白い画像を作る
        white_bg = Image.new("RGB", rotated.size, "white")
        only_margin = Image.composite(white_bg, Image.new("RGB", rotated.size, "black"), mask)

        only_margin.save("only_margin.png")




NODE_CLASS_MAPPINGS = {
    "CalcResolution": CalcResolution
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CalcResolution": "CalcResolution",
}


def simple_test():
    pass


#if __name__ == "__main__":
#    simple_test()
