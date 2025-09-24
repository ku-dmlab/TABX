import os

import tqdm
import glob

import numpy as np
from PIL import Image


def decrease_contrast(img_path: str, decrease_level: float = 0.3) -> None:
    img = Image.open(img_path)
    img_array = np.array(img)
    dark_img_array = np.array(img)
    dark_img_array[:, :, :3] = dark_img_array[:, :, :3] * decrease_level

    # Get full transparent pixels
    mask = img_array[:, :, -1] > 0

    # Decrease contrast
    img_array = np.where(mask[:, :, np.newaxis], dark_img_array, img_array)

    res = Image.fromarray(img_array, mode="RGBA")
    name = img_path.split("/")[-1].split(".")[0]
    res.save(os.path.join("./assets/", f"{name}_dead.png"), format="PNG")


if __name__ == "__main__":
    img_paths = glob.glob("./assets/units/*.png")
    for img_path in tqdm.tqdm(img_paths):
        decrease_contrast(img_path, 0.3)
