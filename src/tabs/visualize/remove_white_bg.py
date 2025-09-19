import os

import tqdm
import glob

import numpy as np
from PIL import Image


def remove_white_bg(img_path: str, white_level: int = 255) -> None:
    img = Image.open(img_path)
    img_array = np.array(img)

    img_a = img.convert("RGBA")
    img_a_array = np.array(img_a)

    # Get white pixels
    mask = img_array >= white_level
    mask = np.sum(mask, axis=-1) >= 3

    # Make white pixels transparent
    img_a_array[:, :, -1] = np.where(mask, 0, img_a_array[:, :, -1])

    res = Image.fromarray(img_a_array, mode="RGBA")
    res.save(os.path.join("./alpha/units", img_path.split("/")[-1]), format="PNG")


if __name__ == "__main__":
    img_paths = glob.glob("./assets/*.png")
    for img_path in tqdm.tqdm(img_paths):
        remove_white_bg(img_path, 230)
