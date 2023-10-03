import numpy as np
import matplotlib.pyplot as plt
import gc
from PIL import Image

import sys


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
  plt.show()
  del mask
  gc.collect()

from transformers import pipeline
generator = pipeline("mask-generation", model="facebook/sam-vit-base", device=-1)

def generate_mask(image):
    return generator(image, points_per_batch=64)

def main():
    raw_image = Image.open("crossfar.png").convert("RGB")
    masks = generate_mask(raw_image)
    
    show_masks_on_image(raw_image, masks)

if __name__ == "__main__":
    main()
