from skimage.measure import label, regionprops
from skimage.color import label2rgb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gc


from modal import Stub, method
import os
from modal import Image, Secret
import pathlib
import modal
from modal import Mount

def download_models():
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained("facebook/sam-vit-large")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "Pillow", "torchvision", "matplotlib", "scikit-image", "openai")
    .run_function(download_models)
)

stub = modal.Stub("chatPID", image=image)

local_dir = pathlib.Path(os.getcwd())
remote_dir = "./"

mount = Mount.from_local_dir(local_dir, remote_path=remote_dir)

@stub.cls(gpu="A100", cpu=16, container_idle_timeout=300)
class ImageProcessor:
    def __enter__(self):
        from transformers import AutoModel, pipeline

        import os

        self.model = AutoModel.from_pretrained("facebook/sam-vit-large")
        self.generator = pipeline("mask-generation", model="facebook/sam-vit-large", device="cuda:0")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @method()
    def process_mask(self, image: np.array) -> dict:
        from PIL import Image
        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.measure import label, regionprops
        import gc
        import torch

        raw_image = Image.fromarray(image).convert("RGB")
        aspect_ratio = raw_image.size[0] / raw_image.size[1]
        new_height = 100
        new_width = int(new_height * aspect_ratio)
        raw_image = raw_image.resize((new_width, new_height), Image.LANCZOS)
        width, height = raw_image.size
        left = width * 0.2
        right = width * 0.8
        top = height * 0.05
        bottom = height * 0.95
        raw_image = raw_image.crop((left, top, right, bottom))

   
        outputs = self.generator(raw_image, points_per_batch=256)
        masks = outputs["masks"]

        return masks