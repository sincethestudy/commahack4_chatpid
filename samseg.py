from modal import Stub, method, gpu, Image
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

class ImageProcessor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @method(gpu=gpu.A100(count=1))
    def process_image(self, image_path: str) -> dict:
        config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
        checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

        # build the model from a config file and a checkpoint file
        model = init_model(config_file, checkpoint_file, device='cuda:0')

        # test a single image and show the results
        img = image_path  # or img = mmcv.imread(img), which will only load it once
        result = inference_model(model, img)
        # visualize the results in a new window
        show_result_pyplot(model, img, result, show=True)
        # or save the visualization results to image files
        # you can change the opacity of the painted segmentation map in (0, 1].
        show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)

        return result

image = Image.debian_slim().pip_install("mmcv", "mmseg")
stub = Stub('ImageProcessor', ImageProcessor, image=image)

