# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL,SD3ControlNetModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
from huggingface_hub import login
import mediapipe as mp
import pathlib
from dotenv import load_dotenv
from huggingface_hub import login
import os
# load .env
load_dotenv()

token = os.environ.get('token')


login(token)  # 토큰 직접 입력

class SD3CannyImageProcessor(VaeImageProcessor):
    def __init__(self):
        super().__init__(do_normalize=False)
    def preprocess(self, image, **kwargs):
        image = super().preprocess(image, **kwargs)
        image = image * 255 * 0.5 + 0.5
        return image
    def postprocess(self, image, do_denormalize=True, **kwargs):
        do_denormalize = [True] * image.shape[0]
        image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
        return image

def load_image_from_pil(pil_image):
  image_np = np.array(pil_image)
  if image_np.ndim == 3 and image_np.shape[2] == 3:
      gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
  else:
      gray = image_np
  edges = cv2.Canny(gray, 200, 400)
  edge_image = Image.fromarray(edges, mode='L')
  edge_image.save('canny.png')
  return edge_image.convert("RGB")

imgProcesser = SD3CannyImageProcessor()


# initialize the models and pipeline
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
)

pipe = pipe.to('cuda')
# pipe.enable_xformers_memory_efficient_attention()
pipe.image_processor = imgProcesser
# get canny image
prompt = "realistic blueberry muffin"

img_path = "C:/Users/X423/Downloads/genAI/genAI/train/canny/cat.jpg"
control_image = load_image_from_pil(Image.open(pathlib.Path(img_path)))


# generate image
generated = pipe(prompt, image=control_image, controlnet_conditioning_scale=0.3, guidance_scale=7.5, num_inference_steps=30).images[0]
generated.save("food_result.png")