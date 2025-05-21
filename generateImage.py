from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from transformers import AutoTokenizer
import torch
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

from huggingface_hub import login
login("token")  # 토큰 직접 입력


controlnet = ControlNetModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

def detect_and_crop_face_region(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=True)

    results = face_mesh.process(img_rgb)
    h, w, _ = img.shape

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        xs = [int(l.x * w) for l in landmarks]
        ys = [int(l.y * h) for l in landmarks]
        x_min, x_max = max(0, min(xs)), min(w, max(xs))
        y_min, y_max = max(0, min(ys)), min(h, max(ys))
        cropped = img[y_min:y_max, x_min:x_max]
        return cropped
    return img  # fallback

def generate_canny_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    return canny_rgb

prompt = "food"

img_path = "./train/canny/0.jpg"
face_crop = detect_and_crop_face_region(img_path)
canny_map = generate_canny_map(face_crop)
control_image = Image.fromarray(canny_map)

generated = pipe(prompt, image=control_image, num_inference_steps=30).images[0]
generated.save("food_result.png")
