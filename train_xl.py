import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

from diffusers import UNet2DConditionModel, ControlNetModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# SDXL-compatible 모델 경로로 전환
CONTROLNET_REPO = "diffusers/controlnet-canny-sdxl-1.0"
UNET_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"

# 하이퍼파라미터
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
LR = 1e-5
EPOCHS = 10
SAVE_DIR = "./lora_food_likeness"

# 이미지 전처리 함수 (SDXL 입력 크기 고려)
preprocess = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Custom Dataset
class FoodFaceDataset(Dataset):
    def __init__(self, canny_dir, target_dir):
        self.canny_paths = sorted([os.path.join(canny_dir, f) for f in os.listdir(canny_dir)])
        self.target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])
        self.transform = preprocess

    def __len__(self):
        return len(self.canny_paths)

    def __getitem__(self, idx):
        canny_img = Image.open(self.canny_paths[idx]).convert("RGB")
        target_img = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(canny_img), self.transform(target_img)

dataset = FoodFaceDataset("./train/canny", "./train/target")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 모델 로딩 (SDXL 전용)
vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=torch.float16).to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device)
controlnet = ControlNetModel.from_pretrained(CONTROLNET_REPO, torch_dtype=torch.float16).to(device)
unet = UNet2DConditionModel.from_pretrained(UNET_REPO, subfolder="unet", torch_dtype=torch.float16).to(device)

# LoRA 적용
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
unet = get_peft_model(unet, lora_config)

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
scheduler = DDPMScheduler.from_pretrained(UNET_REPO, subfolder="scheduler")

# 훈련 루프
for epoch in range(EPOCHS):
    for step, (canny_img, target_img) in enumerate(dataloader):
        canny_img, target_img = canny_img.to(device), target_img.to(device)

        # 텍스트 임베딩 (SDXL는 pooled + hidden_states)
        text_input = tokenizer(["A food image shaped like a dog face"], return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_input = {k: v.to(device) for k, v in text_input.items()}
        text_outputs = text_encoder(**text_input)
        prompt_embeds = text_outputs.last_hidden_state

        # VAE latent 생성
        latents = vae.encode(target_img * 2 - 1).latent_dist.sample()
        latents = latents * 0.13025  # SDXL scaling

        # timestep 샘플링
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (BATCH_SIZE,), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # ControlNet residuals 계산
        down_block_residuals, mid_block_residual = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=canny_img
        )

        # UNet 예측
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_block_residuals,
            mid_block_additional_residual=mid_block_residual,
        ).sample

        # 손실 계산
        loss = F.mse_loss(model_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"[Epoch {epoch}] Step {step}: loss={loss.item():.4f}")

    #checkpoint 저장
    if ((epoch % 100) ==0) :
      # Checkpoint 저장
      os.makedirs(SAVE_DIR, exist_ok=True)
      unet.save_pretrained(SAVE_DIR + f"/epoch_{epoch}")
