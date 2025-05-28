import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

from diffusers import UNet2DConditionModel, ControlNetModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import get_peft_model, LoraConfig, TaskType

#설정
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
LR = 1e-5
EPOCHS = 1000
SAVE_DIR = "./lora_food_likeness"

#전처리 루틴
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

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

#canny에 전처리된 이미지들 넣고, target에 음식사진 넣고
dataset = FoodFaceDataset("./train/canny", "./train/target")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#diffusion = vae + tokenizer + text_encoder + controlnet + unet
#lora config 는 unet에 적용 (가장 파라미터가 많음)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
controlnet = ControlNetModel.from_pretrained("stabilityai/stable-diffusion-3.5-large-controlnet-canny").to(device)

unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-3.5", subfolder="unet").to(device)

#로라 컨피그
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
unet = get_peft_model(unet, lora_config)


optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-3.5", subfolder="scheduler")

#훈련루프
for epoch in range(EPOCHS):
    for step, (canny_img, target_img) in enumerate(dataloader):
        canny_img, target_img = canny_img.to(device), target_img.to(device)

        # 텍스트 임베딩
        text_input = tokenizer(["animal image to food image"], return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_embeds = text_encoder(text_input.input_ids.to(device))[0]

        # VAE latent
        latents = vae.encode(target_img * 2 - 1).latent_dist.sample()
        latents = latents * 0.18215  # SD scaling

        # timestep 샘플링
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (BATCH_SIZE,), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # ControlNet output (conditioning)
        down_block_residuals, mid_block_residual = controlnet(
            noisy_latents, timesteps, encoder_hidden_states=text_embeds, controlnet_cond=canny_img
        )

        # UNet 예측
        model_pred = unet(
            noisy_latents, timesteps,
            encoder_hidden_states=text_embeds,
            down_block_additional_residuals=down_block_residuals,
            mid_block_additional_residual=mid_block_residual,
        ).sample

        loss = F.mse_loss(model_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"[Epoch {epoch}] Step {step}: loss={loss.item():.4f}")

    #checkpoint 저장
    if ((epoch % 100) ==0) :
        os.makedirs(SAVE_DIR, exist_ok=True)
        unet.save_pretrained(SAVE_DIR + f"/epoch_{epoch}")
