from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", trust_remote_code=True)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 로드
def load_pil_image(image_path):
    return Image.open(image_path).convert("RGB")

# 이미지 설명 생성
def describe_image(image: Image.Image):
    prompt = "What shapes and colors are visible in this image except the fact that it is an animal?"

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    generated_ids =model.generate(
    **inputs,
    max_new_tokens=50,
    repetition_penalty=1.2,    
    temperature=0.7,           
    top_k=50,                  
    top_p=0.95,                
    do_sample=True              
)
    caption = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

# 음식 프롬프트 생성
def generate_food_prompt_from_caption(caption: str):
    return f"Generate as if it were a food item that resembles: {caption}."

# 실행
if __name__ == "__main__":
    image_path = "./train/animal/cat.jpg"
    image = load_pil_image(image_path)

    caption = describe_image(image)
    print("이미지 설명:", caption)

    food_prompt = generate_food_prompt_from_caption(caption)
    print("생성 프롬프트:", food_prompt)
