import kagglehub
import os
import shutil

# Download latest version
foodPath = kagglehub.dataset_download("trolukovich/food11-image-dataset")
# Download latest version
animalPath = kagglehub.dataset_download("andrewmvd/animal-faces")

print("Path to food dataset files:", foodPath)
print("Path to animal dataset files:", animalPath)

target_food_dir = "/Users/ihyeonseo/Desktop/2025/GenAI/train/canny"
target_animal_dir = "/Users/ihyeonseo/Desktop/2025/GenAI/train/target"
os.makedirs(target_food_dir, exist_ok=True)
os.makedirs(target_animal_dir, exist_ok=True)

for root, _, files in os.walk(foodPath):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(root, file)
            dst = os.path.join(target_food_dir, file)
            shutil.copy2(src, dst)

for root, _, files in os.walk(animalPath):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(root, file)
            dst = os.path.join(target_animal_dir, file)
            shutil.copy2(src, dst)