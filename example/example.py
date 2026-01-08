import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

# =========================================================
# 1. 병합된 모델 로드
# =========================================================
# 병합된 모델 사용:
# model_path = "./weights/paligemma_race_age"

# Base Model (Pretrained) 테스트:
# model_path = "google/paligemma-3b-pt-224" 

# Model A (Race) 테스트:
# model_path = "NYUAD-ComNets/FaceScanPaliGemma_Race"

# Model B (Age) 테스트:
model_path = "NYUAD-ComNets/FaceScanPaliGemma_Age"

print(f"Loading model: {model_path}...")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16
)
processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224", use_fast=True)

# GPU 사용 가능하면 GPU로, 아니면 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"Model loaded on {device}")

# =========================================================
# 2. 이미지 로드
# =========================================================
image = Image.open("./img/me.jpg").convert("RGB")

print(f"Image loaded: {image.size}")

# =========================================================
# 3. 추론 - Task A (Race)
# =========================================================
print("\n" + "="*50)
print("Task A: Race Classification")
print("="*50)

prompt_race = "<image>what is the race of the person in the image?"
inputs_race = processor(text=prompt_race, images=image, return_tensors="pt").to(device)
input_len = inputs_race["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs_race, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    answer_race = processor.decode(generation, skip_special_tokens=True)
    
print(f"Question: {prompt_race}")
print(f"Answer: {answer_race}")

# =========================================================
# 4. 추론 - Task B (Age)
# =========================================================
print("\n" + "="*50)
print("Task B: Age Group Classification")
print("="*50)

prompt_age = "<image>what is the age group of the person in the image?"
inputs_age = processor(text=prompt_age, images=image, return_tensors="pt").to(device)
input_len = inputs_age["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs_age, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    answer_age = processor.decode(generation, skip_special_tokens=True)
    
print(f"Question: {prompt_age}")
print(f"Answer: {answer_age}")

# =========================================================
# 5. 추가 예시 - 커스텀 질문
# =========================================================
print("\n" + "="*50)
print("Custom Question")
print("="*50)

# 병합된 모델의 능력을 테스트
custom_prompt = "<image>describe the person in the image"
inputs_custom = processor(text=custom_prompt, images=image, return_tensors="pt").to(device)
input_len = inputs_custom["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs_custom, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    answer_custom = processor.decode(generation, skip_special_tokens=True)
    
print(f"Question: {custom_prompt}")
print(f"Answer: {answer_custom}")

print("\n✅ Inference Complete!")
