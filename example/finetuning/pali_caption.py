"""
Pretraining script for paligemma-3b-pt-224 on COCO Captioning dataset.
"""
import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig


MODEL_NAME = "google/paligemma-3b-pt-224"
BATCH_SIZE = 2
EPOCHS = 5
MAX_LENGTH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "D:/datasets/coco/"
IMAGES_DIR = os.path.join(DATA_ROOT, "train2017")
ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_train2017.json")

class CocoCaptionDataset(Dataset):
    def __init__(self, annotation_file, images_dir, transform=None):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        self.samples = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']
            if img_id in self.img_id_to_filename:
                file_name = self.img_id_to_filename[img_id]
                full_path = os.path.join(images_dir, file_name)
                if os.path.exists(full_path):
                    self.samples.append({
                        "image_path": full_path,
                        "caption": caption
                    })
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # PaliGemma input prompt
        input_text = "<image>Describe this image."
        return {"image": image, "input_text": input_text, "caption": item["caption"]}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

coco = CocoCaptionDataset(ANNOTATION_FILE, IMAGES_DIR, transform=transform)
processor = PaliGemmaProcessor.from_pretrained(MODEL_NAME, use_fast=True)

DTYPE = torch.bfloat16

def collate_fn(batch):
    texts = [x["input_text"] for x in batch]
    labels = [x["caption"] for x in batch]
    images = [x["image"] for x in batch]
    tokens = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest")
    if "pixel_values" in tokens:
        tokens["pixel_values"] = tokens["pixel_values"].to(DTYPE).to(DEVICE)
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in tokens.items()}

# 3. Model & LoRA
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)

for param in model.model.vision_tower.parameters():
    param.requires_grad = False
for param in model.model.multi_modal_projector.parameters():
    param.requires_grad = False
    
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Training arguments
training_args = TrainingArguments(
    num_train_epochs=EPOCHS,
    remove_unused_columns=False,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    warmup_steps=200,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_torch_fused",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    output_dir="./models/pali_caption_lora_finetuned",
    # bf16=True,
    dataloader_pin_memory=False
)

# 5. Trainer
trainer = Trainer(
    model=model,
    train_dataset=coco,
    data_collator=collate_fn,
    args=training_args
)

if __name__ == "__main__":
    trainer.train()
