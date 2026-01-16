"""
COCO Captioning Fine-tuning for PaliGemma
Uses 4-bit quantization and gradient checkpointing for efficient training
"""

import torch
import os
import argparse
import json
import numpy as np
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from typing import Dict, List
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# =========================================================
# Data Processing
# =========================================================

def load_coco_annotations(coco_root: str, split: str = "val"):
    """Load COCO annotations from local files."""
    annotation_file = os.path.join(coco_root, "annotations", f"captions_{split}2017.json")
    image_dir = os.path.join(coco_root, f"{split}2017")
    
    print(f"📂 Loading annotations from: {annotation_file}")
    print(f"📂 Image directory: {image_dir}")
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to filename mapping
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Group captions by image_id
    image_captions = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(caption)
    
    # Create dataset list
    dataset = []
    for image_id, captions in image_captions.items():
        if image_id in images:
            image_path = os.path.join(image_dir, images[image_id])
            if os.path.exists(image_path):
                dataset.append({
                    'image_path': image_path,
                    'captions': captions,
                    'image_id': image_id
                })
    
    print(f"✅ Loaded {len(dataset)} images with captions")
    return dataset


class COCOCaptioningDataset(torch.utils.data.Dataset):
    """COCO Captioning Dataset for PaliGemma."""
    
    def __init__(self, data_list, processor, split="train"):
        self.data = data_list
        self.processor = processor
        self.split = split
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading image {item['image_path']}: {e}")
            # Return a dummy sample
            image = Image.new('RGB', (448, 448), color='white')
        
        # Get caption (use first caption)
        caption = item['captions'][0]
        
        # Format prompt for captioning
        prompt = "<image>Describe this image in detail."
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        # Process labels (caption)
        labels = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).input_ids
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = labels.squeeze(0)
        
        # Replace padding token id with -100 for loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        inputs['labels'] = labels
        
        return inputs


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    # Stack all tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'pixel_values': pixel_values,
        'attention_mask': attention_mask,
        'labels': labels
    }


# =========================================================
# Model Setup
# =========================================================

def setup_model_and_processor(
    model_path: str,
    device: str = "cuda",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32
):
    """Setup PaliGemma model with 4-bit quantization and LoRA."""
    
    print(f"📥 Loading model: {model_path}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0},  # Load on first GPU
        torch_dtype=torch.bfloat16,
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    if use_lora:
        # LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load processor
    processor = PaliGemmaProcessor.from_pretrained(model_path)
    
    print("✅ Model setup complete!\n")
    return model, processor


# =========================================================
# Training
# =========================================================

def train(
    model_path: str,
    coco_root: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    train_split_ratio: float = 0.8,
    max_train_samples: int = None,
    max_eval_samples: int = None,
    use_wandb: bool = False,
    wandb_project: str = "paligemma-coco",
    device: str = "cuda"
):
    """Fine-tune PaliGemma on COCO Captioning."""
    
    print("🎯 COCO Captioning Fine-tuning")
    print("="*60)
    
    # Setup model
    model, processor = setup_model_and_processor(model_path, device)
    
    # Load COCO dataset from local files
    print(f"📂 Loading COCO dataset from: {coco_root}")
    full_dataset = load_coco_annotations(coco_root, split="val")
    
    # Split into train/eval (8:2 ratio)
    import random
    random.seed(42)
    random.shuffle(full_dataset)
    
    split_idx = int(len(full_dataset) * train_split_ratio)
    train_data = full_dataset[:split_idx]
    eval_data = full_dataset[split_idx:]
    
    # Limit samples if specified
    if max_train_samples:
        train_data = train_data[:max_train_samples]
    if max_eval_samples:
        eval_data = eval_data[:max_eval_samples]
    
    print(f"   Train samples: {len(train_data)}")
    print(f"   Eval samples: {len(eval_data)}\n")
    
    # Create datasets
    train_dataset = COCOCaptioningDataset(train_data, processor, split="train")
    eval_dataset = COCOCaptioningDataset(eval_data, processor, split="validation")
    
    # Compute metrics function for COCO Captioning
    def compute_metrics(eval_pred):
        """Compute BLEU-4 score for captioning."""
        predictions, labels = eval_pred
        
        # Replace -100 with pad token id for decoding
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        
        # Decode predictions and labels
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        
        # Compute BLEU-4 score
        smooth = SmoothingFunction().method1
        bleu_scores = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_tokens = pred.strip().split()
            label_tokens = label.strip().split()
            
            if len(pred_tokens) > 0 and len(label_tokens) > 0:
                score = sentence_bleu([label_tokens], pred_tokens, 
                                     weights=(0.25, 0.25, 0.25, 0.25),
                                     smoothing_function=smooth)
                bleu_scores.append(score)
        
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        
        return {
            "bleu4": avg_bleu,
            "num_samples": len(bleu_scores)
        }
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else "none",
        run_name=f"paligemma-coco-{num_epochs}ep" if use_wandb else None,
    )
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project=wandb_project, name=f"coco-captioning-{num_epochs}ep")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("🚀 Starting training...")
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    print("\n✅ Training complete!")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PaliGemma on COCO Captioning")
    
    parser.add_argument("--model_path", type=str, default="google/paligemma-3b-pt-448",
                       help="Base model path")
    parser.add_argument("--coco_root", type=str, required=True,
                       help="COCO dataset root directory (e.g., D:/coco2017/)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--train_split_ratio", type=float, default=0.8,
                       help="Train/eval split ratio (default: 0.8 for 8:2 split)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Max training samples (for testing)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                       help="Max evaluation samples")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="paligemma-coco",
                       help="W&B project name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    train(
        model_path=args.model_path,
        coco_root=args.coco_root,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_split_ratio=args.train_split_ratio,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        device=args.device
    )


if __name__ == "__main__":
    main()
