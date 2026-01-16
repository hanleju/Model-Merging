"""
GQA (Visual Reasoning) Fine-tuning for PaliGemma
Structured visual reasoning with 4-bit quantization and gradient checkpointing
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


# =========================================================
# Data Processing
# =========================================================

def load_gqa_data(gqa_root: str, split: str = "val"):
    """Load GQA data from local files."""
    
    # Load questions JSON
    questions_file = os.path.join(gqa_root, "questions1.2", f"{split}_all_questions.json")
    image_dir = os.path.join(gqa_root, "images")
    
    print(f"📂 Loading GQA from: {questions_file}")
    print(f"📂 Image directory: {image_dir}")
    
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    
    # Load questions JSON
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # Parse dataset
    # GQA format: {"question_id": {"imageId": ..., "question": ..., "answer": ..., ...}, ...}
    dataset = []
    
    for question_id, item in questions_data.items():
        # Get image ID
        image_id = item.get('imageId', item.get('image_id', question_id))
        
        # Construct image filename (GQA images are typically named as imageId.jpg)
        image_name = f"{image_id}.jpg"
        image_path = os.path.join(image_dir, image_name)
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            continue
        
        # Get question and answer
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # Skip if no question or answer
        if not question or not answer:
            continue
        
        dataset.append({
            'image_path': image_path,
            'question': question,
            'answer': str(answer),
            'question_id': question_id,
            'image_id': image_id
        })
    
    print(f"✅ Loaded {len(dataset)} QA pairs")
    return dataset


class GQADataset(torch.utils.data.Dataset):
    """GQA Dataset for PaliGemma - Visual Reasoning."""
    
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
        
        # Get question and answer
        question = item['question']
        answer = item['answer']
        
        # GQA prompt format - reasoning focused
        prompt = f"<image>Question: {question}\nAnswer:"
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=196
        )
        
        # Process labels (answer - usually short for GQA)
        labels = self.processor.tokenizer(
            str(answer),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32
        ).input_ids
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = labels.squeeze(0)
        
        # Replace padding token id with -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        inputs['labels'] = labels
        
        return inputs


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
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
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    if use_lora:
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
    
    processor = PaliGemmaProcessor.from_pretrained(model_path)
    
    print("✅ Model setup complete!\n")
    return model, processor


# =========================================================
# Training
# =========================================================

def train(
    model_path: str,
    gqa_root: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    train_split_ratio: float = 0.8,
    max_train_samples: int = None,
    max_eval_samples: int = None,
    use_wandb: bool = False,
    wandb_project: str = "paligemma-gqa",
    device: str = "cuda"
):
    """Fine-tune PaliGemma on GQA."""
    
    print("🎯 GQA Visual Reasoning Fine-tuning")
    print("="*60)
    
    # Setup model
    model, processor = setup_model_and_processor(model_path, device)
    
    # Load GQA dataset from local files
    print(f"📂 Loading GQA dataset from: {gqa_root}")
    full_dataset = load_gqa_data(gqa_root, split="val")
    
    # Split into train/eval (8:2 ratio by default)
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
    train_ds = GQADataset(train_data, processor, split="train")
    eval_ds = GQADataset(eval_data, processor, split="validation")
    
    # Compute metrics function for GQA
    def compute_metrics(eval_pred):
        """Compute accuracy for GQA visual reasoning."""
        predictions, labels = eval_pred
        
        # Replace -100 with pad token id for decoding
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        
        # Decode predictions and labels
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        
        # Compute exact match accuracy
        correct = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            pred = pred.strip().lower()
            label = label.strip().lower()
            if pred == label:
                correct += 1
        
        accuracy = correct / len(decoded_preds) if len(decoded_preds) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "num_samples": len(decoded_preds)
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
        run_name=f"paligemma-gqa-{num_epochs}ep" if use_wandb else None,
    )
    
    if use_wandb:
        wandb.init(project=wandb_project, name=f"gqa-reasoning-{num_epochs}ep")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
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
    parser = argparse.ArgumentParser(description="Fine-tune PaliGemma on GQA")
    
    parser.add_argument("--model_path", type=str, default="google/paligemma-3b-pt-448",
                       help="Base model path")
    parser.add_argument("--gqa_root", type=str, required=True,
                       help="GQA dataset root directory (e.g., D:/GQA/)")
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
                       help="Max training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                       help="Max evaluation samples")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="paligemma-gqa",
                       help="W&B project name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    train(
        model_path=args.model_path,
        gqa_root=args.gqa_root,
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
