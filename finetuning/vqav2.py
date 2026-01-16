"""
VQAv2 Fine-tuning for PaliGemma
Visual Question Answering with 4-bit quantization and gradient checkpointing
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

def load_cocoqa_data(vqa_root: str, split: str = "test"):
    """Load COCO-QA data from local files."""
    
    split_dir = os.path.join(vqa_root, split)
    questions_file = os.path.join(split_dir, "questions.json")
    image_dir = os.path.join(split_dir, "images")
    
    print(f"📂 Loading COCO-QA from: {questions_file}")
    print(f"📂 Image directory: {image_dir}")
    
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    
    # Load questions.json
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # Load labels if available
    labels_file = os.path.join(vqa_root, "labels.txt")
    answers_dict = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                answers_dict[idx] = line.strip()
    
    # Parse dataset
    dataset = []
    
    # Handle different JSON structures
    if isinstance(questions_data, dict):
        if 'questions' in questions_data:
            questions_list = questions_data['questions']
        elif 'data' in questions_data:
            questions_list = questions_data['data']
        else:
            # Assume it's a dict of question_id -> data
            questions_list = list(questions_data.values())
    elif isinstance(questions_data, list):
        questions_list = questions_data
    else:
        raise ValueError(f"Unexpected JSON structure in {questions_file}")
    
    for idx, item in enumerate(questions_list):
        # Get question
        if isinstance(item, dict):
            question = item.get('question', item.get('text', ''))
            image_id = item.get('image_id', item.get('img_id', idx))
            
            # Get answer
            if 'answer' in item:
                answer = item['answer']
            elif 'answers' in item:
                answers = item['answers']
                if isinstance(answers, list) and len(answers) > 0:
                    # Use most common answer or first
                    if isinstance(answers[0], dict):
                        answer_texts = [a.get('answer', a.get('text', '')) for a in answers]
                        answer = max(set(answer_texts), key=answer_texts.count) if answer_texts else ""
                    else:
                        answer = str(answers[0])
                elif isinstance(answers, dict):
                    answer = answers.get('answer', answers.get('text', ''))
                else:
                    answer = str(answers)
            elif idx in answers_dict:
                answer = answers_dict[idx]
            else:
                answer = ""
            
            # Get image filename
            if 'image' in item:
                image_name = item['image']
            elif 'image_id' in item:
                image_name = f"{item['image_id']}.jpg"
            else:
                image_name = f"{image_id}.jpg"
        else:
            # Simple format
            question = str(item)
            image_name = f"{idx}.jpg"
            answer = answers_dict.get(idx, "")
        
        # Build image path
        image_path = os.path.join(image_dir, image_name)
        
        # Try different extensions if jpg doesn't exist
        if not os.path.exists(image_path):
            base_name = os.path.splitext(image_name)[0]
            for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                test_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(test_path):
                    image_path = test_path
                    break
        
        # Skip if image still doesn't exist
        if not os.path.exists(image_path):
            continue
        
        dataset.append({
            'image_path': image_path,
            'question': question,
            'answer': answer,
            'question_id': idx
        })
    
    print(f"✅ Loaded {len(dataset)} QA pairs")
    return dataset


class VQAv2Dataset(torch.utils.data.Dataset):
    """VQAv2 Dataset for PaliGemma."""
    
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
        
        # Format prompt for VQA
        prompt = f"<image>Question: {question}\nAnswer:"
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        # Process labels (answer - VQA answers are usually short)
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
    vqa_root: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    train_split_ratio: float = 0.8,
    max_train_samples: int = None,
    max_eval_samples: int = None,
    use_wandb: bool = False,
    wandb_project: str = "paligemma-vqav2",
    device: str = "cuda"
):
    """Fine-tune PaliGemma on VQAv2."""
    
    print("🎯 VQAv2 Fine-tuning")
    print("="*60)
    
    # Setup model
    model, processor = setup_model_and_processor(model_path, device)
    
    # Load VQAv2 dataset from local files
    print(f"📂 Loading VQAv2 dataset from: {vqa_root}")
    full_dataset = load_cocoqa_data(vqa_root, split="test")
    
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
    train_ds = VQAv2Dataset(train_data, processor, split="train")
    eval_ds = VQAv2Dataset(eval_data, processor, split="validation")
    
    # Compute metrics function for VQAv2
    def compute_metrics(eval_pred):
        """Compute accuracy for VQAv2."""
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
        run_name=f"paligemma-vqav2-{num_epochs}ep" if use_wandb else None,
    )
    
    if use_wandb:
        wandb.init(project=wandb_project, name=f"vqav2-{num_epochs}ep")
    
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
    parser = argparse.ArgumentParser(description="Fine-tune PaliGemma on VQAv2")
    
    parser.add_argument("--model_path", type=str, default="google/paligemma-3b-pt-448",
                       help="Base model path")
    parser.add_argument("--vqa_root", type=str, required=True,
                       help="VQA dataset root directory (e.g., D:/VQA/cocoqa/)")
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
    parser.add_argument("--wandb_project", type=str, default="paligemma-vqav2",
                       help="W&B project name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    train(
        model_path=args.model_path,
        vqa_root=args.vqa_root,
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
