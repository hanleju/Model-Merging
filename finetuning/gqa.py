"""
GQA (Visual Reasoning) Fine-tuning for Moondream2
Structured visual reasoning with 4-bit quantization and gradient checkpointing
"""

import torch
import os
import argparse
import json
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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
    """GQA Dataset for Moondream2 with caching support."""
    
    def __init__(self, data_list, model, tokenizer, split="train"):
        self.data = data_list
        self.model = model
        self.tokenizer = tokenizer
        self.split = split
        self.cached_samples = []
    
    def _preprocess_and_cache(self, cache_file):
        """Preprocess all samples and save to disk."""
        from tqdm import tqdm
        import gc  # 가비지 컬렉터 추가
        
        # 주기적인 정리를 위한 카운터
        cleanup_step = 100
        
        for idx in tqdm(range(len(self.data)), desc="Preprocessing"):
            item = self.data[idx]
            
            # Load and resize image (Moondream2 uses 378x378)
            try:
                image = Image.open(item['image_path']).convert('RGB')
                image = image.resize((378, 378), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"⚠️  Error loading image {item['image_path']}: {e}")
                image = Image.new('RGB', (378, 378), color='white')
            
            # Get question and answer
            question = item['question']
            answer = item['answer']
            
            # Moondream2 format
            prompt = f"\n\nQuestion: {question}\n\nAnswer:"
            full_text = prompt + answer
            
            # Encode image using Moondream2's encode_image
            with torch.no_grad():
                image_embeds = self.model.encode_image(image)
            
            # Tokenize text
            tokens = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256
            )
            
            # Remove batch dimension
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
            
            # Create labels
            labels = input_ids.clone()
            
            # Tokenize just prompt to find its length
            prompt_tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            )
            prompt_len = prompt_tokens['input_ids'].shape[1]
            
            # Mask prompt in labels
            labels[:prompt_len] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Store processed sample
            sample = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                # [수정 1] GPU 텐서를 CPU로 이동시켜야 함!
                'image_embeds': image_embeds.squeeze(0).cpu(),
                'labels': labels
            }
            self.cached_samples.append(sample)
            
            # [수정 2] 반복문 내에서 GPU 메모리 명시적 해제
            del image_embeds
            if idx % cleanup_step == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Save to disk
        torch.save(self.cached_samples, cache_file)
        
        # [수정 3] 캐싱 완료 후 최종 정리
        gc.collect()
        torch.cuda.empty_cache()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # If we have cached samples, return them directly
        if self.cached_samples:
            return self.cached_samples[idx]
        
        # Otherwise, process on-the-fly
        item = self.data[idx]
        
        # Load image and resize to 378x378 (Moondream2 default)
        try:
            image = Image.open(item['image_path']).convert('RGB')
            image = image.resize((378, 378), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"⚠️  Error loading image {item['image_path']}: {e}")
            image = Image.new('RGB', (378, 378), color='white')
        
        # Get question and answer
        question = item['question']
        answer = item['answer']
        
        # Moondream2 format
        prompt = f"\n\nQuestion: {question}\n\nAnswer:"
        full_text = prompt + answer
        
        # Encode image
        with torch.no_grad():
            image_embeds = self.model.encode_image(image)
        
        # Tokenize text
        tokens = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        
        # Create labels
        labels = input_ids.clone()
        
        # Tokenize prompt to find its length
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )
        prompt_len = prompt_tokens['input_ids'].shape[1]
        
        # Mask prompt in labels
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # CPU로 이동하여 GPU 메모리 누수 방지
            'image_embeds': image_embeds.squeeze(0).cpu(),
            'labels': labels
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader (Moondream2)."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    image_embeds = torch.stack([item['image_embeds'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image_embeds': image_embeds,
        'labels': labels
    }


# =========================================================
# Model Setup
# =========================================================

def setup_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32
):
    """Setup Moondream2 model with 4-bit quantization and LoRA."""
    
    print(f"📥 Loading Moondream2: {model_path}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load Moondream2 model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision="2024-08-26"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision="2024-08-26"
    )
    
    # Set pad_token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    
    if use_lora:
        # Moondream2 (Phi-based) LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["Wqkv", "out_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    print("✅ Model setup complete!\n")
    return model, tokenizer


# =========================================================
# Custom Trainer
# =========================================================

class Moondream2Trainer(Trainer):
    """Custom Trainer for Moondream2 that handles image embeddings."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation that passes image_embeds to model."""
        image_embeds = inputs.pop('image_embeds')
        labels = inputs.pop('labels')
        
        # Forward pass with image embeddings
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels,
            image_embeds=image_embeds
        )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


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
    wandb_project: str = "moondream-gqa",
    device: str = "cuda"
):
    """Fine-tune Moondream2 on GQA."""
    
    print("🎯 GQA Visual Reasoning Fine-tuning (Moondream2)")
    print("="*60)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(model_path, device)
    
    # Load GQA dataset from local files
    print(f"📂 Loading GQA dataset from: {gqa_root}")
    full_dataset = load_gqa_data(gqa_root, split="val")
    
    # Split into train/eval
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
    
    # Create datasets with caching support
    train_dataset = GQADataset(train_data, model, tokenizer, split="train")
    eval_dataset = GQADataset(eval_data, model, tokenizer, split="validation")
    
    # Compute metrics function for GQA
    def compute_metrics(eval_pred):
        """Compute accuracy for GQA visual reasoning."""
        predictions, labels = eval_pred
        
        # Replace -100 with pad token id for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
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
        report_to="wandb" if use_wandb else "none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
    )
    
    # Initialize W&B if requested
    if use_wandb:
        wandb.init(project=wandb_project, name=f"gqa-moondream2-{num_epochs}ep")
    
    # Initialize Trainer
    trainer = Moondream2Trainer(
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
    print(f"💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    print("✅ Training complete!")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Moondream2 on GQA")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="vikhyatk/moondream2",
                        help="Path to Moondream2 model")
    
    # Data arguments
    parser.add_argument("--gqa_root", type=str, required=True,
                        help="Path to GQA dataset root directory")
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/gqa_moondream2",
                        help="Output directory for model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--train_split_ratio", type=float, default=0.8,
                        help="Ratio of data to use for training")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Maximum number of evaluation samples")
    
    # W&B arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="moondream-gqa",
                        help="W&B project name")
    
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
    )
