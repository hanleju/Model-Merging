"""
VQAv2 Fine-tuning for Moondream2
Visual Question Answering with 4-bit quantization and gradient checkpointing
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
    """VQAv2 Dataset for Moondream2 with caching support."""
    
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
    vqa_root: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    train_split_ratio: float = 0.8,
    max_train_samples: int = None,
    max_eval_samples: int = None,
    use_wandb: bool = False,
    wandb_project: str = "moondream-vqav2",
    device: str = "cuda"
):
    """Fine-tune Moondream2 on VQAv2."""
    
    print("🎯 VQAv2 Fine-tuning (Moondream2)")
    print("="*60)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(model_path, device)
    
    # Load VQAv2 dataset from local files
    print(f"📂 Loading VQAv2 dataset from: {vqa_root}")
    full_dataset = load_cocoqa_data(vqa_root, split="test")
    
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
    train_dataset = VQAv2Dataset(train_data, model, tokenizer, split="train")
    eval_dataset = VQAv2Dataset(eval_data, model, tokenizer, split="validation")
    
    # Compute metrics function for VQAv2
    def compute_metrics(eval_pred):
        """Compute accuracy for VQA."""
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
        wandb.init(project=wandb_project, name=f"vqav2-moondream2-{num_epochs}ep")
    
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
    parser = argparse.ArgumentParser(description="Fine-tune Moondream2 on VQAv2")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="vikhyatk/moondream2",
                        help="Path to Moondream2 model")
    
    # Data arguments
    parser.add_argument("--vqa_root", type=str, required=True,
                        help="Path to VQAv2 dataset root directory")
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/vqav2_moondream2",
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
    parser.add_argument("--wandb_project", type=str, default="moondream-vqav2",
                        help="W&B project name")
    
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
    )
