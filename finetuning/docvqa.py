"""
DocVQA Fine-tuning for Moondream2
Document Visual Question Answering with 4-bit quantization and gradient checkpointing
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
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    # Fallback implementation
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


# =========================================================
# Data Processing
# =========================================================

def load_docvqa_data(docvqa_root: str, split: str = "val"):
    """Load DocVQA data from local files."""
    
    # Load QA annotations
    qa_file = os.path.join(docvqa_root, "spdocvqa_qas", f"{split}_v1.0_withQT.json")
    image_dir = os.path.join(docvqa_root, "spdocvqa_images")
    
    print(f"📂 Loading DocVQA from: {qa_file}")
    print(f"📂 Image directory: {image_dir}")
    
    if not os.path.exists(qa_file):
        raise FileNotFoundError(f"QA file not found: {qa_file}")
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # Parse data structure
    dataset = []
    
    # DocVQA JSON structure: {"data": [{"image": "documents/xxx.png", "question": ..., "answers": [...], ...}, ...]}
    if isinstance(qa_data, dict) and 'data' in qa_data:
        data_list = qa_data['data']
    elif isinstance(qa_data, list):
        data_list = qa_data
    else:
        raise ValueError(f"Unexpected JSON structure in {qa_file}")
    
    print(f"📊 Found {len(data_list)} items in JSON")
    
    skipped_no_image = 0
    skipped_no_question = 0
    
    for idx, item in enumerate(data_list):
        # Get image filename - the path is like "documents/pybv0228_81.png"
        if 'image' in item:
            image_name = item['image']
            # Extract just the filename from "documents/xxx.png" format
            if '/' in image_name:
                image_name = image_name.split('/')[-1]  # Get "pybv0228_81.png"
        elif 'image_id' in item:
            image_name = item['image_id']
        elif 'ucsf_document_id' in item:
            image_name = item['ucsf_document_id'] + '.png'
        else:
            skipped_no_image += 1
            continue
        
        # Ensure .png extension
        if not image_name.endswith('.png'):
            image_name = image_name + '.png'
        
        image_path = os.path.join(image_dir, image_name)
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            skipped_no_image += 1
            continue
        
        # Get question and answers
        question = item.get('question', '')
        if not question:
            skipped_no_question += 1
            continue
            
        answers = item.get('answers', [])
        
        # Handle different answer formats
        if isinstance(answers, list) and len(answers) > 0:
            # Use first answer
            answer = answers[0] if isinstance(answers[0], str) else str(answers[0])
        elif isinstance(answers, dict):
            # Sometimes answers is {"text": [...], ...}
            answer_list = answers.get('text', answers.get('answer', []))
            answer = answer_list[0] if answer_list else ""
        elif isinstance(answers, str):
            answer = answers
        else:
            answer = str(answers) if answers else ""
        
        dataset.append({
            'image_path': image_path,
            'question': question,
            'answer': answer,
            'question_id': item.get('question_id', item.get('questionId', idx))
        })
    
    print(f"✅ Loaded {len(dataset)} QA pairs")
    if skipped_no_image > 0:
        print(f"⚠️  Skipped {skipped_no_image} items (missing images)")
    if skipped_no_question > 0:
        print(f"⚠️  Skipped {skipped_no_question} items (missing questions)")
    
    return dataset


class DocVQADataset(torch.utils.data.Dataset):
    """DocVQA Dataset for Moondream2 with caching support."""
    
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
            
            # Moondream2 format: "\n\nQuestion: {question}\n\nAnswer:"
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
                max_length=512
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
            
            # [수정 2] 반복문 내에서 GPU 메모리 명시적 해제 (선택사항이지만 권장)
            del image_embeds
            if idx % cleanup_step == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Save to disk
        torch.save(self.cached_samples, cache_file)
        
        # [수정 3] 캐싱 완료 후 모델이 생성한 임시 찌꺼기 최종 정리
        gc.collect()
        torch.cuda.empty_cache()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # If we have cached samples, return them directly (very fast!)
        if self.cached_samples:
            return self.cached_samples[idx]
        
        # Otherwise, process on-the-fly (slower)
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
            max_length=512
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
    
    # Set pad_token if not present (Moondream2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare model for training (Moondream2 doesn't support gradient checkpointing)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    
    if use_lora:
        # Moondream2 (Phi-based) LoRA configuration
        # Phi models use different module names: Wqkv (combined q,k,v) and out_proj
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["Wqkv", "out_proj"],  # Phi model specific modules
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    print("✅ Model setup complete!\n")
    return model, tokenizer


# =========================================================
# Training
# =========================================================

def train(
    model_path: str,
    docvqa_root: str,
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    train_split_ratio: float = 0.8,
    max_train_samples: int = None,
    max_eval_samples: int = None,
    use_wandb: bool = False,
    wandb_project: str = "moondream2-docvqa",
    device: str = "cuda",
):
    """Fine-tune Moondream2 on DocVQA."""
    
    print("🎯 DocVQA Fine-tuning (Moondream2)")
    print("="*60)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(model_path, device)
    
    # Load DocVQA dataset from local files
    print(f"📂 Loading DocVQA dataset from: {docvqa_root}")
    full_dataset = load_docvqa_data(docvqa_root, split="val")
    
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
    
    # Create datasets with caching
    train_ds = DocVQADataset(train_data, model, tokenizer, split="train")
    eval_ds = DocVQADataset(eval_data, model, tokenizer, split="validation")
    
    # Compute metrics function for DocVQA
    def compute_metrics(eval_pred):
        """Compute ANLS (Average Normalized Levenshtein Similarity) for DocVQA."""
        predictions, labels = eval_pred
        
        # Replace -100 with pad token id for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ANLS
        anls_scores = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            pred = pred.strip().lower()
            label = label.strip().lower()
            
            if len(label) == 0:
                score = 1.0 if len(pred) == 0 else 0.0
            else:
                edit_dist = levenshtein_distance(pred, label)
                max_len = max(len(pred), len(label))
                score = 1.0 - (edit_dist / max_len) if max_len > 0 else 0.0
            
            anls_scores.append(score)
        
        avg_anls = np.mean(anls_scores) if anls_scores else 0.0
        
        return {
            "anls": avg_anls,
            "num_samples": len(anls_scores)
        }
    
    # Training arguments - optimized for speed
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,  # Moondream2는 더 작아서 배치 크기 늘릴 수 있음
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=max(1, batch_size // 4),
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else "none",
        run_name=f"moondream2-docvqa-{num_epochs}ep" if use_wandb else None,
    )
    
    if use_wandb:
        wandb.init(project=wandb_project, name=f"docvqa-{num_epochs}ep")
    
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
    tokenizer.save_pretrained(output_dir)
    
    print("\n✅ Training complete!")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Moondream2 on DocVQA")
    
    parser.add_argument("--model_path", type=str, default="vikhyatk/moondream2",
                       help="Base model path (default: vikhyatk/moondream2)")
    parser.add_argument("--docvqa_root", type=str, required=True,
                       help="DocVQA dataset root directory (e.g., D:/VQA/docvqa/)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Effective batch size (will use gradient accumulation)")
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
    parser.add_argument("--wandb_project", type=str, default="moondream2-docvqa",
                       help="W&B project name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    train(
        model_path=args.model_path,
        docvqa_root=args.docvqa_root,
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
