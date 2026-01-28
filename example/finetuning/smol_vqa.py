import os
import json
import torch
import random
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datasets import Dataset, Features, Value, Image as DatasetImage
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer, SFTConfig

# -----------------------------------------------------------------------------
# 1. 설정 및 상수 정의
# -----------------------------------------------------------------------------
# 사용자의 요청에 따른 경로 설정
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DATA_ROOT = "D:/VQA/cocoqa/test"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
QUESTIONS_JSON = os.path.join(DATA_ROOT, "questions.json")
OUTPUT_DIR = "./models/smolvlm-vqa-finetune"

# 하이퍼파라미터 설정
BATCH_SIZE = 4      # VRAM 8~12GB 기준
GRAD_ACCUMULATION = 8 # 실질적 배치 사이즈 = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_RANK = 64      # 표현력을 위해 64 또는 128 권장 
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# -----------------------------------------------------------------------------
# 2. 데이터 로딩 및 전처리 클래스
# -----------------------------------------------------------------------------

def load_vqa_data(questions_json, images_dir):
    """
    VQA 데이터를 로드합니다.
    questions.json 형식: [["question", "answer", image_id], ...]
    """
    print(f"[Info] Loading questions from {questions_json}...")
    if not os.path.exists(questions_json):
        raise FileNotFoundError(f"Questions file not found: {questions_json}")

    with open(questions_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    missing_count = 0
    
    # 각 항목은 [question, answer, image_id] 형식
    for item in data:
        if len(item) < 3:
            print(f"[Warning] Invalid item format: {item}")
            continue
            
        question = item[0]
        answer = item[1]
        image_id = item[2]
        
        # 이미지 파일명 생성 (COCO 형식 가정: 000000{image_id}.jpg)
        # 필요에 따라 파일명 형식을 조정하세요
        file_name = f"{image_id:012d}.jpg"  # 12자리 0 패딩
        full_path = os.path.join(images_dir, file_name)
        
        # 실제 파일 존재 여부 확인
        if os.path.exists(full_path):
            formatted_data.append({
                "image_path": full_path,
                "question": question,
                "answer": answer,
                "image_id": image_id
            })
        else:
            # 다른 파일 확장자 시도 (.png 등)
            alt_path = os.path.join(images_dir, f"{image_id}.jpg")
            if os.path.exists(alt_path):
                formatted_data.append({
                    "image_path": alt_path,
                    "question": question,
                    "answer": answer,
                    "image_id": image_id
                })
            else:
                missing_count += 1
                if missing_count <= 5:  # 처음 5개만 출력
                    print(f"[Warning] Missing image: {full_path}")
    
    print(f"[Info] Found {len(formatted_data)} valid QA pairs. (Missing files: {missing_count})")
    return formatted_data

class VQACustomCollator:
    """
    VQA 학습을 위한 커스텀 Collator.
    이미지(PIL)와 질문/답변을 받아 모델이 처리 가능한 텐서로 변환합니다.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        
        for example in examples:
            # 이미지 경로에서 로드
            try:
                image = Image.open(example["image_path"]).convert("RGB")
            except Exception as e:
                print(f"Error loading image {example['image_path']}: {e}")
                continue
            
            # VQA 형식의 대화 구성
            # User: 이미지 + 질문 -> Assistant: 답변
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["question"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example["answer"]}
                    ]
                }
            ]
            
            # Processor의 채팅 템플릿 적용
            prompt = self.processor.apply_chat_template(messages, tokenize=False)
            texts.append(prompt)
            images.append(image)
            
        # Processor 호출: 이미지 처리 + 텍스트 토큰화
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        
        # 레이블 생성: Causal Language Modeling
        labels = batch["input_ids"].clone()
        # 패딩 토큰은 손실 계산에서 제외
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

def create_dataset_generator(samples):
    """
    메모리 효율성을 위해 이미지 경로만 저장.
    실제 이미지 로딩은 Collator에서 배치 생성 시 수행.
    """
    def gen():
        for sample in samples:
            # SFTTrainer가 요구하는 text 필드 추가
            yield {
                "image_path": sample['image_path'],
                "question": sample['question'],
                "answer": sample['answer'],
                "image_id": sample['image_id'],
                "text": f"{sample['question']} {sample['answer']}"  # SFTTrainer용 더미 필드
            }

    return Dataset.from_generator(gen)

# -----------------------------------------------------------------------------
# 3. 메인 학습 루틴
# -----------------------------------------------------------------------------

def main():
    print(f"[Init] Starting VQA fine-tuning for {MODEL_ID}")
    
    # 1. 프로세서 로드 (토크나이저 + 이미지 프로세서)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        size={"longest_edge": 1*256}
    )
    
    # 2. VQA 데이터 준비
    raw_data = load_vqa_data(QUESTIONS_JSON, IMAGES_DIR) 
    
    if len(raw_data) == 0:
        raise ValueError("No valid data found. Please check your data paths and file formats.")
    
    # 데이터 셔플 및 분할
    random.seed(42)
    random.shuffle(raw_data)

    split_idx = int(len(raw_data) * 0.8)  # 80% train, 20% val
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]

    print(f"[Data] Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    train_dataset = create_dataset_generator(train_data)
    val_dataset = create_dataset_generator(val_data)
    
    # 3. 모델 로드 및 양자화 설정 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # 4-bit NormalFloat
        bnb_4bit_compute_dtype=torch.bfloat16, # 연산은 BF16으로 수행
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16
    )
    
    # 4. LoRA 설정
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM, 
        modules_to_save=["lm_head", "embed_tokens"]
    )
    
    # k-bit 학습을 위한 전처리
    model = prepare_model_for_kbit_training(model)
    
    # 5. 학습 인자 설정 (SFTConfig)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,
        logging_steps=150,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="text",
        packing=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    # 6. Trainer 초기화
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VQACustomCollator(processor),
        peft_config=peft_config,
    )
    
    # 7. 학습 시작
    print("[Training] Beginning training loop...")
    trainer.train()
    
    # 8. 모델 저장
    print(f"[Saving] Saving adapters to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("[Complete] VQA fine-tuning complete.")

if __name__ == "__main__":
    main()
