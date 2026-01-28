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
DATA_ROOT = "D:/datasets/coco/"
IMAGES_DIR = os.path.join(DATA_ROOT, "val2017")
ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_val2017.json")
OUTPUT_DIR = "./models/smolvlm-caption-finetune_3"

# 하이퍼파라미터 설정
# SmolVLM-500M은 작지만, 이미지 처리로 인해 VRAM을 차지하므로 배치 사이즈 조절 필요
BATCH_SIZE = 4      # VRAM 8~12GB 기준
GRAD_ACCUMULATION = 8 # 실질적 배치 사이즈 = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
LORA_RANK = 64      # 표현력을 위해 64 또는 128 권장 
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# -----------------------------------------------------------------------------
# 2. 데이터 로딩 및 전처리 클래스
# -----------------------------------------------------------------------------

def load_coco_captions(annotation_file, images_dir):
    """
    COCO 주석 파일을 파싱하여 이미지 경로와 캡션 쌍을 반환합니다.
    데이터 무결성 검사를 수행하여 존재하지 않는 이미지는 제외합니다.
    """
    print(f"[Info] Loading annotations from {annotation_file}...")
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 이미지 ID -> 파일명 매핑 생성 (O(1) 조회를 위해 딕셔너리 사용)
    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    formatted_data = []
    missing_count = 0
    
    # 캡션 데이터 순회
    for ann in data['annotations']:
        img_id = ann['image_id']
        caption = ann['caption']
        
        if img_id in img_id_to_filename:
            file_name = img_id_to_filename[img_id]
            full_path = os.path.join(images_dir, file_name)
            
            # 실제 파일 존재 여부 확인 (매우 중요: 학습 중단 방지)
            if os.path.exists(full_path):
                formatted_data.append({
                    "image_path": full_path,
                    "caption": caption
                })
            else:
                missing_count += 1
    
    print(f"[Info] Found {len(formatted_data)} valid pairs. (Missing files: {missing_count})")
    return formatted_data

class SmolVLMCustomCollator:
    """
    VLM 학습을 위한 커스텀 Collator.
    이미지(PIL)와 텍스트(대화형 프롬프트)를 받아 모델이 처리 가능한 텐서로 변환합니다.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        
        for example in examples:
            # 이미지 경로에서 로드 (이 시점에만 디스크 I/O 발생)
            try:
                image = Image.open(example["image_path"]).convert("RGB")
            except Exception as e:
                print(f"Error loading image {example['image_path']}: {e}")
                continue
            
            # SmolVLM 채팅 포맷 구성 - assistant 응답 포함
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["caption"]}]
                }
            ]
            
            # Processor의 채팅 템플릿 적용 (토큰화 전 텍스트 변환)
            prompt = self.processor.apply_chat_template(messages, tokenize=False)
            texts.append(prompt)
            images.append(image)
            
        # Processor 호출: 이미지 리사이징/정규화 + 텍스트 토큰화 + 패딩
        # truncation 비활성화: 이미지 토큰이 잘리면 오류 발생
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        
        # 레이블 생성: Assistant 응답 부분만 학습하도록 마스킹
        labels = batch["input_ids"].clone()
        
        # 각 샘플에 대해 assistant 응답 부분만 유지하고 나머지는 마스킹
        for idx, (text, example) in enumerate(zip(texts, examples)):
            # User 프롬프트만 있는 버전 생성 (assistant 응답 제외)
            messages_without_assistant = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."}
                    ]
                }
            ]
            prompt_only = self.processor.apply_chat_template(
                messages_without_assistant, 
                tokenize=False,
                add_generation_prompt=True  # <|im_start|>assistant까지 포함
            )
            
            # 프롬프트 부분의 토큰 길이 계산
            prompt_tokens = self.processor.tokenizer(
                prompt_only,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0]
            
            prompt_length = len(prompt_tokens)
            
            # 프롬프트 부분은 마스킹 (-100)
            labels[idx, :prompt_length] = -100
        
        # 패딩 토큰은 손실(Loss) 계산에서 제외 (-100)
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
            # 이미지는 로드하지 않고 경로와 캡션만 저장
            # SFTTrainer가 요구하는 text 필드 추가 (더미)
            yield {
                "image_path": sample['image_path'],
                "caption": sample['caption'],
                "text": sample['caption']  # SFTTrainer용 더미 필드
            }

    return Dataset.from_generator(gen)

# -----------------------------------------------------------------------------
# 3. 메인 학습 루틴
# -----------------------------------------------------------------------------

def main():
    print(f"[Init] Starting fine-tuning for {MODEL_ID}")
    
    # 1. 프로세서 로드 (토크나이저 + 이미지 프로세서)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        size={"longest_edge": 1*256}
    )
    
    # 2. 데이터 준비
    raw_data = load_coco_captions(ANNOTATION_FILE, IMAGES_DIR) 
    
    random.seed(42)
    random.shuffle(raw_data)

    split_idx = int(len(raw_data) * 0.8)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]

    print(f"[Data] Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    train_dataset = create_dataset_generator(train_data)
    
    # 3. 모델 로드 및 양자화 설정 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # 4-bit NormalFloat
        bnb_4bit_compute_dtype=torch.bfloat16, # 연산은 BF16으로 수행 (안정성)
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16
    )
    
    # 4. LoRA 설정
    # SmolLM2(Llama 기반)의 모든 선형 레이어를 타겟팅하여 학습 효과 극대화
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "fc1", "fc2"],
        task_type=TaskType.CAUSAL_LM, 
        modules_to_save=["lm_head", "embed_tokens"] # 특수 토큰 학습을 위해 필요할 수 있음
    )
    
    # k-bit 학습을 위한 전처리 (Gradient Checkpointing 등 활성화)
    model = prepare_model_for_kbit_training(model)
    
    # 5. 학습 인자 설정 (SFTConfig)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,       # BFloat16 사용 (모델이 bfloat16으로 로드되었으므로)
        logging_steps=200,
        save_strategy="epoch",
        report_to="none", # 텐서보드 등을 사용하려면 'tensorboard'
        remove_unused_columns=False, # 커스텀 Collator 사용 시 필수
        dataset_text_field="text",       # SFTTrainer가 요구하는 텍스트 필드 지정
        packing=False,               # VLM에서는 Packing 사용 시 주의 필요
        dataloader_num_workers=4,    # Windows에서는 0으로 설정해야 멀티프로세싱 에러 방지
        gradient_checkpointing=True, # 메모리 절약을 위해 필수
        gradient_checkpointing_kwargs={"use_reentrant": False} # 호환성 확보
    )
    
    # 6. Trainer 초기화 (SFTTrainer가 peft_config를 받아 자동으로 PEFT 모델 래핑)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SmolVLMCustomCollator(processor),
        peft_config=peft_config,
    )
    
    # 7. 학습 시작
    print(" Beginning training loop...")
    trainer.train()
    
    # 8. 모델 저장
    print(f" Saving adapters to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(" Fine-tuning complete.")

if __name__ == "__main__":
    main()