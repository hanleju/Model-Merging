# PaliGemma Fine-tuning Scripts

4개의 VL 데이터셋에 대한 PaliGemma fine-tuning 스크립트입니다.

## 🎯 지원 데이터셋

1. **COCO Captioning** - 이미지 캡셔닝
2. **DocVQA** - 문서 이미지 질의응답
3. **GQA** - 시각적 추론 및 질의응답
4. **VQAv2** - 일반 시각적 질의응답

## 📋 Requirements

```bash
pip install torch transformers datasets peft bitsandbytes accelerate pillow wandb
```

## 🚀 사용법

### 1. COCO Captioning

```bash
python finetuning/coco_captioning.py \
  --model_path google/paligemma-3b-pt-448 \
  --output_dir ./models/paligemma-coco \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --use_wandb
```

### 2. DocVQA

```bash
python finetuning/docvqa.py \
  --model_path google/paligemma-3b-pt-448 \
  --output_dir ./models/paligemma-docvqa \
  --num_epochs 5 \
  --batch_size 4 \
  --learning_rate 2e-4
```

### 3. GQA (Visual Reasoning)

```bash
python finetuning/gqa.py \
  --model_path google/paligemma-3b-pt-448 \
  --output_dir ./models/paligemma-gqa \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

### 4. VQAv2

```bash
python finetuning/vqav2.py \
  --model_path google/paligemma-3b-pt-448 \
  --output_dir ./models/paligemma-vqav2 \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

## ⚙️ 주요 기능

### 🔧 4-bit Quantization
- **BitsAndBytes** NF4 quantization 사용
- 메모리 사용량 75% 감소
- 학습 속도 유지

### 💾 Gradient Checkpointing
- 메모리 효율적 학습
- 큰 배치 사이즈 가능

### 🎯 LoRA (Low-Rank Adaptation)
- 파라미터 효율적 fine-tuning
- Rank: 16, Alpha: 32
- Target modules: attention & FFN layers

## 📊 권장 하이퍼파라미터

| Dataset | Epochs | Batch Size | Learning Rate | Memory (GPU) |
|---------|--------|------------|---------------|--------------|
| COCO Caption | 3 | 4 | 2e-4 | ~18GB |
| DocVQA | 5 | 4 | 2e-4 | ~18GB |
| GQA | 3 | 4 | 2e-4 | ~18GB |
| VQAv2 | 3 | 4 | 2e-4 | ~18GB |

## 🔄 테스트용 소규모 학습

빠른 테스트를 위해 샘플 수를 제한할 수 있습니다:

```bash
python finetuning/vqav2.py \
  --model_path google/paligemma-3b-pt-448 \
  --output_dir ./models/test \
  --num_epochs 1 \
  --batch_size 2 \
  --max_train_samples 1000 \
  --max_eval_samples 100
```

## 📈 Monitoring with W&B

```bash
# 로그인 (최초 1회)
wandb login

# 학습 시 --use_wandb 플래그 추가
python finetuning/coco_captioning.py \
  --model_path google/paligemma-3b-pt-448 \
  --output_dir ./models/paligemma-coco \
  --use_wandb \
  --wandb_project my-paligemma-project
```

## 🎓 모델 병합 (Merging)

학습 완료 후 모델들을 병합하려면:

```bash
python merge_vlm.py \
  --base_model google/paligemma-3b-pt-448 \
  --model_a ./models/paligemma-vqav2 \
  --model_b ./models/paligemma-coco \
  --output ./models/merged-vqa-caption \
  --mode ties \
  --density 0.3
```

## 💡 Tips

1. **메모리 부족 시**: batch_size 줄이기 또는 gradient_accumulation_steps 늘리기
2. **빠른 수렴**: learning_rate를 3e-4로 높이기 (단, overfitting 주의)
3. **긴 텍스트**: DocVQA는 max_length=256 사용 (문서 이해)
4. **데이터셋 크기**: 전체 학습은 시간이 오래 걸림 (VQAv2 ~400k samples)

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# batch_size 줄이기
--batch_size 2

# 또는 더 aggressive한 gradient accumulation
--gradient_accumulation_steps 8
```

### Dataset Loading Error
- 일부 데이터셋은 수동 다운로드 필요
- HuggingFace 계정 로그인 확인: `huggingface-cli login`

### BitsAndBytes Error
```bash
# CUDA 버전 확인
pip install bitsandbytes --upgrade
```
