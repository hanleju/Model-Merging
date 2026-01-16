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

## � Evaluation Metrics

각 태스크별로 사용하는 표준 평가 지표:

### VQAv2 (일반 VQA)
- **Metric**: VQA Accuracy
- **계산법**: `min(답변 일치 수 / 3, 1.0)`
- 10명의 annotator 중 최소 3명이 동일한 답변을 한 경우 정답

```python
# 간단한 구현 예시
def vqa_accuracy(pred, gt_answers):
    """gt_answers: list of 10 human annotations"""
    count = sum(1 for ans in gt_answers if ans == pred)
    return min(count / 3.0, 1.0)
```

### DocVQA (문서 VQA)
- **Metric**: ANLS (Average Normalized Levenshtein Similarity)
- **범위**: 0~1 (1이 완벽한 일치)
- 문서에서는 정확한 매칭보다 유사도가 중요

```python
from Levenshtein import distance

def anls(pred, gt):
    """Average Normalized Levenshtein Similarity"""
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    edit_dist = distance(pred.lower(), gt.lower())
    max_len = max(len(pred), len(gt))
    return 1.0 - (edit_dist / max_len)
```

### GQA (Visual Reasoning)
- **Metric**: Accuracy + Consistency Score
- **Accuracy**: 정확히 일치하는 답변의 비율
- **Consistency**: Compositional reasoning 평가

```python
def gqa_accuracy(pred, gt):
    """Simple exact match"""
    return 1.0 if pred.lower().strip() == gt.lower().strip() else 0.0
```

### COCO Captioning
- **주요 Metric**: CIDEr (Consensus-based Image Description Evaluation)
- **보조 Metrics**: BLEU-4, METEOR, ROUGE-L, SPICE

```python
# pycocoevalcap 사용
from pycocoevalcap.cider.cider import Cider

cider = Cider()
score, scores = cider.compute_score(gts, res)
# gts: {image_id: [ref1, ref2, ...]}
# res: {image_id: [pred]}
```

## 🧪 Evaluation 실행

평가 스크립트는 별도로 제공됩니다:

```bash
# VQAv2 평가
python eval.py \
  --task vqav2 \
  --model_path ./models/paligemma-vqav2 \
  --data_root D:/VQA/cocoqa

# COCO Captioning 평가
python eval.py \
  --task captioning \
  --model_path ./models/paligemma-coco \
  --data_root D:/coco2017
```

**필요 패키지**:
```bash
pip install python-Levenshtein
pip install pycocoevalcap  # COCO captioning metrics
```

## �🐛 Troubleshooting

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
