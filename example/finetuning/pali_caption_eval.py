import os
import json
import torch
import random
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import defaultdict
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig
)

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge


# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
MODEL_ID = "google/paligemma-3b-ft-cococap-224"
DATA_ROOT = "D:/datasets/coco/"
IMAGES_DIR = os.path.join(DATA_ROOT, "val2017")
ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_val2017.json")
OUTPUT_FILE = "./paligemma_caption_evaluation_results.json"

# 생성 파라미터
MAX_NEW_TOKENS = 40  # COCO captions are typically short (10-20 words)
TEMPERATURE = 0.0  # Greedy decoding for consistency
DO_SAMPLE = False

# -----------------------------------------------------------------------------
# 데이터 로딩
# -----------------------------------------------------------------------------

def load_coco_captions_for_eval(annotation_file, images_dir):
    """
    COCO 평가용 데이터 로드.
    이미지별로 여러 개의 reference caption을 그룹화합니다.
    """
    print(f"[Info] Loading annotations from {annotation_file}...")
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 이미지 ID -> 파일명 매핑
    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # 이미지 ID -> 캡션 리스트 (한 이미지당 여러 reference captions)
    image_to_captions = defaultdict(list)
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        caption = ann['caption']
        image_to_captions[img_id].append(caption)
    
    # 이미지 경로와 캡션들을 리스트로 변환
    formatted_data = []
    missing_count = 0
    
    for img_id, captions in image_to_captions.items():
        if img_id in img_id_to_filename:
            file_name = img_id_to_filename[img_id]
            full_path = os.path.join(images_dir, file_name)
            
            if os.path.exists(full_path):
                formatted_data.append({
                    "image_id": img_id,
                    "image_path": full_path,
                    "captions": captions  # 여러 reference captions
                })
            else:
                missing_count += 1
    
    print(f"[Info] Found {len(formatted_data)} images with captions. (Missing: {missing_count})")
    return formatted_data

# -----------------------------------------------------------------------------
# 평가 함수
# -----------------------------------------------------------------------------

def calculate_caption_metrics(predictions: Dict[int, List[str]], 
                              references: Dict[int, List[str]]) -> Dict:
    """
    CIDEr, BLEU, METEOR, ROUGE 등의 지표를 계산합니다.
    
    Args:
        predictions: {image_id: [predicted_caption]}
        references: {image_id: [ref_caption1, ref_caption2, ...]}
    """
    print("[Info] Calculating captioning metrics...")
    print("[Info] Using CIDEr, BLEU, and ROUGE metrics (METEOR excluded due to Java dependency)")
    
    # 평가 지표 초기화 (METEOR 제외)
    scorers = [
        (Cider(), "CIDEr"),
        (Bleu(4), "BLEU"),
        (Rouge(), "ROUGE_L")
    ]
    
    metrics = {}
    
    for scorer, method in scorers:
        print(f"[Info] Computing {method}...")
        try:
            score, scores = scorer.compute_score(references, predictions)
            
            if method == "BLEU":
                # BLEU는 BLEU-1, BLEU-2, BLEU-3, BLEU-4를 반환
                for i, s in enumerate(score):
                    metrics[f"{method}-{i+1}"] = s
            else:
                metrics[method] = score
                
        except Exception as e:
            print(f"[Error] Failed to compute {method}: {e}")
            metrics[method] = 0.0
    
    return metrics

# -----------------------------------------------------------------------------
# 모델 로드 및 추론
# -----------------------------------------------------------------------------

def load_model_and_processor(model_id):
    """
    PaliGemma 모델과 프로세서를 로드합니다.
    """
    print(f"[Init] Loading PaliGemma model from {model_id}...")
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 양자화 설정 (옵션 - 메모리 절약용)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 모델 로드
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    model.eval()
    print("[Info] Model loaded successfully")
    return model, processor

def generate_caption(model, processor, image_path: str) -> str:
    """
    이미지에 대한 캡션을 생성합니다.
    PaliGemma는 간단한 프롬프트를 사용합니다.
    """
    try:
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        
        # PaliGemma 프롬프트 (<image> 토큰 포함)
        # PaliGemma-ft-cococap 모델은 "<image>caption en" 프롬프트를 사용
        prompt = "<image>caption en"
        
        # 입력 처리
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="longest"
        ).to(model.device)
        
        # 캡션 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE if DO_SAMPLE else None,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # 디코딩 (입력 프롬프트 제외)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        caption = processor.decode(generated_ids, skip_special_tokens=True)
        
        # 캡션 정리 및 첫 번째 문장만 추출
        caption = caption.strip()
        
        # 첫 번째 온점까지만 자르기 (하나의 문장만)
        if '.' in caption:
            caption = caption.split('.')[0] + '.'
        
        return caption
        
    except Exception as e:
        print(f"[Error] Failed to generate caption for {image_path}: {e}")
        return ""

# -----------------------------------------------------------------------------
# 메인 평가 루틴
# -----------------------------------------------------------------------------

def main():
    print("="*80)
    print("PaliGemma Image Captioning Evaluation")
    print("="*80)
    
    # 1. 데이터 로드
    raw_data = load_coco_captions_for_eval(ANNOTATION_FILE, IMAGES_DIR)
    
    if len(raw_data) == 0:
        print("[Error] No data found!")
        return
    
    # 학습 시와 동일하게 데이터 분할 (80% train, 20% val)
    random.seed(42)
    random.shuffle(raw_data)
    
    split_idx = int(len(raw_data) * 0.8)
    train_data = raw_data[:split_idx]
    test_data = raw_data[split_idx:]  # validation 데이터만 평가
    
    print(f"[Info] Total images: {len(raw_data)}")
    print(f"[Info] Train: {len(train_data)}, Validation: {len(test_data)}")
    print(f"[Info] Evaluating on validation set: {len(test_data)} images")
    
    # 2. 모델 및 프로세서 로드
    model, processor = load_model_and_processor(MODEL_ID)
    
    # 3. 추론 수행
    print("\n[Inference] Generating captions...")
    predictions = {}  # {image_id: [predicted_caption]}
    references = {}   # {image_id: [ref_caption1, ref_caption2, ...]}
    detailed_results = []
    
    for idx, item in enumerate(tqdm(test_data, desc="Generating captions")):
        img_id = item["image_id"]
        
        # 캡션 생성
        pred_caption = generate_caption(model, processor, item["image_path"])
        
        # 평가 형식에 맞게 저장
        predictions[img_id] = [pred_caption]
        references[img_id] = item["captions"]
        
        detailed_results.append({
            "image_id": img_id,
            "image_path": item["image_path"],
            "predicted_caption": pred_caption,
            "reference_captions": item["captions"]
        })
        
        # 진행 상황 출력 (처음 3개)
        if idx < 3:
            print(f"\n--- Sample {idx+1} ---")
            print(f"Image ID: {img_id}")
            print(f"Predicted: {pred_caption}")
            print(f"References: {item['captions'][:2]}")  # 처음 2개만 출력
    
    # 4. 메트릭 계산
    print("\n[Evaluation] Calculating metrics...")
    metrics = calculate_caption_metrics(predictions, references)
    
    # 5. 결과 출력
    print("\n" + "="*80)
    print("EVALUATION RESULTS - PaliGemma")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"Total Images Evaluated: {len(test_data)}")
    print(f"\nMetrics:")
    for metric_name, score in metrics.items():
        if isinstance(score, float):
            print(f"  {metric_name}: {score:.4f}")
        else:
            print(f"  {metric_name}: {score}")
    print("="*80)
    
    # 6. 결과 저장
    output_data = {
        "model_info": {
            "model_id": MODEL_ID,
            "model_type": "PaliGemma"
        },
        "generation_config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "do_sample": DO_SAMPLE
        },
        "dataset_info": {
            "total_images": len(raw_data),
            "train_images": len(train_data),
            "val_images": len(test_data)
        },
        "metrics": metrics,
        "detailed_results": detailed_results[:50]  # 처음 50개만 저장
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Saved] Results saved to {OUTPUT_FILE}")
    
    # 7. 샘플 결과 출력
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (First 3)")
    print("="*80)
    
    for idx, result in enumerate(detailed_results[:3]):
        print(f"\n--- Sample {idx+1} ---")
        print(f"Predicted: {result['predicted_caption']}")
        print(f"Reference 1: {result['reference_captions'][0]}")
        if len(result['reference_captions']) > 1:
            print(f"Reference 2: {result['reference_captions'][1]}")
    
    print("\n[Complete] Evaluation finished!")

if __name__ == "__main__":
    main()
