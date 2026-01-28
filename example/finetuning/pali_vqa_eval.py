import os
import json
import torch
import re
import random
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig
)

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
# MODEL_ID = "google/paligemma-3b-ft-vqav2-224"
MODEL_ID = "./merge_weights/pali_vqa_caption"
DATA_ROOT = "D:/VQA/cocoqa/test"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
QUESTIONS_JSON = os.path.join(DATA_ROOT, "questions.json")
OUTPUT_FILE = "./merge_weights/paligemma_vqa_evaluation_results.json"

# 생성 파라미터
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0  # Greedy decoding for deterministic answers
DO_SAMPLE = False

# -----------------------------------------------------------------------------
# 데이터 로딩
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
    
    for item in data:
        if len(item) < 3:
            print(f"[Warning] Invalid item format: {item}")
            continue
            
        question = item[0]
        answer = item[1]
        image_id = item[2]
        
        # 이미지 파일명 생성 (COCO 형식)
        file_name = f"{image_id:012d}.jpg"
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
            # 다른 파일 형식 시도
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
    
    print(f"[Info] Found {len(formatted_data)} valid QA pairs. (Missing: {missing_count})")
    return formatted_data

# -----------------------------------------------------------------------------
# 답변 정규화 및 평가 함수
# -----------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    """
    답변을 정규화합니다.
    - 소문자 변환
    - 구두점 제거
    - 연속된 공백을 하나로
    - 앞뒤 공백 제거
    """
    # 소문자 변환
    answer = answer.lower()
    
    # 구두점 제거 (알파벳, 숫자, 공백만 유지)
    answer = re.sub(r'[^a-z0-9\s]', '', answer)
    
    # 연속된 공백을 하나로
    answer = re.sub(r'\s+', ' ', answer)
    
    # 앞뒤 공백 제거
    answer = answer.strip()
    
    return answer

def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> Dict:
    """
    예측값과 정답을 비교하여 accuracy를 계산합니다.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    exact_match = 0
    normalized_match = 0
    total = len(predictions)
    
    results = []
    
    for pred, gt in zip(predictions, ground_truths):
        # Exact match (대소문자 무시)
        is_exact = (pred.lower() == gt.lower())
        if is_exact:
            exact_match += 1
        
        # Normalized match (대소문자 + 구두점 + 공백 정규화)
        norm_pred = normalize_answer(pred)
        norm_gt = normalize_answer(gt)
        is_normalized = (norm_pred == norm_gt)
        if is_normalized:
            normalized_match += 1
        
        results.append({
            "prediction": pred,
            "ground_truth": gt,
            "exact_match": is_exact,
            "normalized_match": is_normalized
        })
    
    metrics = {
        "total_samples": total,
        "exact_match_accuracy": exact_match / total * 100,
        "normalized_accuracy": normalized_match / total * 100,
        "exact_match_count": exact_match,
        "normalized_match_count": normalized_match
    }
    
    return metrics, results

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
    
    # 양자화 설정
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

def generate_answer(model, processor, image_path: str, question: str) -> str:
    """
    이미지와 질문을 입력받아 답변을 생성합니다.
    PaliGemma VQA 모델은 질문을 프롬프트로 사용합니다.
    """
    try:
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        
        # PaliGemma VQA 프롬프트 (<image> 토큰 포함)
        # VQA 모델은 질문을 그대로 프롬프트로 사용
        prompt = f"<image>{question}"
        
        # 입력 처리
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="longest"
        ).to(model.device)
        
        # 답변 생성
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
        answer = processor.decode(generated_ids, skip_special_tokens=True)
        
        # 답변 정리
        answer = answer.strip()
        
        return answer
        
    except Exception as e:
        print(f"[Error] Failed to generate answer: {e}")
        return ""

# -----------------------------------------------------------------------------
# 메인 평가 루틴
# -----------------------------------------------------------------------------

def main():
    print("="*80)
    print("PaliGemma VQA Model Evaluation")
    print("="*80)
    
    # 1. 데이터 로드
    raw_data = load_vqa_data(QUESTIONS_JSON, IMAGES_DIR)
    
    if len(raw_data) == 0:
        print("[Error] No test data found!")
        return
    
    # 학습 시와 동일하게 데이터 분할 (80% train, 20% val)
    random.seed(42)
    random.shuffle(raw_data)
    
    split_idx = int(len(raw_data) * 0.8)  # 80% train, 20% val
    train_data = raw_data[:split_idx]
    test_data = raw_data[split_idx:]  # validation 데이터만 평가에 사용
    
    print(f"[Info] Total data: {len(raw_data)} samples")
    print(f"[Info] Train: {len(train_data)} samples, Validation: {len(test_data)} samples")
    print(f"[Info] Evaluating on validation set: {len(test_data)} samples")
    
    # 2. 모델 및 프로세서 로드
    model, processor = load_model_and_processor(MODEL_ID)
    
    # 3. 추론 수행
    print("\n[Inference] Generating answers...")
    predictions = []
    ground_truths = []
    detailed_results = []
    
    for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
        # 답변 생성
        pred_answer = generate_answer(
            model, 
            processor, 
            item["image_path"], 
            item["question"]
        )
        
        predictions.append(pred_answer)
        ground_truths.append(item["answer"])
        
        detailed_results.append({
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "question": item["question"],
            "predicted_answer": pred_answer,
            "ground_truth": item["answer"]
        })
        
        # 진행 상황 출력 (처음 5개)
        if idx < 5:
            print(f"\n--- Sample {idx+1} ---")
            print(f"Q: {item['question']}")
            print(f"Pred: {pred_answer}")
            print(f"GT: {item['answer']}")
    
    # 4. 정확도 계산
    print("\n[Evaluation] Calculating metrics...")
    metrics, comparison_results = calculate_accuracy(predictions, ground_truths)
    
    # 5. 결과 출력
    print("\n" + "="*80)
    print("EVALUATION RESULTS - PaliGemma VQA")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}% ({metrics['exact_match_count']}/{metrics['total_samples']})")
    print(f"Normalized Accuracy: {metrics['normalized_accuracy']:.2f}% ({metrics['normalized_match_count']}/{metrics['total_samples']})")
    print("="*80)
    
    # 6. 결과 저장
    output_data = {
        "model_info": {
            "model_id": MODEL_ID,
            "model_type": "PaliGemma-VQA"
        },
        "generation_config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "do_sample": DO_SAMPLE
        },
        "metrics": metrics,
        "detailed_results": detailed_results,
        "comparison_results": comparison_results
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Saved] Results saved to {OUTPUT_FILE}")
    
    # 7. 오답 분석 (처음 3개)
    print("\n" + "="*80)
    print("ERROR ANALYSIS (First 3 mistakes)")
    print("="*80)
    
    mistakes = [r for r in comparison_results if not r["normalized_match"]][:3]
    for idx, mistake in enumerate(mistakes):
        print(f"\n--- Mistake {idx+1} ---")
        print(f"Predicted: {mistake['prediction']}")
        print(f"Ground Truth: {mistake['ground_truth']}")
    
    print("\n[Complete] Evaluation finished!")

if __name__ == "__main__":
    main()
