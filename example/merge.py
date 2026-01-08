import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

# =========================================================
# 1. 모델 설정 (수정됨)
# =========================================================
# Base를 Pretrain으로 사용, 두 개의 서로 다른 task 병합
base_model_id = "google/paligemma-3b-pt-224"           # Pre-trained Base
model_a_id = "NYUAD-ComNets/FaceScanPaliGemma_Race"    # Task A: 인종 분류
model_b_id = "NYUAD-ComNets/FaceScanPaliGemma_Age"     # Task B: 나이 그룹 분류

output_path = "./weights/paligemma_race_age"

# =========================================================
# 2. TiES-Merging 함수 (VLM 특화 수정)
# =========================================================
def ties_merging_vlm_cpu(base_model, model_a, model_b, k=0.2, lam=1.0):
    
    base_sd = base_model.state_dict()
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    
    merged_sd = {}
    all_keys = list(base_sd.keys())
    
    print(f"\nStart Merging {len(all_keys)} tensors...")
    
    with torch.no_grad():
        for idx, key in enumerate(all_keys):
            if (idx + 1) % 100 == 0:
                print(f"Progress: {idx + 1}/{len(all_keys)}")

            # [중요] Vision 관련 모듈은 병합하지 않고 Model B(ShareGPT4V) 것을 사용
            # 이유: 비전 인코더와 프로젝터는 섞으면 성능이 급격히 저하됨
            if "vision_tower" in key or "multi_modal_projector" in key:
                # ShareGPT4V의 비전 모듈이 더 강력하므로 채택
                if key in sd_b:
                    merged_sd[key] = sd_b[key].clone()
                else:
                    merged_sd[key] = base_sd[key].clone()
                continue
            
            # 키가 모델들에 없는 경우 방어 로직
            if key not in sd_a or key not in sd_b:
                print(f"Skipping key {key} (missing in one of the models)")
                merged_sd[key] = base_sd[key]
                continue

            # --- TiES Merging Logic (LLM Backbone만 적용) ---
            tv_base = base_sd[key].float()
            tv_a = sd_a[key].float()
            tv_b = sd_b[key].float()
            
            # 1. Task Vector
            delta_a = tv_a - tv_base
            delta_b = tv_b - tv_base
            
            # 2. Trim (상위 k%)
            def trim(tensor, density):
                if density >= 1.0: return tensor
                if tensor.numel() == 0: return tensor # 빈 텐서 방지
                
                k_idx = int(tensor.numel() * density)
                if k_idx == 0: return tensor * 0
                
                # 절댓값 기준 상위 k개
                target_k = tensor.numel() - k_idx + 1
                threshold = torch.kthvalue(tensor.abs().flatten(), target_k).values
                return tensor * (tensor.abs() >= threshold)

            delta_a_trimmed = trim(delta_a, k)
            delta_b_trimmed = trim(delta_b, k)
            
            # 3. Elect Sign
            stacked = torch.stack([delta_a_trimmed, delta_b_trimmed])
            summed = stacked.sum(dim=0)
            sign = torch.sign(summed)
            
            # 방향 투표 (Disjoint Merge)
            mask = (torch.sign(stacked) == sign)
            elected = stacked * mask
            
            # 4. Merge
            final_delta = elected.mean(dim=0) * lam
            merged_sd[key] = (tv_base + final_delta).to(dtype=torch.float16)

    return merged_sd

# =========================================================
# 3. 실행부
# =========================================================
print("Loading models... (This requires high RAM)")

# 1. Base Model 로드
print(f"Loading Base: {base_model_id}")
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    base_model_id, dtype=torch.bfloat16, device_map="cpu"
)

# 2. Model A 로드
print(f"Loading Model A: {model_a_id}")
model_a = PaliGemmaForConditionalGeneration.from_pretrained(
    model_a_id, dtype=torch.bfloat16, device_map="cpu"
)

# 3. Model B 로드
print(f"Loading Model B: {model_b_id}")
model_b = PaliGemmaForConditionalGeneration.from_pretrained(
    model_b_id, dtype=torch.bfloat16, device_map="cpu"
)

# 4. 병합 실행
print("Starting TiES Merging...")
# k=0.3 (상위 30% 파라미터만 사용), lam=1.0 (가중치 스케일)
final_state_dict = ties_merging_vlm_cpu(base_model, model_a, model_b, k=0.3, lam=1.0)

# 5. 결과 적용 및 저장
print("Applying merged weights...")
base_model.load_state_dict(final_state_dict)

print(f"Saving to {output_path}...")
base_model.save_pretrained(output_path)

# Processor 복사 (Base 사용)
print("Saving processor from Base (PaliGemma-PT-224)...")
processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224", use_fast=True)
processor.save_pretrained(output_path)

print("✅ Merging Complete!")