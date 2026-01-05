import torch
import copy
import gc
from transformers import AutoTokenizer, LlavaForConditionalGeneration

# =========================================================
# 1. 모델 경로 설정
# =========================================================
base_model_id = "llava-hf/llava-1.5-7b-hf"
model_a_id = "llava-hf/vip-llava-7b-hf"       # Visual Prompting 능력 강화
model_b_id = "gokul-mbzuai/sharegpt4v-7b-hf"  # 상세 캡셔닝/설명 능력 강화

# =========================================================
# 2. TiES-Merging 로직 (CPU 모드)
# =========================================================
def ties_merging_cpu(base_sd, model_a_sd, model_b_sd, k=0.2, lam=1.0):
    """
    TiES-Merging: Trim, Elect Sign, Merge
    모든 연산이 CPU에서 이루어집니다.
    """
    merged_sd = {}
    all_keys = list(base_sd.keys())
    
    print(f"Start Merging {len(all_keys)} tensors on CPU...")
    
    # 불필요한 그래디언트 계산 방지
    with torch.no_grad():
        for key in all_keys:
            # Vision Tower는 보통 Freeze 되어 있으므로 Base를 그대로 사용
            # 주의: 만약 Model A/B가 Vision Tower를 파인튜닝했다면 이 로직은 그 변화를 무시합니다.
            if "vision_tower" in key:
                merged_sd[key] = base_sd[key]
                continue
            
            # 키가 다른 모델에 없는 경우 (드문 경우) Base 유지
            if key not in model_a_sd or key not in model_b_sd:
                merged_sd[key] = base_sd[key]
                continue

            # 텐서 가져오기
            tv_base = base_sd[key].float()
            tv_a = model_a_sd[key].float()
            tv_b = model_b_sd[key].float()
            
            # 1. Task Vector 계산
            delta_a = tv_a - tv_base
            delta_b = tv_b - tv_base
            
            # 2. Trim (상위 k%만 남기기)
            def trim(tensor, density):
                if density >= 1.0: return tensor
                k_idx = int(tensor.numel() * density)
                if k_idx == 0: return tensor * 0 # density가 너무 작으면 0으로
                
                # kthvalue: k번째로 '작은' 값을 찾음. 
                # 상위 k개를 찾으려면 (전체 - k + 1)번째 작은 값을 임계값으로 설정
                target_k = tensor.numel() - k_idx + 1
                threshold = torch.kthvalue(tensor.abs().flatten(), target_k).values
                return tensor * (tensor.abs() >= threshold)

            delta_a_trimmed = trim(delta_a, k)
            delta_b_trimmed = trim(delta_b, k)
            
            # 3. Elect Sign (방향 투표)
            stacked = torch.stack([delta_a_trimmed, delta_b_trimmed])
            summed = stacked.sum(dim=0)
            sign = torch.sign(summed)
            
            # 방향이 같은 것만 살리기 (Disjoint Merge)
            mask = (torch.sign(stacked) == sign)
            elected = stacked * mask
            
            # 4. Merge (평균 후 Base에 더하기)
            final_delta = elected.mean(dim=0) * lam
            merged_sd[key] = (tv_base + final_delta).to(dtype=torch.float16)
            
    return merged_sd

# =========================================================
# 3. 효율적인 로딩 및 실행
# =========================================================
print("Loading state_dicts to CPU RAM... (Warning: Requires >45GB RAM)")

# LLaVA 모델 로드를 위해 LlavaForConditionalGeneration 사용
# Base 모델 로드
print(f"Loading Base: {base_model_id}")
base_model = LlavaForConditionalGeneration.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16, 
    device_map="cpu",
    low_cpu_mem_usage=True
)
base_sd = base_model.state_dict()

# Model A 로드
print(f"Loading Model A: {model_a_id}")
model_a = LlavaForConditionalGeneration.from_pretrained(
    model_a_id, 
    torch_dtype=torch.float16, 
    device_map="cpu", 
    low_cpu_mem_usage=True
)
sd_a = model_a.state_dict()
del model_a
gc.collect()

# Model B 로드
print(f"Loading Model B: {model_b_id}")
model_b = LlavaForConditionalGeneration.from_pretrained(
    model_b_id, 
    torch_dtype=torch.float16, 
    device_map="cpu", 
    low_cpu_mem_usage=True
)
sd_b = model_b.state_dict()
del model_b
gc.collect()

# 병합 수행
final_state_dict = ties_merging_cpu(base_sd, sd_a, sd_b, k=0.2, lam=1.0)

# 메모리 정리
del base_sd, sd_a, sd_b
gc.collect()

# =========================================================
# 4. 저장
# =========================================================
print("Saving merged model...")
base_model.load_state_dict(final_state_dict)
base_model.save_pretrained("./cpu_merged_llava")

# Tokenizer 및 Processor 저장 (LLaVA는 Processor도 중요)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained("./cpu_merged_llava")

# Processor 저장 (이미지 처리를 위해 필수)
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained("./cpu_merged_llava")

print("Done! Check './cpu_merged_llava' folder.")
import torch
import copy
import gc
from transformers import AutoTokenizer, LlavaForConditionalGeneration

# =========================================================
# 1. 모델 경로 설정
# =========================================================
base_model_id = "llava-hf/llava-1.5-7b-hf"
model_a_id = "llava-hf/vip-llava-7b-hf"       # Visual Prompting 능력 강화
model_b_id = "gokul-mbzuai/sharegpt4v-7b-hf"  # 상세 캡셔닝/설명 능력 강화

# =========================================================
# 2. TiES-Merging 로직 (CPU 모드)
# =========================================================
def ties_merging_cpu(base_sd, model_a_sd, model_b_sd, k=0.2, lam=1.0):
    """
    TiES-Merging: Trim, Elect Sign, Merge
    모든 연산이 CPU에서 이루어집니다.
    """
    merged_sd = {}
    all_keys = list(base_sd.keys())
    
    print(f"Start Merging {len(all_keys)} tensors on CPU...")
    
    # 불필요한 그래디언트 계산 방지
    with torch.no_grad():
        for key in all_keys:
            # Vision Tower는 보통 Freeze 되어 있으므로 Base를 그대로 사용
            # 주의: 만약 Model A/B가 Vision Tower를 파인튜닝했다면 이 로직은 그 변화를 무시합니다.
            if "vision_tower" in key:
                merged_sd[key] = base_sd[key]
                continue
            
            # 키가 다른 모델에 없는 경우 (드문 경우) Base 유지
            if key not in model_a_sd or key not in model_b_sd:
                merged_sd[key] = base_sd[key]
                continue

            # 텐서 가져오기
            tv_base = base_sd[key].float()
            tv_a = model_a_sd[key].float()
            tv_b = model_b_sd[key].float()
            
            # 1. Task Vector 계산
            delta_a = tv_a - tv_base
            delta_b = tv_b - tv_base
            
            # 2. Trim (상위 k%만 남기기)
            def trim(tensor, density):
                if density >= 1.0: return tensor
                k_idx = int(tensor.numel() * density)
                if k_idx == 0: return tensor * 0 # density가 너무 작으면 0으로
                
                # kthvalue: k번째로 '작은' 값을 찾음. 
                # 상위 k개를 찾으려면 (전체 - k + 1)번째 작은 값을 임계값으로 설정
                target_k = tensor.numel() - k_idx + 1
                threshold = torch.kthvalue(tensor.abs().flatten(), target_k).values
                return tensor * (tensor.abs() >= threshold)

            delta_a_trimmed = trim(delta_a, k)
            delta_b_trimmed = trim(delta_b, k)
            
            # 3. Elect Sign (방향 투표)
            stacked = torch.stack([delta_a_trimmed, delta_b_trimmed])
            summed = stacked.sum(dim=0)
            sign = torch.sign(summed)
            
            # 방향이 같은 것만 살리기 (Disjoint Merge)
            mask = (torch.sign(stacked) == sign)
            elected = stacked * mask
            
            # 4. Merge (평균 후 Base에 더하기)
            final_delta = elected.mean(dim=0) * lam
            merged_sd[key] = (tv_base + final_delta).to(dtype=torch.float16)
            
    return merged_sd

# =========================================================
# 3. 효율적인 로딩 및 실행
# =========================================================
print("Loading state_dicts to CPU RAM... (Warning: Requires >45GB RAM)")

# LLaVA 모델 로드를 위해 LlavaForConditionalGeneration 사용
# Base 모델 로드
print(f"Loading Base: {base_model_id}")
base_model = LlavaForConditionalGeneration.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16, 
    device_map="cpu",
    low_cpu_mem_usage=True
)
base_sd = base_model.state_dict()

# Model A 로드
print(f"Loading Model A: {model_a_id}")
model_a = LlavaForConditionalGeneration.from_pretrained(
    model_a_id, 
    torch_dtype=torch.float16, 
    device_map="cpu", 
    low_cpu_mem_usage=True
)
sd_a = model_a.state_dict()
del model_a
gc.collect()

# Model B 로드
print(f"Loading Model B: {model_b_id}")
model_b = LlavaForConditionalGeneration.from_pretrained(
    model_b_id, 
    torch_dtype=torch.float16, 
    device_map="cpu", 
    low_cpu_mem_usage=True
)
sd_b = model_b.state_dict()
del model_b
gc.collect()

# 병합 수행
final_state_dict = ties_merging_cpu(base_sd, sd_a, sd_b, k=0.2, lam=1.0)

# 메모리 정리
del base_sd, sd_a, sd_b
gc.collect()

# =========================================================
# 4. 저장
# =========================================================
print("Saving merged model...")
base_model.load_state_dict(final_state_dict)
base_model.save_pretrained("./cpu_merged_llava")

# Tokenizer 및 Processor 저장 (LLaVA는 Processor도 중요)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained("./cpu_merged_llava")

# Processor 저장 (이미지 처리를 위해 필수)
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained("./cpu_merged_llava")

print("Done! Check './cpu_merged_llava' folder.")