
"""
Target-Only Membership Inference Attack for Vision-Language Models (COCO Caption)
Perplexity-based MIA, COCO train2017=member, val2017=non-member
"""


import os
import json
import torch
import random
import pandas as pd
import numpy as np
from scipy.stats import norm
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_CONFIG = {
    "model_id": "google/paligemma-3b-ft-cococap-224",
    "merged_model_path": "./models/merge_weights/pali_vqa_caption",  # merge_vlm.py output dir
    "is_lora_adapter": False
}
MODEL_ID = MODEL_CONFIG["model_id"]
DATA_ROOT = "D:/datasets/coco/"
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train2017")
VAL_IMAGES_DIR = os.path.join(DATA_ROOT, "val2017")
TRAIN_ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_train2017.json")
VAL_ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_val2017.json")
OUTPUT_DIR = "./attack_results/eval/0128/pali_merge_target_only"
NUM_EVAL_SAMPLES = 500
PROMPT = "Describe this image."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42



# -----------------------------------------------------------------------------
# Data Loading (COCO)
# -----------------------------------------------------------------------------
def load_coco_data(annotation_file, images_dir, num_samples=None, seed=42):
    """Load COCO caption data with ground truth captions."""
    print(f"[Info] Loading data from {annotation_file}...")
    if not os.path.exists(annotation_file):
        print(f"[Error] Annotation file not found: {annotation_file}")
        return []
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    image_to_captions = defaultdict(list)
    for ann in data['annotations']:
        img_id = ann['image_id']
        caption = ann['caption']
        image_to_captions[img_id].append(caption)
    formatted_data = []
    for img_id, captions in image_to_captions.items():
        if img_id in img_id_to_filename:
            file_name = img_id_to_filename[img_id]
            full_path = os.path.join(images_dir, file_name)
            if os.path.exists(full_path):
                formatted_data.append({
                    "image_id": img_id,
                    "image_path": full_path,
                    "captions": captions
                })
    print(f"[Info] Loaded {len(formatted_data)} images with captions")
    if num_samples:
        random.seed(seed)
        random.shuffle(formatted_data)
        formatted_data = formatted_data[:num_samples]
        print(f"[Info] Sampled {len(formatted_data)} images for evaluation")
    return formatted_data

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
def load_model_and_processor():
    config = MODEL_CONFIG
    model_id = config["model_id"]
    merged_model_path = config.get("merged_model_path", None)
    is_lora_adapter = config.get("is_lora_adapter", False)
    print(f"[Info] Loading model: {model_id}")
    # Processor: merged_model_path 우선, 없으면 model_id
    processor = AutoProcessor.from_pretrained(
        merged_model_path if merged_model_path and os.path.exists(merged_model_path) else model_id
    )
    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    # Merged model 우선, 없으면 base model
    if merged_model_path and os.path.exists(merged_model_path):
        adapter_config_path = os.path.join(merged_model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"[Warning] Removing adapter_config.json from merged model directory: {adapter_config_path}")
            os.remove(adapter_config_path)
        print(f"[Info] Loading merged model from: {merged_model_path}")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            merged_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        print(f"[Info] Loading base model: {model_id}")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    model.eval()
    print(f"[Info] Model loaded successfully")
    return model, processor


# =========================================================
# Similarity Computation
# =========================================================

def compute_similarity_with_temperature(
    model,
    processor,
    image: Image.Image,
    attribute: str,
    gt_label: str,
    temperature: float,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute similarity between model output and ground truth at given temperature.
    
    Returns:
        Dictionary with various similarity metrics
    """
    prompt = PROMPT
    
   # Prepare inputs with ground truth caption
    prompt_with_image = f"<image>{PROMPT}"
    inputs = processor(
        text=prompt_with_image,
        images=image,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        # Generate with temperature
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=1
        )
    
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    prompt_without_image = prompt.replace("<image>", "")
    answer = generated_text.replace(prompt_without_image, "").strip().lower()
    
    gt_label_lower = gt_label.lower()
    
    # Simple similarity metrics
    exact_match = 1.0 if answer == gt_label_lower else 0.0
    contains_match = 1.0 if gt_label_lower in answer else 0.0
    
    # Token overlap (simple word-based)
    answer_tokens = set(answer.split())
    gt_tokens = set(gt_label_lower.split())
    
    if len(answer_tokens) > 0:
        precision = len(answer_tokens & gt_tokens) / len(answer_tokens)
        recall = len(answer_tokens & gt_tokens) / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        precision = recall = f1 = 0.0
    
    return {
        'exact_match': exact_match,
        'contains': contains_match,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_dataset_similarities(
    model,
    processor,
    data_df: pd.DataFrame,
    attribute: str,
    temperatures: List[float],
    device: str = "cuda",
    dataset_name: str = "dataset"
) -> List[Dict]:
    """Compute similarities for all samples at multiple temperatures."""
    results = []
    
    print(f"🔄 Computing similarities for {dataset_name}...")
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"Processing {dataset_name}"):
        try:
            image = Image.open(row['image_path']).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading {row['image_path']}: {e}")
            continue
        
        gt_label = row[attribute]
        sample_result = {'image': row['file'], 'gt_label': gt_label}
        
        for temp in temperatures:
            similarities = compute_similarity_with_temperature(
                model, processor, image, attribute, gt_label, temp, device
            )
            sample_result[f'similarity_{temp}'] = similarities
        
        results.append(sample_result)
    
    return results


# =========================================================
# Statistical Testing
# =========================================================

def perform_z_test(group1: List[float], group2: List[float]) -> float:
    """Perform z-test between two groups."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    pooled_se = np.sqrt(std1**2/n1 + std2**2/n2)
    
    # Handle case when pooled_se is 0 or very small
    if pooled_se < 1e-10:
        # If means are different, return very small p-value, otherwise large p-value
        if abs(mean1 - mean2) > 1e-10:
            return 0.0
        else:
            return 1.0
    
    z = (mean1 - mean2) / pooled_se
    p_value = norm.sf(z)
    
    # Handle NaN or infinite values
    if np.isnan(p_value) or np.isinf(p_value):
        return 0.5  # Return neutral value
    
    return p_value

def target_only_inference(
    member_data: List[Dict],
    non_member_data: List[Dict],
    granularity: int,
    temperature_low: float,
    temperature_high: float,
    metric: str,
    num_tests: int = 1000
    ) -> Tuple[float, float, float]:
    """
    Perform target-only membership inference attack.
    
    Returns:
        Tuple of (AUC, Attack Accuracy, TPR@5%FPR)
    """
    p_list = []
    label_list = []
    
    for _ in range(num_tests):
        # Sample members
        member_samples = np.random.choice(len(member_data), granularity, replace=False)
        member_low = [member_data[i][f'similarity_{temperature_low}'][metric] for i in member_samples]
        member_high = [member_data[i][f'similarity_{temperature_high}'][metric] for i in member_samples]
        p_member = perform_z_test(member_low, member_high)
        p_list.append(p_member)
        label_list.append(0)  # Member label
        
        # Sample non-members
        non_member_samples = np.random.choice(len(non_member_data), granularity, replace=False)
        non_member_low = [non_member_data[i][f'similarity_{temperature_low}'][metric] for i in non_member_samples]
        non_member_high = [non_member_data[i][f'similarity_{temperature_high}'][metric] for i in non_member_samples]
        p_non_member = perform_z_test(non_member_low, non_member_high)
        p_list.append(p_non_member)
        label_list.append(1)  # Non-member label
    
    # Calculate AUC
    p_array = np.array(p_list)
    label_array = np.array(label_list)
    
    plt.hist(p_array[label_array==0], bins=50, alpha=0.5, label='Member')
    plt.hist(p_array[label_array==1], bins=50, alpha=0.5, label='Non-member')
    plt.legend()
    plot_file = os.path.join(OUTPUT_DIR, 'p_value_distribution.png')
    plt.savefig(plot_file)
    plt.close()

    # Check for NaN values
    valid_mask = ~np.isnan(p_array)
    if valid_mask.sum() < len(p_array):
        print(f"⚠️  Warning: {(~valid_mask).sum()} NaN values detected and removed")
        p_array = p_array[valid_mask]
        label_array = label_array[valid_mask]
    
    # Check if we have enough valid samples
    if len(p_array) < 10:
        print("❌ Error: Too few valid samples for evaluation")
        return 0.5, 0.5, 0.0
    
    auc = roc_auc_score(label_array, p_array)
    
    # Calculate Attack Accuracy (using optimal threshold)
    fpr, tpr, thresholds = roc_curve(label_array, p_array)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = [1 if p >= optimal_threshold else 0 for p in p_array]
    attack_acc = accuracy_score(label_array, predictions)
    
    # Calculate TPR@5%FPR
    target_fpr = 0.05
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) > 0:
        tpr_at_5fpr = tpr[idx[-1]]
    else:
        tpr_at_5fpr = 0.0
    
    return auc, attack_acc, tpr_at_5fpr

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Target-Only Membership Inference Attack (COCO Caption)")
    print("=" * 80)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"\n[Model] {MODEL_ID}")
    model, processor = load_model_and_processor()
    print("\n[Step 1] Loading member data (train2017)...")
    member_data = load_coco_data(
        TRAIN_ANNOTATION_FILE,
        TRAIN_IMAGES_DIR,
        num_samples=NUM_EVAL_SAMPLES,
        seed=SEED
    )
    print("\n[Step 2] Loading non-member data (val2017)...")
    nonmember_data = load_coco_data(
        VAL_ANNOTATION_FILE,
        VAL_IMAGES_DIR,
        num_samples=NUM_EVAL_SAMPLES,
        seed=SEED
    )
    if len(member_data) == 0 or len(nonmember_data) == 0:
        print("[Error] Failed to load data. Please check file paths.")
        return
    print("\n[Step 3] Compute metrics and evaluate MIA...")
    # Parameters
    attribute = "captions"  # For COCO caption, use the first caption as GT
    temperatures = [0.7, 1.3]
    metric = "f1"  # You can also use 'exact_match', 'contains', etc.
    granularity = 5
    num_tests = 1000

    # Convert to DataFrame and use first caption as GT label
    member_df = pd.DataFrame(member_data)
    member_df["file"] = member_df["image_path"].apply(lambda x: os.path.basename(x))
    member_df["caption"] = member_df["captions"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "")
    nonmember_df = pd.DataFrame(nonmember_data)
    nonmember_df["file"] = nonmember_df["image_path"].apply(lambda x: os.path.basename(x))
    nonmember_df["caption"] = nonmember_df["captions"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "")

    # Compute similarities for both datasets
    member_results = compute_dataset_similarities(
        model, processor, member_df, attribute="caption", temperatures=temperatures, device=DEVICE, dataset_name="member"
    )
    nonmember_results = compute_dataset_similarities(
        model, processor, nonmember_df, attribute="caption", temperatures=temperatures, device=DEVICE, dataset_name="non-member"
    )

    # Run target-only inference attack
    auc, attack_acc, tpr_at_5fpr = target_only_inference(
        member_results, nonmember_results, granularity=granularity,
        temperature_low=temperatures[0], temperature_high=temperatures[1],
        metric=metric, num_tests=num_tests
    )

    print("\n=================== MIA Results ===================")
    print(f"AUC: {auc:.4f}")
    print(f"Attack Accuracy: {attack_acc:.4f}")
    print(f"TPR@5%FPR: {tpr_at_5fpr:.4f}")
    print("==================================================")

    # Optionally save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(member_results).to_json(os.path.join(OUTPUT_DIR, "member_results.json"), orient="records", force_ascii=False)
    pd.DataFrame(nonmember_results).to_json(os.path.join(OUTPUT_DIR, "nonmember_results.json"), orient="records", force_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "mia_metrics.txt"), "w") as f:
        f.write(f"AUC: {auc:.4f}\nAttack Accuracy: {attack_acc:.4f}\nTPR@5%FPR: {tpr_at_5fpr:.4f}\n")
    print(f"\n[Info] Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
