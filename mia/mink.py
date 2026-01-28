"""
Min-k% Probability Membership Inference Attack for Vision-Language Models
Based on: "Detecting Pretraining Data from Large Language Models" (ICLR 2024)

This script implements the Min-k% probability attack for COCO Captioning with PaliGemma.
"""

import os
import json
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
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
    "merged_model_path": False,  # merge_vlm.py output dir
    "is_lora_adapter": False
}
MODEL_ID = MODEL_CONFIG["model_id"]
DATA_ROOT = "D:/datasets/coco/"
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train2017")
VAL_IMAGES_DIR = os.path.join(DATA_ROOT, "val2017")
TRAIN_ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_train2017.json")
VAL_ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_val2017.json")
OUTPUT_DIR = "./attack_results/eval/0128/pali_mink/"

NUM_EVAL_SAMPLES = 500
PROMPT = "Describe this image."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_coco_data(annotation_file, images_dir, num_samples=None, seed=42):
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
    print(f"[Info] Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(
        merged_model_path if merged_model_path and os.path.exists(merged_model_path) else model_id
    )
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
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

# -----------------------------------------------------------------------------
# Min-k% Probability Attack
# -----------------------------------------------------------------------------
def compute_min_k_prob(model, processor, image_path, caption, device, k_percent=10):
    try:
        image = Image.open(image_path).convert('RGB')
        prompt_with_image = f"<image>{PROMPT}"
        inputs = processor(
            text=prompt_with_image,
            images=image,
            return_tensors="pt"
        ).to(device)
        caption_tokens = processor.tokenizer(
            caption,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(device)
        labels = torch.cat([
            torch.full((1, inputs.input_ids.shape[1]), -100, dtype=torch.long).to(device),
            caption_tokens
        ], dim=1)
        full_input_ids = torch.cat([inputs.input_ids, caption_tokens], dim=1)
        attention_mask = torch.ones_like(full_input_ids)
        with torch.no_grad():
            outputs = model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                pixel_values=inputs.pixel_values,
                labels=labels,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True
            )
        logits = outputs.logits[0, inputs.input_ids.shape[1]:-1, :]
        target_ids = caption_tokens[0]
        # Align lengths to avoid device-side assert
        seq_len = min(logits.shape[0], target_ids.shape[0])
        if seq_len == 0:
            return None, None
        probs = torch.softmax(logits[:seq_len], dim=-1)
        token_probs = probs[range(seq_len), target_ids[:seq_len]].cpu().numpy()
        k = max(1, int(np.ceil(len(token_probs) * k_percent / 100)))
        min_k_probs = np.partition(token_probs, k-1)[:k]
        min_k_prob = float(np.mean(min_k_probs))
        return min_k_prob, token_probs.tolist()
    except Exception as e:
        print(f"[Error] Failed to compute min-k% prob for {image_path}: {e}")
        return None, None

def compute_min_k_probs_batch(model, processor, data_samples, device, k_percent=10):
    min_k_probs = []
    all_token_probs = []
    print(f"[Info] Computing min-{k_percent}% probabilities for {len(data_samples)} samples...")
    for sample in tqdm(data_samples, desc=f"Computing min-{k_percent}% probs"):
        caption = sample["captions"][0]
        min_k_prob, token_probs = compute_min_k_prob(
            model, processor, sample["image_path"], caption, device, k_percent
        )
        min_k_probs.append(min_k_prob)
        all_token_probs.append(token_probs)
    return min_k_probs, all_token_probs

def evaluate_min_k_mia(member_min_k, nonmember_min_k, output_dir, k_percent):
    member_min_k = [p for p in member_min_k if p is not None]
    nonmember_min_k = [p for p in nonmember_min_k if p is not None]
    if len(member_min_k) == 0 or len(nonmember_min_k) == 0:
        print("[Error] No valid min-k% probability samples for evaluation. Check for CUDA errors or data issues.")
        return None
    print(f"\n[Info] Evaluating Min-{k_percent}% Prob MIA with {len(member_min_k)} member and {len(nonmember_min_k)} non-member samples")
    y_true = [1] * len(member_min_k) + [0] * len(nonmember_min_k)
    min_k_all = member_min_k + nonmember_min_k
    member_mean = np.mean(member_min_k)
    member_std = np.std(member_min_k)
    nonmember_mean = np.mean(nonmember_min_k)
    nonmember_std = np.std(nonmember_min_k)
    print(f"\n[Statistics] (Min-{k_percent}% Prob)")
    print(f"  Member min-k% prob:     {member_mean:.4f} ± {member_std:.4f}")
    print(f"  Non-member min-k% prob: {nonmember_mean:.4f} ± {nonmember_std:.4f}")
    print(f"  Difference:             {abs(member_mean - nonmember_mean):.4f}")
    auc = roc_auc_score(y_true, min_k_all)
    fpr_list, tpr_list, roc_thresholds = roc_curve(y_true, min_k_all)
    def get_tpr_at_fpr(target_fpr):
        idx = np.argmin(np.abs(fpr_list - target_fpr))
        return tpr_list[idx], fpr_list[idx]
    tpr_at_1fpr, actual_fpr_1 = get_tpr_at_fpr(0.01)
    tpr_at_5fpr, actual_fpr_5 = get_tpr_at_fpr(0.05)
    tpr_at_10fpr, actual_fpr_10 = get_tpr_at_fpr(0.10)
    print(f"\n[MIA Evaluation Results] (Min-{k_percent}% Prob)")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  TPR@1%FPR:         {tpr_at_1fpr:.4f} (at FPR={actual_fpr_1:.4f})")
    print(f"  TPR@5%FPR:         {tpr_at_5fpr:.4f} (at FPR={actual_fpr_5:.4f})")
    print(f"  TPR@10%FPR:        {tpr_at_10fpr:.4f} (at FPR={actual_fpr_10:.4f})")
    threshold_viz = (member_mean + nonmember_mean) / 2
    y_pred = [1 if p > threshold_viz else 0 for p in min_k_all]
    accuracy_viz = accuracy_score(y_true, y_pred)
    print(f"\n[Optional] Accuracy with median threshold ({threshold_viz:.4f}): {accuracy_viz:.4f}")
    os.makedirs(output_dir, exist_ok=True)
    detailed_results = {
        'model_id': MODEL_ID,
        'method': f'min_{k_percent}_prob',
        'auc_roc': float(auc),
        'tpr_at_1fpr': float(tpr_at_1fpr),
        'actual_fpr_1': float(actual_fpr_1),
        'tpr_at_5fpr': float(tpr_at_5fpr),
        'actual_fpr_5': float(actual_fpr_5),
        'tpr_at_10fpr': float(tpr_at_10fpr),
        'actual_fpr_10': float(actual_fpr_10),
        'visualization_threshold': float(threshold_viz),
        'visualization_accuracy': float(accuracy_viz),
        'num_member_samples': len(member_min_k),
        'num_nonmember_samples': len(nonmember_min_k),
        'statistics': {
            'member_mean': float(member_mean),
            'member_std': float(member_std),
            'nonmember_mean': float(nonmember_mean),
            'nonmember_std': float(nonmember_std),
            'difference': float(abs(member_mean - nonmember_mean))
        },
        'member_min_k_probs': [float(p) for p in member_min_k],
        'nonmember_min_k_probs': [float(p) for p in nonmember_min_k]
    }
    results_file = os.path.join(output_dir, f"min_{k_percent}_prob_mia_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Info] Detailed results saved to {results_file}")
    # Plot distributions
    plt.figure(figsize=(12, 6))
    bins = 30
    plt.hist(member_min_k, bins=bins, alpha=0.5, label='Member (train2017)', color='blue', density=True)
    plt.hist(nonmember_min_k, bins=bins, alpha=0.5, label='Non-member (val2017)', color='red', density=True)
    plt.axvline(x=threshold_viz, color='green', linestyle='--', linewidth=2,
               label=f'Median Threshold: {threshold_viz:.4f} (visualization only)')
    plt.xlabel(f'Min-{k_percent}% Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Min-{k_percent}% Probability Distribution: Member vs Non-member', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plot_file = os.path.join(output_dir, f"min_{k_percent}_prob_distribution.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Info] Min-{k_percent}% prob distribution plot saved to {plot_file}")
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, min_k_all)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve for Min-{k_percent}% Prob MIA', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plot_file = os.path.join(output_dir, f"min_{k_percent}_prob_roc_curve.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Info] Min-{k_percent}% prob ROC curve plot saved to {plot_file}")
    return detailed_results

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Min-k% Probability Membership Inference Attack (MIA)")
    print("=" * 80)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
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
    k_percent = 10  # You can change this value as needed
    print(f"\n[Step 3] Computing member min-{k_percent}% probabilities...")
    member_min_k_probs, _ = compute_min_k_probs_batch(
        model, processor, member_data, DEVICE, k_percent=k_percent
    )
    print(f"\n[Step 4] Computing non-member min-{k_percent}% probabilities...")
    nonmember_min_k_probs, _ = compute_min_k_probs_batch(
        model, processor, nonmember_data, DEVICE, k_percent=k_percent
    )
    print(f"\n[Step 5] Evaluating Min-{k_percent}% Probability MIA...")
    min_k_results = evaluate_min_k_mia(member_min_k_probs, nonmember_min_k_probs, OUTPUT_DIR, k_percent)
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
