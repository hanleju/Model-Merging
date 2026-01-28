"""
Perplexity-based Membership Inference Attack for Vision-Language Models
Based on: "Privacy risk in machine learning: Analyzing the connection to overfitting"

Basic MIA using perplexity (likelihood) of ground truth captions:
- Member data: Lower perplexity (model is confident on training data)
- Non-member data: Higher perplexity (model is uncertain on unseen data)
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
    "model_id": "google/paligemma-3b-pt-224",
    "merged_model_path": "./models/merge_weights/pali_vqa_caption",  # merge_vlm.py output dir
    "is_lora_adapter": False
}
MODEL_ID = MODEL_CONFIG["model_id"]
DATA_ROOT = "D:/datasets/coco/"
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train2017")
VAL_IMAGES_DIR = os.path.join(DATA_ROOT, "val2017")
TRAIN_ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_train2017.json")
VAL_ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_val2017.json")
OUTPUT_DIR = "./attack_results/eval/0128/pali_merge_perplexity"

# MIA parameters
NUM_EVAL_SAMPLES = 500  # Number of samples to evaluate (per class)
PROMPT = "Describe this image."  # PaliGemma captioning prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# -----------------------------------------------------------------------------
# Data Loading
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
                    "captions": captions  # Ground truth captions
                })
    
    print(f"[Info] Loaded {len(formatted_data)} images with captions")
    
    # Shuffle and sample
    if num_samples:
        random.seed(seed)
        random.shuffle(formatted_data)
        formatted_data = formatted_data[:num_samples]
        print(f"[Info] Sampled {len(formatted_data)} images for evaluation")
    
    return formatted_data

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------

def load_model_and_processor(model_id):
    """Load PaliGemma model and processor."""
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
        # adapter_config.json 있으면 삭제 (PEFT 오작동 방지)
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
# Perplexity Computation
# -----------------------------------------------------------------------------

def compute_perplexity(model, processor, image_path, caption, device):
    """
    Compute perplexity of a ground truth caption given an image.
    
    Perplexity = exp(average cross-entropy loss)
    Lower perplexity = model is more confident (likely member)
    Higher perplexity = model is less confident (likely non-member)
    
    Returns:
        perplexity: Perplexity value
        loss: Average cross-entropy loss per token
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare inputs with ground truth caption
        prompt_with_image = f"<image>{PROMPT}"
        inputs = processor(
            text=prompt_with_image,
            images=image,
            return_tensors="pt"
        ).to(device)
        
        # Tokenize ground truth caption (target)
        # Add special tokens to match model's expected format
        caption_tokens = processor.tokenizer(
            caption,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(device)
        
        # Prepare labels: -100 for prompt tokens (ignore), caption tokens for loss
        # PaliGemma format: prompt tokens should be ignored in loss computation
        labels = torch.cat([
            torch.full((1, inputs.input_ids.shape[1]), -100, dtype=torch.long).to(device),
            caption_tokens
        ], dim=1)
        
        # Concatenate input_ids with caption tokens
        full_input_ids = torch.cat([inputs.input_ids, caption_tokens], dim=1)
        
        # Prepare attention mask
        attention_mask = torch.ones_like(full_input_ids)
        
        # Forward pass to compute loss
        with torch.no_grad():
            outputs = model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                pixel_values=inputs.pixel_values,
                labels=labels
            )
        
        # Extract loss (cross-entropy)
        loss = outputs.loss.item()
        
        # Compute perplexity
        perplexity = np.exp(loss)
        
        return perplexity, loss
        
    except Exception as e:
        print(f"[Error] Failed to compute perplexity for {image_path}: {e}")
        return None, None

def compute_perplexities_batch(model, processor, data_samples, device):
    """
    Compute perplexities for a batch of samples using their ground truth captions.
    
    Returns:
        perplexities: List of perplexity values
        losses: List of loss values
    """
    perplexities = []
    losses = []
    
    print(f"[Info] Computing perplexities for {len(data_samples)} samples...")
    
    for sample in tqdm(data_samples, desc="Computing perplexities"):
        # Use first ground truth caption
        caption = sample["captions"][0]
        
        perplexity, loss = compute_perplexity(
            model, processor, sample["image_path"], caption, device
        )
        
        if perplexity is not None:
            perplexities.append(perplexity)
            losses.append(loss)
        else:
            # If computation failed, append None
            perplexities.append(None)
            losses.append(None)
    
    return perplexities, losses

# -----------------------------------------------------------------------------
# MIA Evaluation
# -----------------------------------------------------------------------------

def evaluate_mia(member_perplexities, nonmember_perplexities, output_dir):
    """
    Evaluate MIA performance using perplexity-based classification.
    
    Args:
        member_perplexities: List of perplexity values for member samples
        nonmember_perplexities: List of perplexity values for non-member samples
        output_dir: Directory to save results
    
    Returns:
        results: Dictionary of evaluation metrics
    """
    # Filter out None values (failed computations)
    member_perplexities = [p for p in member_perplexities if p is not None]
    nonmember_perplexities = [p for p in nonmember_perplexities if p is not None]
    
    print(f"\n[Info] Evaluating MIA with {len(member_perplexities)} member samples "
          f"and {len(nonmember_perplexities)} non-member samples")
    
    # Prepare labels
    # Member = 1, Non-member = 0
    y_true = [1] * len(member_perplexities) + [0] * len(nonmember_perplexities)
    perplexities = member_perplexities + nonmember_perplexities
    
    # Compute statistics
    member_mean = np.mean(member_perplexities)
    member_std = np.std(member_perplexities)
    nonmember_mean = np.mean(nonmember_perplexities)
    nonmember_std = np.std(nonmember_perplexities)
    
    print(f"\n[Statistics] (Perplexity)")
    print(f"  Member perplexity:     {member_mean:.4f} ± {member_std:.4f}")
    print(f"  Non-member perplexity: {nonmember_mean:.4f} ± {nonmember_std:.4f}")
    print(f"  Difference:            {abs(member_mean - nonmember_mean):.4f}")
    
    # Compute AUC-ROC (threshold-free metric)
    # Note: We use perplexity directly (higher perplexity = higher score for non-member)
    # So we negate for member classification
    auc = roc_auc_score(y_true, [-p for p in perplexities])
    
    # Compute ROC curve for TPR@FPR metrics
    fpr_list, tpr_list, roc_thresholds = roc_curve(y_true, [-p for p in perplexities])
    
    # Compute TPR@FPR metrics
    def get_tpr_at_fpr(target_fpr):
        idx = np.argmin(np.abs(fpr_list - target_fpr))
        return tpr_list[idx], fpr_list[idx]
    
    tpr_at_1fpr, actual_fpr_1 = get_tpr_at_fpr(0.01)
    tpr_at_5fpr, actual_fpr_5 = get_tpr_at_fpr(0.05)
    tpr_at_10fpr, actual_fpr_10 = get_tpr_at_fpr(0.10)
    
    # Print threshold-free metrics
    print(f"\n[MIA Evaluation Results] (Threshold-Free Metrics)")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  TPR@1%FPR:         {tpr_at_1fpr:.4f} (at FPR={actual_fpr_1:.4f})")
    print(f"  TPR@5%FPR:         {tpr_at_5fpr:.4f} (at FPR={actual_fpr_5:.4f})")
    print(f"  TPR@10%FPR:        {tpr_at_10fpr:.4f} (at FPR={actual_fpr_10:.4f})")
    
    # Optional: Compute threshold for visualization (using median of means)
    threshold_viz = (member_mean + nonmember_mean) / 2
    y_pred = [1 if p < threshold_viz else 0 for p in perplexities]
    accuracy_viz = accuracy_score(y_true, y_pred)
    
    print(f"\n[Optional] Accuracy with median threshold ({threshold_viz:.4f}): {accuracy_viz:.4f}")
    print(f"  (Note: This is for reference only - not the main evaluation metric)")
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_results = {
        'model_id': MODEL_ID,
        'method': 'perplexity',
        'auc_roc': float(auc),
        'tpr_at_1fpr': float(tpr_at_1fpr),
        'actual_fpr_1': float(actual_fpr_1),
        'tpr_at_5fpr': float(tpr_at_5fpr),
        'actual_fpr_5': float(actual_fpr_5),
        'tpr_at_10fpr': float(tpr_at_10fpr),
        'actual_fpr_10': float(actual_fpr_10),
        'visualization_threshold': float(threshold_viz),
        'visualization_accuracy': float(accuracy_viz),
        'num_member_samples': len(member_perplexities),
        'num_nonmember_samples': len(nonmember_perplexities),
        'statistics': {
            'member_mean': float(member_mean),
            'member_std': float(member_std),
            'nonmember_mean': float(nonmember_mean),
            'nonmember_std': float(nonmember_std),
            'difference': float(abs(member_mean - nonmember_mean))
        },
        'member_perplexities': [float(p) for p in member_perplexities],
        'nonmember_perplexities': [float(p) for p in nonmember_perplexities]
    }
    
    results_file = os.path.join(output_dir, "perplexity_mia_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Info] Detailed results saved to {results_file}")
    
    # Plot distributions
    plot_perplexity_distributions(
        member_perplexities, nonmember_perplexities, 
        threshold_viz, output_dir
    )
    
    # Plot ROC curve
    plot_roc_curve(y_true, perplexities, auc, output_dir)
    
    return detailed_results

def plot_perplexity_distributions(member_perplexities, nonmember_perplexities, 
                                   threshold, output_dir):
    """Plot perplexity distributions for members and non-members."""
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    bins = 30
    plt.hist(member_perplexities, bins=bins, alpha=0.5, label='Member (train2017)', 
             color='blue', density=True)
    plt.hist(nonmember_perplexities, bins=bins, alpha=0.5, label='Non-member (val2017)', 
             color='red', density=True)
    
    # Plot threshold line (for visualization only)
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
               label=f'Median Threshold: {threshold:.4f} (visualization only)')
    
    plt.xlabel('Perplexity', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Perplexity Distribution: Member vs Non-member', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_file = os.path.join(output_dir, "perplexity_distribution.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] Perplexity distribution plot saved to {plot_file}")

def plot_roc_curve(y_true, perplexities, auc, output_dir):
    """Plot ROC curve."""
    # Compute ROC curve
    # Note: We negate perplexity because lower perplexity = member (positive class)
    fpr, tpr, _ = roc_curve(y_true, [-p for p in perplexities])
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, 
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Perplexity-based MIA', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plot_file = os.path.join(output_dir, "perplexity_roc_curve.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] ROC curve plot saved to {plot_file}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    """Main execution function."""
    print("=" * 80)
    print("Perplexity-based Membership Inference Attack (MIA)")
    print("=" * 80)
    
    # Set random seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Load model and processor
    model, processor = load_model_and_processor(MODEL_ID)
    
    # Load data
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
    
    # Compute perplexities
    print("\n[Step 3] Computing perplexities...")
    
    print("\n[3.1] Computing member perplexities...")
    member_perplexities, member_losses = compute_perplexities_batch(
        model, processor, member_data, DEVICE
    )
    
    print("\n[3.2] Computing non-member perplexities...")
    nonmember_perplexities, nonmember_losses = compute_perplexities_batch(
        model, processor, nonmember_data, DEVICE
    )
    
    # Evaluate MIA
    print("\n[Step 4] Evaluating membership inference attack...")
    results = evaluate_mia(member_perplexities, nonmember_perplexities, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
