"""
Membership Inference Attack (MIA) Evaluation
Using Entropy-based Classification

Goal: Evaluate MIA performance on PaliGemma model by classifying samples
      as members (train2017) or non-members (val2017) based on generation entropy.
"""

import os
import json
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
from peft import PeftModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Model selection: "pali" or "smol"
MODEL = "pali"  # "pali" for PaliGemma, "smol" for SmolVLM

# Model configurations
MODEL_CONFIGS = {
    "pali": {
        "model_id": "google/paligemma-3b-pt-224",
        "model_class": PaliGemmaForConditionalGeneration,
        "adapter_path": False,  # Set to merged model path if using merge_vlm.py output
        "merged_model_path": "./models/merge_weights/pali_vqa_caption",  # Path to merged model
        "use_image_token": True,  # PaliGemma requires <image> token
        "quantization": "8bit",
        "is_lora_adapter": False  # False if using full merged model from merge_vlm.py
    },
    "smol": {
        "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "model_class": AutoModelForImageTextToText,
        "adapter_path": "./models/smolvlm-caption-finetune_3",  # LoRA adapter path
        "use_image_token": False,  # SmolVLM uses chat template
        "quantization": "4bit",
        "is_lora_adapter": True  # True if using LoRA adapter
    }
}

MODEL_ID = MODEL_CONFIGS[MODEL]["model_id"]
DATA_ROOT = "D:/datasets/coco/"
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train2017")
VAL_IMAGES_DIR = os.path.join(DATA_ROOT, "val2017")
TRAIN_ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_train2017.json")
VAL_ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_val2017.json")
OUTPUT_DIR = "./attack_results/eval/0128/pali_merge_optimize"

# Attack parameters
PROMPT = "Describe breach Remedies pige."  # Prompt to use for MIA evaluation

# Renyi entropy parameter (α=2 for collision entropy, as used in NeurIPS 2024 paper)
# Paper tests α = 0.5, 1, 2, ∞ and K = 0, 10, 100
RENYI_ALPHA = 2.0

# MaxRényi-K%: Top K% tokens with largest Rényi entropies
# K=0: max entropy only, K=100: average all tokens
MAX_K_PERCENT = 100  # Use average entropy (K=100)

NUM_EVAL_SAMPLES = 500  # Number of samples to evaluate (per class)
MAX_NEW_TOKENS = 40
TEMPERATURE = 1.0  # Use sampling to compute entropy
DO_SAMPLE = True
TOP_P = 1.0
TOP_K = 0

BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_coco_data(annotation_file, images_dir, num_samples=None, seed=42):
    """Load COCO caption data."""
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

def load_model_and_processor(model_type):
    """Load model and processor based on model type."""
    config = MODEL_CONFIGS[model_type]
    model_id = config["model_id"]
    is_lora_adapter = config.get("is_lora_adapter", False)
    merged_model_path = config.get("merged_model_path", None)
    adapter_path = config.get("adapter_path", None)

    print(f"[Info] Loading model: {model_id} (type: {model_type})")

    if is_lora_adapter:
        # LoRA adapter: load base model, then adapter
        load_path = model_id
        print(f"[Info] Will load LoRA adapter from: {adapter_path}")
    elif merged_model_path and os.path.exists(merged_model_path):
        # Merged model: load directly
        load_path = merged_model_path
        print(f"[Info] Using merged model from: {merged_model_path}")
    else:
        # No adapter or merged model: use base model
        load_path = model_id
        if merged_model_path:
            print(f"[Warning] Merged model path not found: {merged_model_path}")
        print(f"[Info] Using base model")

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(
            merged_model_path if (not is_lora_adapter and merged_model_path and os.path.exists(merged_model_path))
            else (adapter_path if is_lora_adapter and adapter_path and os.path.exists(adapter_path) else model_id)
        )
        print(f"[Info] Loaded processor")
    except:
        processor = AutoProcessor.from_pretrained(model_id)
        print(f"[Info] Loaded processor from base model")

    # Configure quantization based on model
    if config["quantization"] == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    else:  # 4bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )



    # Load model
    model_class = config["model_class"]
    try:
        if not is_lora_adapter and merged_model_path and os.path.exists(merged_model_path):
            # Merged model: load base model, but use merged weights
            safetensors_path = os.path.join(merged_model_path, "model.safetensors")
            bin_path = os.path.join(merged_model_path, "pytorch_model.bin")
            if os.path.exists(safetensors_path):
                weights_path = safetensors_path
            elif os.path.exists(bin_path):
                weights_path = bin_path
            else:
                raise FileNotFoundError(f"Merged model weights not found: {safetensors_path} or {bin_path}")
            print(f"[Info] Loading merged weights from: {weights_path}")
            base_model = model_class.from_pretrained(
                merged_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if config["quantization"] == "4bit" else torch.float16
            )
        else:
            base_model = model_class.from_pretrained(
                load_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if config["quantization"] == "4bit" else torch.float16
            )
    except Exception as e:
        print(f"[Error] Failed to load model from {load_path} with model_class {model_class}: {e}")
        print("[Hint] If this is a merged model, ensure model_class is correct (e.g., AutoModelForXXX). If this is a LoRA adapter, check adapter_path and is_lora_adapter.")
        raise


    # Only use PeftModel for LoRA adapters
    if is_lora_adapter and adapter_path and os.path.exists(adapter_path):
        try:
            print(f"[Info] Loading LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model = model.merge_and_unload()
            print("[Info] Adapter loaded and merged")
        except Exception as e:
            print(f"[Error] Failed to load LoRA adapter: {e}")
            raise
    else:
        model = base_model

    model.eval()
    print(f"[Info] Model loaded successfully")

    return model, processor

# -----------------------------------------------------------------------------
# Entropy Computation
# -----------------------------------------------------------------------------

def compute_generation_entropy(model, processor, image_path, prompt, device, model_type):
    """
    Compute Renyi entropy of the generated caption distribution.
    Using α=2 (collision entropy) as per NeurIPS 2024 paper:
    H_α(X) = 1/(1-α) * log(Σ p_i^α)
    
    Returns:
        entropy: Average Renyi entropy per token
        caption: Generated caption (for reference)
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare inputs based on model type
        config = MODEL_CONFIGS[model_type]
        
        if config["use_image_token"]:
            # PaliGemma: requires <image> token
            prompt_with_image = f"<image>{prompt}"
            inputs = processor(
                text=prompt_with_image,
                images=image,
                return_tensors="pt"
            ).to(device)
        else:
            # SmolVLM: uses chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)
        
        # Generate with output scores to compute entropy
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K if TOP_K > 0 else None,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else None,
                eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None
            )
        
        # Compute Renyi entropy from logits
        # outputs.scores is a tuple of tensors, each of shape (batch_size, vocab_size)
        entropies = []
        for score in outputs.scores:
            # Convert logits to probabilities
            probs = torch.softmax(score, dim=-1)
            # Renyi entropy: H_α(X) = 1/(1-α) * log(Σ p_i^α)
            if RENYI_ALPHA == 2.0:
                # Collision entropy (α=2): H_2(X) = -log(Σ p_i^2)
                entropy = -torch.log(torch.sum(probs ** 2, dim=-1) + 1e-10)
            else:
                # General Renyi entropy
                entropy = (1 / (1 - RENYI_ALPHA)) * torch.log(
                    torch.sum(probs ** RENYI_ALPHA, dim=-1) + 1e-10
                )
            entropies.append(entropy.item())
        
        # Average Renyi entropy across all generated tokens
        avg_entropy = np.mean(entropies) if entropies else 0.0
        
        # Decode generated caption
        generated_ids = outputs.sequences
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        # Remove prompt from caption
        caption = caption.replace(prompt, "").strip()
        
        return avg_entropy, caption
        
    except Exception as e:
        print(f"[Error] Failed to process image {image_path}: {e}")
        return None, None

def compute_entropies_batch(model, processor, data_samples, prompt, device, model_type):
    """
    Compute entropies for a batch of samples.
    
    Returns:
        entropies: List of entropy values
        captions: List of generated captions
    """
    entropies = []
    captions = []
    
    print(f"[Info] Computing entropies for {len(data_samples)} samples...")
    
    for sample in tqdm(data_samples, desc="Computing entropies"):
        entropy, caption = compute_generation_entropy(
            model, processor, sample["image_path"], prompt, device, model_type
        )
        
        if entropy is not None:
            entropies.append(entropy)
            captions.append(caption)
        else:
            # If computation failed, append None
            entropies.append(None)
            captions.append(None)
    
    return entropies, captions

# -----------------------------------------------------------------------------
# MIA Evaluation
# -----------------------------------------------------------------------------

def evaluate_mia(member_entropies, nonmember_entropies, output_dir):
    """
    Evaluate MIA performance using threshold-free metrics (as in NeurIPS 2024 paper).
    
    The paper does not use fixed thresholds, but instead evaluates with:
    - AUC-ROC: Area under ROC curve (threshold-independent)
    - TPR@FPR: True Positive Rate at specific False Positive Rate
    
    Args:
        member_entropies: List of Rényi entropy values for member samples
        nonmember_entropies: List of Rényi entropy values for non-member samples
        output_dir: Directory to save results
    
    Returns:
        results: Dictionary of evaluation metrics
    """
    # Filter out None values (failed computations)
    member_entropies = [e for e in member_entropies if e is not None]
    nonmember_entropies = [e for e in nonmember_entropies if e is not None]
    
    print(f"\n[Info] Evaluating MIA with {len(member_entropies)} member samples "
          f"and {len(nonmember_entropies)} non-member samples")
    
    # Prepare labels
    # Member = 1, Non-member = 0
    y_true = [1] * len(member_entropies) + [0] * len(nonmember_entropies)
    entropies = member_entropies + nonmember_entropies
    
    # Compute statistics
    member_mean = np.mean(member_entropies)
    member_std = np.std(member_entropies)
    nonmember_mean = np.mean(nonmember_entropies)
    nonmember_std = np.std(nonmember_entropies)
    
    print(f"\n[Statistics] (Rényi Entropy, α={RENYI_ALPHA}, K={MAX_K_PERCENT}%)")
    print(f"  Member entropy:     {member_mean:.4f} ± {member_std:.4f}")
    print(f"  Non-member entropy: {nonmember_mean:.4f} ± {nonmember_std:.4f}")
    print(f"  Difference:         {abs(member_mean - nonmember_mean):.4f}")
    
    # Compute AUC-ROC (threshold-free metric)
    # Note: We negate entropy for AUC computation (lower entropy = higher score for member)
    auc = roc_auc_score(y_true, [-e for e in entropies])
    
    # Compute ROC curve for TPR@FPR metrics
    fpr_list, tpr_list, roc_thresholds = roc_curve(y_true, [-e for e in entropies])
    
    # Compute TPR@FPR metrics (as used in the paper)
    def get_tpr_at_fpr(target_fpr):
        idx = np.argmin(np.abs(fpr_list - target_fpr))
        return tpr_list[idx], fpr_list[idx]
    
    tpr_at_1fpr, actual_fpr_1 = get_tpr_at_fpr(0.01)
    tpr_at_5fpr, actual_fpr_5 = get_tpr_at_fpr(0.05)
    tpr_at_10fpr, actual_fpr_10 = get_tpr_at_fpr(0.10)
    
    # Print threshold-free metrics (main results as in paper)
    print(f"\n[MIA Evaluation Results] (Threshold-Free Metrics)")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  TPR@1%FPR:         {tpr_at_1fpr:.4f} (at FPR={actual_fpr_1:.4f})")
    print(f"  TPR@5%FPR:         {tpr_at_5fpr:.4f} (at FPR={actual_fpr_5:.4f})")
    print(f"  TPR@10%FPR:        {tpr_at_10fpr:.4f} (at FPR={actual_fpr_10:.4f})")
    
    # Optional: Compute threshold for visualization (using median of means)
    threshold_viz = (member_mean + nonmember_mean) / 2
    y_pred = [1 if e < threshold_viz else 0 for e in entropies]
    accuracy_viz = accuracy_score(y_true, y_pred)
    
    print(f"\n[Optional] Accuracy with median threshold ({threshold_viz:.4f}): {accuracy_viz:.4f}")
    print(f"  (Note: This is for reference only - not the main evaluation metric)")
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_results = {
        'model_id': MODEL_ID,
        'prompt': PROMPT,
        'renyi_alpha': RENYI_ALPHA,
        'max_k_percent': MAX_K_PERCENT,
        'auc_roc': float(auc),
        'tpr_at_1fpr': float(tpr_at_1fpr),
        'actual_fpr_1': float(actual_fpr_1),
        'tpr_at_5fpr': float(tpr_at_5fpr),
        'actual_fpr_5': float(actual_fpr_5),
        'tpr_at_10fpr': float(tpr_at_10fpr),
        'actual_fpr_10': float(actual_fpr_10),
        'visualization_threshold': float(threshold_viz),
        'visualization_accuracy': float(accuracy_viz),
        'num_member_samples': len(member_entropies),
        'num_nonmember_samples': len(nonmember_entropies),
        'statistics': {
            'member_mean': float(member_mean),
            'member_std': float(member_std),
            'nonmember_mean': float(nonmember_mean),
            'nonmember_std': float(nonmember_std),
            'difference': float(abs(member_mean - nonmember_mean))
        },
        'member_entropies': [float(e) for e in member_entropies],
        'nonmember_entropies': [float(e) for e in nonmember_entropies]
    }
    
    results_file = os.path.join(output_dir, "mia_evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Info] Detailed results saved to {results_file}")
    
    # Plot distributions
    plot_entropy_distributions(
        member_entropies, nonmember_entropies, 
        threshold_viz, output_dir
    )
    
    # Plot ROC curve
    plot_roc_curve(y_true, entropies, auc, output_dir)
    
    return detailed_results

def plot_entropy_distributions(member_entropies, nonmember_entropies, 
                                threshold, output_dir):
    """Plot entropy distributions for members and non-members."""
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    bins = 30
    plt.hist(member_entropies, bins=bins, alpha=0.5, label='Member (train2017)', 
             color='blue', density=True)
    plt.hist(nonmember_entropies, bins=bins, alpha=0.5, label='Non-member (val2017)', 
             color='red', density=True)
    
    # Plot threshold line (for visualization only)
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
               label=f'Median Threshold: {threshold:.4f} (visualization only)')
    
    plt.xlabel(f'Renyi Entropy (α={RENYI_ALPHA})', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Renyi Entropy Distribution: Member vs Non-member', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_file = os.path.join(output_dir, "entropy_distribution.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] Entropy distribution plot saved to {plot_file}")

def plot_roc_curve(y_true, entropies, auc, output_dir):
    """Plot ROC curve."""
    # Compute ROC curve
    # Note: We negate entropy because lower entropy = member (positive class)
    fpr, tpr, _ = roc_curve(y_true, [-e for e in entropies])
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, 
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Membership Inference Attack', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plot_file = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] ROC curve plot saved to {plot_file}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    """Main execution function."""
    print("=" * 80)
    print("Membership Inference Attack (MIA) Evaluation")
    print("=" * 80)
    
    # Set random seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Print model info
    print(f"\n[Model] {MODEL.upper()}: {MODEL_ID}")
    
    # Load model and processor
    model, processor = load_model_and_processor(MODEL)
    
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
    
    # Select prompt
    prompt = PROMPT
    
    print(f"\n[Prompt] '{prompt}'")
    print(f"[Entropy Type] MaxRényi-{MAX_K_PERCENT}% (α={RENYI_ALPHA})")
    
    # Compute entropies
    print("\n[Step 3] Computing entropies for member samples...")
    
    print("\n[3.1] Computing member entropies...")
    member_entropies, member_captions = compute_entropies_batch(
        model, processor, member_data, prompt, DEVICE, MODEL
    )
    
    print("\n[3.2] Computing non-member entropies...")
    nonmember_entropies, nonmember_captions = compute_entropies_batch(
        model, processor, nonmember_data, prompt, DEVICE, MODEL
    )
    
    # Evaluate MIA
    print("\n[Step 4] Evaluating membership inference attack...")
    results = evaluate_mia(member_entropies, nonmember_entropies, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
