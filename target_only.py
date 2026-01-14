"""
Target-Only Membership Inference Attack for Vision-Language Models
Uses FairFace (member) and UTKFace (non-member) datasets
"""

import torch
import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from typing import Dict, List, Tuple
import json


# =========================================================
# Prompt Template
# =========================================================

PROMPT_TEMPLATES = {
    'age': '<image>What is the age group of the person?',
    'gender': '<image>What is the gender of the person?',
    'race': '<image>What is the race of the person?'
}


# =========================================================
# Model Functions
# =========================================================

def load_model_and_processor(model_path: str, device: str = "cuda"):
    """Load PaliGemma model and processor."""
    print(f"📥 Loading model from: {model_path}")
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    
    try:
        processor = PaliGemmaProcessor.from_pretrained(model_path)
    except (OSError, ValueError):
        print("⚠️  Processor not found, using base PaliGemma processor...")
        processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")
    
    print("✅ Model loaded successfully!\n")
    return model, processor


# =========================================================
# Data Loading Functions
# =========================================================

def load_dataset(data_dir: str, split: str = "val", max_samples: int = None) -> pd.DataFrame:
    """Load dataset from FairFace format."""
    csv_path = os.path.join(data_dir, "labels", f"{split}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['image_path'] = df['file'].apply(lambda x: os.path.join(data_dir, "images", x))
    df = df[df['image_path'].apply(os.path.exists)]
    
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    
    return df


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
    prompt = PROMPT_TEMPLATES[attribute]
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
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


# =========================================================
# Main Function
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Target-Only Membership Inference Attack using FairFace and UTKFace"
    )
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model or HuggingFace model ID")
    
    # Data arguments
    parser.add_argument("--member_data_dir", type=str, required=True,
                       help="FairFace directory (member data)")
    parser.add_argument("--non_member_data_dir", type=str, required=True,
                       help="UTKFace directory (non-member data)")
    parser.add_argument("--member_split", type=str, default="train",
                       help="FairFace split for members (default: train)")
    parser.add_argument("--non_member_split", type=str, default="val",
                       help="UTKFace split for non-members (default: val)")
    parser.add_argument("--member_samples", type=int, default=1000,
                       help="Number of member samples (default: 1000)")
    parser.add_argument("--non_member_samples", type=int, default=1000,
                       help="Number of non-member samples (default: 1000)")
    
    # Attack parameters
    parser.add_argument("--attribute", type=str, default="age",
                       choices=["age", "gender", "race"],
                       help="Attribute to evaluate (default: age)")
    parser.add_argument("--temperature_low", type=float, default=0.1,
                       help="Low temperature (default: 0.1)")
    parser.add_argument("--temperature_high", type=float, default=1.6,
                       help="High temperature (default: 1.6)")
    parser.add_argument("--granularity", type=int, default=50,
                       help="Samples per test (default: 50)")
    parser.add_argument("--metric", type=str, default="f1",
                       choices=["exact_match", "contains", "precision", "recall", "f1"],
                       help="Similarity metric (default: f1)")
    parser.add_argument("--num_runs", type=int, default=5,
                       help="Number of attack runs (default: 5)")
    parser.add_argument("--num_tests", type=int, default=1000,
                       help="Number of tests per run (default: 1000)")
    
    # Output
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory (default: output)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (default: cuda)")
    
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = "cpu"
    
    print("🎯 Target-Only Membership Inference Attack")
    print("="*60)
    print(f"Model           : {args.model_path}")
    print(f"Member data     : {args.member_data_dir}")
    print(f"Non-member data : {args.non_member_data_dir}")
    print(f"Attribute       : {args.attribute}")
    print(f"Temperature     : [{args.temperature_low}, {args.temperature_high}]")
    print(f"Granularity     : {args.granularity}")
    print(f"Metric          : {args.metric}")
    print("="*60 + "\n")
    
    # Load model
    model, processor = load_model_and_processor(args.model_path, args.device)
    
    # Load datasets
    print(f"📂 Loading member data (FairFace)...")
    member_df = load_dataset(args.member_data_dir, args.member_split, args.member_samples)
    print(f"   Loaded {len(member_df)} member samples\n")
    
    print(f"📂 Loading non-member data (UTKFace)...")
    non_member_df = load_dataset(args.non_member_data_dir, args.non_member_split, args.non_member_samples)
    print(f"   Loaded {len(non_member_df)} non-member samples\n")
    
    # Compute similarities
    temperatures = [args.temperature_low, args.temperature_high]
    
    member_similarities = compute_dataset_similarities(
        model, processor, member_df, args.attribute, temperatures, args.device, "Members"
    )
    
    non_member_similarities = compute_dataset_similarities(
        model, processor, non_member_df, args.attribute, temperatures, args.device, "Non-members"
    )
    
    # Run attack
    print(f"\n🔄 Running membership inference attack ({args.num_runs} runs)...")
    aucs = []
    attack_accs = []
    tpr_at_5fprs = []
    
    for run_idx in range(args.num_runs):
        auc, attack_acc, tpr_at_5fpr = target_only_inference(
            member_similarities,
            non_member_similarities,
            args.granularity,
            args.temperature_low,
            args.temperature_high,
            args.metric,
            args.num_tests
        )
        aucs.append(auc)
        attack_accs.append(attack_acc)
        tpr_at_5fprs.append(tpr_at_5fpr)
        print(f"   Run {run_idx + 1}/{args.num_runs}: AUC={auc:.4f}, Acc={attack_acc:.4f}, TPR@5%FPR={tpr_at_5fpr:.4f}")
    
    # Results
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    avg_acc = np.mean(attack_accs)
    std_acc = np.std(attack_accs)
    avg_tpr = np.mean(tpr_at_5fprs)
    std_tpr = np.std(tpr_at_5fprs)
    
    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print("="*60)
    print(f"AUC:         {avg_auc:.4f} ± {std_auc:.4f}  (min: {min(aucs):.4f}, max: {max(aucs):.4f})")
    print(f"Attack Acc:  {avg_acc:.4f} ± {std_acc:.4f}  (min: {min(attack_accs):.4f}, max: {max(attack_accs):.4f})")
    print(f"TPR@5%FPR:   {avg_tpr:.4f} ± {std_tpr:.4f}  (min: {min(tpr_at_5fprs):.4f}, max: {max(tpr_at_5fprs):.4f})")
    print("="*60)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    model_name = args.model_path.replace('/', '_').replace('\\', '_')
    output_file = os.path.join(args.output, f"mia_{model_name}_{args.attribute}.json")
    
    results = {
        'model': args.model_path,
        'parameters': {
            'attribute': args.attribute,
            'member_data': args.member_data_dir,
            'non_member_data': args.non_member_data_dir,
            'member_samples': len(member_df),
            'non_member_samples': len(non_member_df),
            'temperature_low': args.temperature_low,
            'temperature_high': args.temperature_high,
            'granularity': args.granularity,
            'metric': args.metric,
            'num_runs': args.num_runs
        },
        'results': {
            'auc': {
                'values': aucs,
                'mean': float(avg_auc),
                'std': float(std_auc),
                'min': float(min(aucs)),
                'max': float(max(aucs))
            },
            'attack_accuracy': {
                'values': attack_accs,
                'mean': float(avg_acc),
                'std': float(std_acc),
                'min': float(min(attack_accs)),
                'max': float(max(attack_accs))
            },
            'tpr_at_5fpr': {
                'values': tpr_at_5fprs,
                'mean': float(avg_tpr),
                'std': float(std_tpr),
                'min': float(min(tpr_at_5fprs)),
                'max': float(max(tpr_at_5fprs))
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ Attack complete!")


if __name__ == "__main__":
    main()
