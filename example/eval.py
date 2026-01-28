"""
FairFace Dataset Evaluation for PaliGemma VLM Models
Evaluates age, gender, and race prediction accuracy
"""

import torch
import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from typing import Dict, List


# =========================================================
# Constants
# =========================================================

AGE_GROUPS = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
GENDERS = ['Male', 'Female']
# Fair_RACES = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
UTK_RACES = ['White', 'Black', 'East Asian', 'Indian', 'Others']

PROMPT_TEMPLATES = {
    'age':'<image>What is the age group of the person?',
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
        processor = PaliGemmaProcessor.from_pretrained(model_path, use_fast=True)
        print("✅ Model loaded successfully!\n")
    except (OSError, ValueError) as e:
        print("⚠️  Processor not found in model, using base PaliGemma processor...")
        base_processor_path = "google/paligemma-3b-pt-224"
        processor = PaliGemmaProcessor.from_pretrained(base_processor_path, use_fast=True)
        print("✅ Model and base processor loaded successfully!\n")
    
    return model, processor


# =========================================================
# Data Functions
# =========================================================

def load_fairface_data(data_dir: str, split: str = "val") -> pd.DataFrame:
    """Load FairFace dataset from CSV and validate image paths."""
    csv_path = os.path.join(data_dir, "labels", f"{split}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['image_path'] = df['file'].apply(lambda x: os.path.join(data_dir, "images", x))
    df = df[df['image_path'].apply(os.path.exists)]
    
    print(f"📊 Loaded {len(df)} samples from {split} split")
    return df

# =========================================================
# Prediction Functions
# =========================================================

def predict_attribute(
    model,
    processor,
    image: Image.Image,
    attribute: str,
    device: str = "cuda"
) -> str:
    """Predict a single attribute for an image."""
    prompt = PROMPT_TEMPLATES[attribute]

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from generated text (without <image> token)
    prompt_without_image = prompt.replace("<image>", "")
    answer = generated_text.replace(prompt_without_image, "").strip()
    
    return answer


def normalize_answer(answer: str, attribute: str, gt_label: str = None) -> str:
    """Normalize model output to match ground truth format."""
    answer = answer.strip().lower()
    
    if attribute == "age":
        # Try to match exact age group first
        for age_group in AGE_GROUPS:
            if age_group.lower() in answer:
                return age_group
        
        # Handle "from X to Y" format
        import re
        match = re.search(r'from\s+(\d+)\s+to\s+(\d+)', answer)
        if match and gt_label:
            pred_start = int(match.group(1))
            pred_end = int(match.group(2))
            
            # Parse ground truth range
            gt_match = re.match(r'(\d+)-(\d+|\+)', gt_label)
            if gt_match:
                gt_start = int(gt_match.group(1))
                gt_end = gt_match.group(2)
                
                if gt_end == '+':
                    # Handle "70+" case
                    if pred_start >= gt_start:
                        return gt_label
                else:
                    gt_end = int(gt_end)
                    # Check if predicted range overlaps with ground truth
                    if pred_start <= gt_end and pred_end >= gt_start:
                        return gt_label
    
    elif attribute == "gender":
        if "male" in answer and "female" not in answer:
            return "Male"
        elif "female" in answer:
            return "Female"
    
    elif attribute == "race":
        race_map = {
            "white": "White",
            "black": "Black",
            "latino": "Latino_Hispanic",
            "hispanic": "Latino_Hispanic",
            "east asian": "East Asian",
            "southeast asian": "Southeast Asian",
            "indian": "Indian",
            "middle eastern": "Middle Eastern"
        }
        for key, value in race_map.items():
            if key in answer:
                return value
    
    return answer


# =========================================================
# Evaluation Functions
# =========================================================

def evaluate_model(
    model,
    processor,
    data_df: pd.DataFrame,
    attributes: List[str],
    device: str = "cuda",
    max_samples: int = None,
    output_file: str = None
) -> Dict[str, float]:
    """Evaluate model on FairFace dataset."""
    if max_samples:
        data_df = data_df.head(max_samples)
    
    results = {attr: {"correct": 0, "total": 0} for attr in attributes}
    
    print(f"🔄 Starting evaluation on {len(data_df)} samples...")
    print("="*60 + "\n")
    
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Evaluating"):
        try:
            image = Image.open(row['image_path']).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading {row['image_path']}: {e}")
            continue
        
        for attr in attributes:
            gt_label = row[attr]
            pred = predict_attribute(model, processor, image, attr, device)
            pred_normalized = normalize_answer(pred, attr, gt_label)
            
            is_correct = (pred_normalized == gt_label)
            results[attr]["correct"] += int(is_correct)
            results[attr]["total"] += 1
            
            # Show first 3 samples for debugging
            if idx < 3:
                print(f"[Sample {idx+1}] {attr.upper()}: GT={gt_label}, Pred={pred_normalized} {'✓' if is_correct else '✗'}")
    
    return compute_accuracy(results, output_file)


def compute_accuracy(results: Dict[str, Dict[str, int]], output_file: str = None) -> Dict[str, float]:
    """Calculate and display accuracy metrics."""
    output_lines = []
    
    header = "\n" + "="*60 + "\n" + "📊 EVALUATION RESULTS" + "\n" + "="*60
    print(header)
    output_lines.append(header)
    
    accuracies = {}
    total_correct = 0
    total_samples = 0
    
    for attr, res in results.items():
        if res["total"] > 0:
            acc = res["correct"] / res["total"] * 100
            accuracies[attr] = acc
            line = f"{attr.upper():8s}: {acc:6.2f}% ({res['correct']}/{res['total']})"
            print(line)
            output_lines.append(line)
            total_correct += res["correct"]
            total_samples += res["total"]
        else:
            accuracies[attr] = 0.0
            line = f"{attr.upper():8s}: N/A"
            print(line)
            output_lines.append(line)
    
    if total_samples > 0:
        overall_acc = total_correct / total_samples * 100
        line = f"\nOVERALL : {overall_acc:6.2f}% ({total_correct}/{total_samples})"
        print(line)
        output_lines.append(line)
    
    footer = "="*60
    print(footer)
    output_lines.append(footer)
    
    # Save to file if output_file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\n💾 Results saved to: {output_file}")
    
    return accuracies


# =========================================================
# Main Function
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate PaliGemma VLM on FairFace Dataset")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model directory")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Root directory of FairFace dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                       help="Dataset split to evaluate (default: val)")
    parser.add_argument("--attributes", type=str, nargs="+", default=["age", "gender", "race"],
                       choices=["age", "gender", "race"],
                       help="Attributes to evaluate (default: age gender race)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory for results (default: output)")
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = "cpu"
    
    print("🚀 FairFace Evaluation")
    print("="*60)
    print(f"Model      : {args.model_path}")
    print(f"Data       : {args.data_dir}")
    print(f"Split      : {args.split}")
    print(f"Evaluating : {', '.join(args.attributes)}")
    print(f"Device     : {args.device}")
    print("="*60 + "\n")
    
    # Load model
    model, processor = load_model_and_processor(args.model_path, args.device)
    
    # Load data
    data_df = load_fairface_data(args.data_dir, args.split)
    
    # Prepare output file path
    model_name = args.model_path.replace('/', '_').replace('\\', '_')
    output_file = os.path.join(args.output, f"eval_{model_name}_{args.split}.txt")
    
    # Evaluate
    accuracies = evaluate_model(
        model=model,
        processor=processor,
        data_df=data_df,
        attributes=args.attributes,
        device=args.device,
        max_samples=args.max_samples,
        output_file=output_file
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()