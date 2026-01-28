"""
Adversarial Prompt Generation for Membership Inference Attack on VLMs
Using Entropy-based Optimization (GCG-inspired approach)

Goal: Generate a universal prompt that maximizes the separation between
      member and non-member data based on generation entropy.
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
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
from peft import PeftModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
ADAPTER_PATH = "./models/smolvlm-caption-finetune_3"  # LoRA 어댑터 경로 (None이면 base 모델 사용)
DATA_ROOT = "D:/datasets/coco/"
VAL_IMAGES_DIR = os.path.join(DATA_ROOT, "val2017")
TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test2017")
VAL_ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_val2017.json")
OUTPUT_DIR = "./attack_results/beast_attack/"

# Attack parameters
INITIAL_PROMPT = "Describe this image."
NUM_ITERATIONS = 50  # Reduced for faster experiments
NUM_CANDIDATES = 30  # Number of token candidates to try per position
BATCH_SIZE = 16  # Number of images to evaluate per iteration
BEAM_WIDTH = 5  # Number of best prompts to keep (BEAST algorithm)

# Entropy calculation mode
USE_FORWARD_ONLY = False  # True: forward() only (10x faster), False: generate()
NUM_ENTROPY_TOKENS = 5  # Number of tokens to consider for entropy (1-3 recommended)

MAX_NEW_TOKENS = 10  # For caption generation (only used if USE_FORWARD_ONLY=False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_coco_data(annotation_file, images_dir, num_samples=None):
    """Load COCO caption data or test data (without captions)."""
    print(f"[Info] Loading data from {annotation_file}...")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # Check if annotations exist (val2017 has them, test2017 doesn't)
    has_annotations = 'annotations' in data and len(data['annotations']) > 0
    
    if has_annotations:
        # Load with captions (val2017)
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
    else:
        # Load without captions (test2017)
        formatted_data = []
        for img_id, file_name in img_id_to_filename.items():
            full_path = os.path.join(images_dir, file_name)
            
            if os.path.exists(full_path):
                formatted_data.append({
                    "image_id": img_id,
                    "image_path": full_path,
                    "captions": []  # Empty captions for test data
                })
    
    if num_samples:
        formatted_data = formatted_data[:num_samples]
    
    return formatted_data

def load_test_images(images_dir):
    """
    Load test images directly from directory (no annotations needed).
    Used for non-member data in membership inference attack.
    """
    print(f"[Info] Loading test images from {images_dir}...")
    
    if not os.path.exists(images_dir):
        print(f"[Warning] Directory not found: {images_dir}")
        return []
    
    formatted_data = []
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for file_name in image_files:
        full_path = os.path.join(images_dir, file_name)
        # Extract image_id from filename (e.g., '000000123456.jpg' -> 123456)
        try:
            img_id = int(file_name.split('.')[0])
        except:
            img_id = hash(file_name)  # Fallback to hash if filename doesn't contain number
        
        formatted_data.append({
            "image_id": img_id,
            "image_path": full_path,
            "captions": []  # No captions needed for non-member test data
        })
    
    print(f"[Info] Found {len(formatted_data)} test images")
    return formatted_data

def split_data_for_attack(val_data, test_data=None, seed=42):
    """
    Split data according to finetuning setup:
    - val_data: val2017 split 8:2 (same as finetuning)
    - Returns data for prompt optimization and final evaluation
    
    Returns:
        prompt_member: From val2017's 80% (train split) - for prompt optimization
        prompt_nonmember: From val2017's 20% (val split) - for prompt optimization  
        eval_member: From remaining val2017's 80% - for final evaluation
        eval_nonmember: From test2017 - for final evaluation
    """
    random.seed(seed)
    random.shuffle(val_data)
    
    # Split val2017 into 8:2 (same as finetuning)
    split_idx = int(len(val_data) * 0.8)
    train_split = val_data[:split_idx]  # 80% - member data
    val_split = val_data[split_idx:]    # 20% - non-member data
    
    # Calculate split sizes
    val_split_size = len(val_split)  # 20% of val2017
    
    # For prompt optimization:
    # - Member: use same amount as val_split from train_split (80%)
    # - Non-member: use entire val_split (20%)
    prompt_member = train_split[:val_split_size]
    prompt_nonmember = val_split  # Use all of val split
    
    # For final evaluation:
    # - Member: use same amount from remaining train_split
    # - Non-member: use same amount from test2017
    remaining_train = train_split[val_split_size:]
    eval_member = remaining_train[:val_split_size]
    
    if test_data:
        random.seed(seed)
        random.shuffle(test_data)
        eval_nonmember = test_data[:val_split_size]
    else:
        # Fallback: this shouldn't happen but just in case
        print("[Warning] No test data available, using remaining val_split")
        eval_nonmember = []
    
    return prompt_member, prompt_nonmember, eval_member, eval_nonmember

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------

def load_model_and_processor(base_model_id, adapter_path=None):
    """
    기본 모델과 LoRA 어댑터를 로드합니다.
    
    Args:
        base_model_id: 기본 모델 ID
        adapter_path: LoRA 어댑터 경로 (None이면 base 모델만 사용)
    
    Returns:
        model: 로드된 모델
        processor: 프로세서
    """
    print(f"[Init] Loading model from {base_model_id}...")
    
    # 프로세서 로드
    if adapter_path and os.path.exists(adapter_path):
        try:
            processor = AutoProcessor.from_pretrained(adapter_path)
            print(f"[Info] Loaded processor from adapter path")
        except:
            processor = AutoProcessor.from_pretrained(base_model_id)
            print(f"[Info] Loaded processor from base model")
    else:
        processor = AutoProcessor.from_pretrained(base_model_id)
        print(f"[Info] Loaded processor from base model")
    
    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 기본 모델 로드
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # LoRA 어댑터 로드
    if adapter_path and os.path.exists(adapter_path):
        print(f"[Info] Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
        print("[Info] Adapter loaded and merged")
    else:
        if adapter_path:
            print(f"[Warning] Adapter path not found: {adapter_path}")
        print(f"[Info] Using base model only")
        model = base_model
    
    model.eval()
    return model, processor

# -----------------------------------------------------------------------------
# Entropy Calculation
# -----------------------------------------------------------------------------

def calculate_generation_entropy(model, processor, image_path: str, prompt: str) -> float:
    """
    Calculate entropy of model's next token prediction.
    
    Two modes:
    1. USE_FORWARD_ONLY=True: Fast forward pass to get first token logits (10x faster)
    2. USE_FORWARD_ONLY=False: Generate tokens and compute average entropy
    
    Lower entropy = more confident (member)
    Higher entropy = less confident (non-member)
    """
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Prepare input with prompt
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
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)
        
        if USE_FORWARD_ONLY:
            # Fast mode: Only forward pass to get next token logits
            # NOTE: Forward mode can only efficiently compute entropy for the FIRST token
            # To compute entropy for multiple tokens, we'd need multiple forward passes
            # which defeats the speed advantage. Use generate mode for NUM_ENTROPY_TOKENS > 1.
            
            if NUM_ENTROPY_TOKENS > 1:
                print(f"[Warning] NUM_ENTROPY_TOKENS={NUM_ENTROPY_TOKENS} with USE_FORWARD_ONLY=True")
                print(f"[Warning] Forward mode only uses first token. Set USE_FORWARD_ONLY=False for multiple tokens.")
            
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values,
                    attention_mask=inputs.attention_mask if hasattr(inputs, 'attention_mask') else None
                )
            
            # Get logits for next token (only first token in forward mode)
            # outputs.logits shape: [batch_size, seq_len, vocab_size]
            next_token_logits = outputs.logits[:, -1, :]  # Last position only
            probs = torch.softmax(next_token_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            avg_entropy = entropy.item()
            
        else:
            # Slow mode: Generate tokens and compute entropy (supports multiple tokens)
            # Only generate NUM_ENTROPY_TOKENS for efficiency (no need to generate MAX_NEW_TOKENS)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=NUM_ENTROPY_TOKENS,  # Only generate what we need
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            # Calculate entropy from scores
            entropies = []
            for score in outputs.scores[:NUM_ENTROPY_TOKENS]:  # Only use first N tokens
                probs = torch.softmax(score[0], dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                entropies.append(entropy.item())
            
            avg_entropy = np.mean(entropies) if entropies else 0.0
        
        return avg_entropy
        
    except Exception as e:
        print(f"[Error] Failed to calculate entropy: {e}")
        return 0.0

def evaluate_prompt_on_batch(model, processor, data_batch: List[Dict], prompt: str) -> List[float]:
    """Evaluate prompt on a batch of images and return entropies."""
    entropies = []
    
    for item in data_batch:
        entropy = calculate_generation_entropy(model, processor, item["image_path"], prompt)
        entropies.append(entropy)
    
    return entropies

# -----------------------------------------------------------------------------
# Prompt Optimization (GCG-inspired)
# -----------------------------------------------------------------------------

def get_token_candidates(processor, current_token_id: int, num_candidates: int) -> List[int]:
    """
    Get candidate token IDs for replacement.
    Sample from vocabulary excluding special tokens.
    """
    vocab_size = len(processor.tokenizer)
    special_token_ids = set(processor.tokenizer.all_special_ids)
    
    # Sample random tokens from vocabulary
    candidates = []
    attempts = 0
    max_attempts = num_candidates * 10
    
    while len(candidates) < num_candidates and attempts < max_attempts:
        token_id = random.randint(0, vocab_size - 1)
        if token_id not in special_token_ids:
            candidates.append(token_id)
        attempts += 1
    
    # Always include current token
    if current_token_id not in candidates:
        candidates[0] = current_token_id
    
    return candidates[:num_candidates]

def compute_objective(member_entropies: List[float], nonmember_entropies: List[float]) -> Tuple[float, float]:
    """
    Compute contrastive optimization objective.
    Goal: 1) Member entropy should be LOW (confident on training data)
          2) Non-member entropy should be HIGH (uncertain on unseen data)
          3) Maximize entropy gap (non-member - member)
    
    Returns:
        objective: Value to minimize (negative gap with penalty)
        entropy_gap: Actual gap for logging
    """
    member_avg = np.mean(member_entropies)
    nonmember_avg = np.mean(nonmember_entropies)
    
    # Entropy gap: non-member - member (should be positive and large)
    entropy_gap = nonmember_avg - member_avg
    
    # Check if member entropy is actually lower than non-member
    if entropy_gap <= 0:
        # Invalid: member entropy is higher than or equal to non-member
        # Apply large penalty
        objective = 1000.0 + abs(entropy_gap)  # Large positive value (bad)
    else:
        # Valid: member < non-member
        # Also penalize if member entropy is too high (should be confident)
        # Objective = -gap + lambda * member_avg
        # We want to minimize this, so maximize gap and minimize member entropy
        lambda_member = 0.1  # Weight for member entropy penalty
        objective = -entropy_gap + lambda_member * member_avg
    
    return objective, entropy_gap

def optimize_prompt_beam(
    model,
    processor,
    beams: List[Tuple[List[int], float]],  # (prompt_tokens, objective_score)
    member_batch: List[Dict],
    nonmember_batch: List[Dict]
) -> List[Tuple[List[int], float]]:
    """
    BEAST algorithm: Beam search optimization for prompts with GPU parallelization.
    Evaluates all beam-position-candidate combinations in large batches.
    
    Args:
        beams: List of (prompt_tokens, objective_score) tuples
        member_batch: Member data samples
        nonmember_batch: Non-member data samples
    
    Returns:
        new_beams: Top-k best beams after optimization
    """
    print(f"\n  [Beam Search] Generating all candidates...")
    
    # Generate all candidates in advance (beam × position × candidate)
    all_test_tokens = []
    candidate_info = []  # (beam_idx, token_pos, candidate_token)
    
    for beam_idx, (beam_tokens, beam_score) in enumerate(beams):
        for token_pos in range(len(beam_tokens)):
            current_token = beam_tokens[token_pos]
            candidates = get_token_candidates(processor, current_token, NUM_CANDIDATES)
            
            for candidate_token in candidates:
                # Skip if identical to current
                if candidate_token == current_token:
                    continue
                    
                # Create candidate prompt
                test_tokens = beam_tokens.copy()
                test_tokens[token_pos] = candidate_token
                
                all_test_tokens.append(test_tokens)
                candidate_info.append((beam_idx, token_pos, candidate_token))
    
    # Add original beams
    original_start_idx = len(all_test_tokens)
    for beam_tokens, beam_score in beams:
        all_test_tokens.append(beam_tokens)
    
    print(f"  [Batch Evaluation] Evaluating {len(all_test_tokens)} candidates in batches...")
    
    # Convert to prompts
    all_prompts = [processor.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in all_test_tokens]
    
    # Batch evaluation (excluding originals for now)
    new_candidates_count = len(all_test_tokens) - len(beams)
    all_objectives = []
    
    # Evaluate in batches to avoid OOM
    eval_batch_size = 32  # Adjust based on GPU memory
    
    for batch_start in tqdm(range(0, new_candidates_count, eval_batch_size), desc="  Batch eval"):
        batch_end = min(batch_start + eval_batch_size, new_candidates_count)
        batch_prompts = all_prompts[batch_start:batch_end]
        
        # Evaluate on member batch
        member_entropies_batch = []
        for prompt in batch_prompts:
            entropies = evaluate_prompt_on_batch(model, processor, member_batch, prompt)
            member_entropies_batch.append(entropies)
        
        # Evaluate on non-member batch
        nonmember_entropies_batch = []
        for prompt in batch_prompts:
            entropies = evaluate_prompt_on_batch(model, processor, nonmember_batch, prompt)
            nonmember_entropies_batch.append(entropies)
        
        # Compute objectives
        for member_ent, nonmember_ent in zip(member_entropies_batch, nonmember_entropies_batch):
            objective, _ = compute_objective(member_ent, nonmember_ent)
            all_objectives.append(objective)
    
    # Add original beam objectives
    for _, beam_score in beams:
        all_objectives.append(beam_score)
    
    # Create list of (tokens, objective)
    all_candidates = [(all_test_tokens[i], all_objectives[i]) for i in range(len(all_test_tokens))]
    
    # Sort by objective (lower is better)
    all_candidates.sort(key=lambda x: x[1])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for tokens, obj in all_candidates:
        tokens_tuple = tuple(tokens)
        if tokens_tuple not in seen:
            seen.add(tokens_tuple)
            unique_candidates.append((tokens, obj))
    
    # Return top-k beams
    top_k = unique_candidates[:BEAM_WIDTH]
    
    print(f"\n  [Selected] Top {len(top_k)} beams:")
    for i, (tokens, obj) in enumerate(top_k[:3]):  # Show top 3
        prompt = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"    {i+1}. Objective: {obj:.4f} | Prompt: '{prompt}'")
    
    return top_k

# -----------------------------------------------------------------------------
# Main Attack Loop
# -----------------------------------------------------------------------------

def run_adversarial_prompt_attack(
    model,
    processor,
    member_data: List[Dict],
    nonmember_data: List[Dict]
):
    """
    Main attack loop using BEAST (Beam Search) algorithm.
    Maintains top-k prompt candidates and explores them in parallel.
    """
    print("\n" + "="*80)
    print("ADVERSARIAL PROMPT ATTACK - BEAST (BEAM SEARCH) ALGORITHM")
    print("="*80)
    
    # Initialize prompt
    prompt = INITIAL_PROMPT
    prompt_tokens = processor.tokenizer.encode(prompt, add_special_tokens=False)
    
    print(f"\n[Init] Initial prompt: '{prompt}'")
    print(f"[Init] Token IDs: {prompt_tokens}")
    print(f"[Init] Number of tokens: {len(prompt_tokens)}")
    print(f"[Init] Beam width: {BEAM_WIDTH}")
    
    # Initialize beams with initial prompt
    # Evaluate initial prompt
    member_batch = random.sample(member_data, min(BATCH_SIZE, len(member_data)))
    nonmember_batch = random.sample(nonmember_data, min(BATCH_SIZE, len(nonmember_data)))
    
    member_entropies = evaluate_prompt_on_batch(model, processor, member_batch, prompt)
    nonmember_entropies = evaluate_prompt_on_batch(model, processor, nonmember_batch, prompt)
    initial_objective, initial_gap = compute_objective(member_entropies, nonmember_entropies)
    
    # Initialize beams: [(tokens, objective_score)]
    beams = [(prompt_tokens, initial_objective)]
    
    print(f"\n[Init] Initial objective: {initial_objective:.4f} (gap: {initial_gap:.4f})")
    
    # History tracking
    history = {
        "iteration": [],
        "best_prompt": [],
        "best_objective": [],
        "best_gap": [],
        "beam_diversity": []  # Number of unique beams
    }
    
    # Best result tracking
    best_prompt = prompt
    best_tokens = prompt_tokens.copy()
    best_objective = initial_objective
    best_gap = initial_gap
    best_iteration = 0
    
    # Main optimization loop (BEAST)
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'='*80}")
        
        # Sample batches for this iteration
        member_batch = random.sample(member_data, min(BATCH_SIZE, len(member_data)))
        nonmember_batch = random.sample(nonmember_data, min(BATCH_SIZE, len(nonmember_data)))
        
        # Current best beam
        current_best_tokens, current_best_obj = beams[0]
        current_best_prompt = processor.tokenizer.decode(current_best_tokens, skip_special_tokens=True)
        
        print(f"\n[Current Best] Prompt: '{current_best_prompt}'")
        print(f"[Current Best] Objective: {current_best_obj:.4f}")
        print(f"[Current Best] Gap: {-current_best_obj:.4f}")
        
        # Beam search optimization
        print(f"\n[Optimizing] Exploring {len(beams)} beams...")
        beams = optimize_prompt_beam(
            model, processor, beams, member_batch, nonmember_batch
        )
        
        # Update best if improved
        new_best_tokens, new_best_obj = beams[0]
        new_best_prompt = processor.tokenizer.decode(new_best_tokens, skip_special_tokens=True)
        new_best_gap = -new_best_obj if new_best_obj < 0 else 0.0  # Only count valid positive gaps
        
        # Re-evaluate to get actual entropy values for logging
        member_eval = evaluate_prompt_on_batch(model, processor, member_batch[:4], new_best_prompt)
        nonmember_eval = evaluate_prompt_on_batch(model, processor, nonmember_batch[:4], new_best_prompt)
        member_eval_avg = np.mean(member_eval)
        nonmember_eval_avg = np.mean(nonmember_eval)
        
        print(f"[Current Best] Member Entropy: {member_eval_avg:.4f}")
        print(f"[Current Best] Non-member Entropy: {nonmember_eval_avg:.4f}")
        
        if new_best_obj < best_objective:
            best_objective = new_best_obj
            best_gap = new_best_gap
            best_prompt = new_best_prompt
            best_tokens = new_best_tokens
            best_iteration = iteration + 1
            print(f"\n[Best] ⭐ New best found at iteration {iteration + 1}!")
            print(f"[Best] Objective: {best_objective:.4f}, Gap: {best_gap:.4f}")
            print(f"[Best] Member: {member_eval_avg:.4f}, Non-member: {nonmember_eval_avg:.4f}")
        
        # Save history
        history["iteration"].append(iteration + 1)
        history["best_prompt"].append(best_prompt)
        history["best_objective"].append(best_objective)
        history["best_gap"].append(best_gap)
        history["beam_diversity"].append(len(beams))
    
    # Final results
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    
    print(f"\n[Best] Found at iteration: {best_iteration}/{NUM_ITERATIONS}")
    print(f"[Best] Objective: {best_objective:.4f}")
    print(f"[Best] Entropy Gap: {best_gap:.4f}")
    print(f"\n[Final] Optimized Prompt: '{best_prompt}'")
    print(f"[Final] Token IDs: {best_tokens}")
    
    # Evaluate on larger sample
    print(f"\n[Evaluating] Full member dataset...")
    member_entropies_full = []
    for item in tqdm(member_data[:100], desc="Member eval"):  # Limit for speed
        entropy = calculate_generation_entropy(model, processor, item["image_path"], best_prompt)
        member_entropies_full.append(entropy)
    
    print(f"\n[Evaluating] Full non-member dataset...")
    nonmember_entropies_full = []
    for item in tqdm(nonmember_data[:100], desc="Non-member eval"):
        entropy = calculate_generation_entropy(model, processor, item["image_path"], best_prompt)
        nonmember_entropies_full.append(entropy)
    
    member_avg_final = np.mean(member_entropies_full)
    nonmember_avg_final = np.mean(nonmember_entropies_full)
    gap_final = nonmember_avg_final - member_avg_final
    
    print(f"\n[Final Metrics]")
    print(f"  Member Entropy: {member_avg_final:.4f} ± {np.std(member_entropies_full):.4f}")
    print(f"  Non-member Entropy: {nonmember_avg_final:.4f} ± {np.std(nonmember_entropies_full):.4f}")
    print(f"  Entropy Gap: {gap_final:.4f}")
    
    return best_prompt, history, member_entropies_full, nonmember_entropies_full

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_results(history, member_entropies, nonmember_entropies, output_dir):
    """Plot optimization results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Objective and Gap over iterations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: Best objective over iterations
    axes[0, 0].plot(history["iteration"], history["best_objective"], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Best Objective (lower is better)')
    axes[0, 0].set_title('Best Objective Evolution (BEAST)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Entropy gap over iterations
    axes[0, 1].plot(history["iteration"], history["best_gap"], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Entropy Gap (Non-member - Member)')
    axes[0, 1].set_title('Entropy Gap Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Beam diversity
    axes[1, 0].plot(history["iteration"], history["beam_diversity"], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Number of Unique Beams')
    axes[1, 0].set_title('Beam Diversity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Final entropy distribution
    axes[1, 1].hist(member_entropies, bins=30, alpha=0.5, label='Member', color='blue')
    axes[1, 1].hist(nonmember_entropies, bins=30, alpha=0.5, label='Non-member', color='red')
    axes[1, 1].set_xlabel('Entropy')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Entropy Distribution (Final Prompt)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_curve.png"), dpi=300)
    plt.close()
    
    # Plot 2: Separate entropy distribution plot (higher resolution)
    plt.figure(figsize=(10, 6))
    plt.hist(member_entropies, bins=30, alpha=0.5, label='Member', color='blue')
    plt.hist(nonmember_entropies, bins=30, alpha=0.5, label='Non-member', color='red')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Entropy Distribution (Final Prompt - BEAST Algorithm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "entropy_distribution.png"), dpi=300)
    plt.close()
    
    print(f"\n[Saved] Plots to {output_dir}")

def save_results(final_prompt, history, member_entropies, nonmember_entropies, output_dir):
    """Save all results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "algorithm": "BEAST (Beam Search)",
        "final_prompt": final_prompt,
        "initial_prompt": INITIAL_PROMPT,
        "num_iterations": NUM_ITERATIONS,
        "beam_width": BEAM_WIDTH,
        "num_candidates": NUM_CANDIDATES,
        "history": history,
        "final_metrics": {
            "member_entropy_mean": float(np.mean(member_entropies)),
            "member_entropy_std": float(np.std(member_entropies)),
            "nonmember_entropy_mean": float(np.mean(nonmember_entropies)),
            "nonmember_entropy_std": float(np.std(nonmember_entropies)),
            "entropy_gap": float(np.mean(nonmember_entropies) - np.mean(member_entropies))
        }
    }
    
    output_file = os.path.join(output_dir, "attack_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Saved] Results to {output_file}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("\n" + "="*80)
    print("MEMBERSHIP INFERENCE ATTACK - BEAST (BEAM SEARCH) ALGORITHM")
    print("="*80)
    
    # Load model
    print(f"\n[Loading] Model: {MODEL_ID}")
    if ADAPTER_PATH:
        print(f"[Loading] Adapter: {ADAPTER_PATH}")
    model, processor = load_model_and_processor(MODEL_ID, ADAPTER_PATH)
    
    # Load data
    print(f"\n[Loading] Validation data from {VAL_ANNOTATION_FILE}")
    val_data = load_coco_data(VAL_ANNOTATION_FILE, VAL_IMAGES_DIR, num_samples=None)
    
    print(f"[Loading] Test images from {TEST_IMAGES_DIR}")
    test_data = load_test_images(TEST_IMAGES_DIR)
    
    # Split data according to finetuning setup
    prompt_member, prompt_nonmember, eval_member, eval_nonmember = split_data_for_attack(
        val_data, test_data, seed=42
    )
    
    print(f"\n[Data Split] Prompt Optimization:")
    print(f"  - Member samples (from val2017 train split): {len(prompt_member)}")
    print(f"  - Non-member samples (from val2017 val split): {len(prompt_nonmember)}")
    print(f"\n[Data Split] Final Evaluation:")
    print(f"  - Member samples (from remaining val2017 train): {len(eval_member)}")
    print(f"  - Non-member samples (from test2017): {len(eval_nonmember)}")
    
    # Run attack (prompt optimization phase)
    print("\n[Phase 1] Adversarial Prompt Optimization...")
    final_prompt, history, _, _ = run_adversarial_prompt_attack(
        model, processor, prompt_member, prompt_nonmember
    )
    
    # Final evaluation with separate data
    print("\n[Phase 2] Final Evaluation on Held-out Data...")
    print(f"Using optimized prompt: '{final_prompt}'")
    
    eval_member_entropies = evaluate_prompt_on_batch(model, processor, eval_member, final_prompt)
    eval_nonmember_entropies = evaluate_prompt_on_batch(model, processor, eval_nonmember, final_prompt)
    
    print(f"\n[Evaluation Results]")
    print(f"  Member entropy (mean): {np.mean(eval_member_entropies):.4f} ± {np.std(eval_member_entropies):.4f}")
    print(f"  Non-member entropy (mean): {np.mean(eval_nonmember_entropies):.4f} ± {np.std(eval_nonmember_entropies):.4f}")
    print(f"  Entropy gap: {np.mean(eval_nonmember_entropies) - np.mean(eval_member_entropies):.4f}")
    
    # Run attack visualization with evaluation data
    member_entropies = eval_member_entropies
    nonmember_entropies = eval_nonmember_entropies
    
    # Save and visualize
    plot_results(history, member_entropies, nonmember_entropies, OUTPUT_DIR)
    save_results(final_prompt, history, member_entropies, nonmember_entropies, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("ATTACK COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()