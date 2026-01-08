"""
VLM Merging Implementation for PaliGemma
Based on ICML 2025: "Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging"
GitHub: https://github.com/shiqichen17/VLM_Merging
"""

import torch
import os
import argparse
from typing import List, Dict, Literal
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

# =========================================================
# Merging Utility Functions (from merge_utils.py)
# =========================================================

def prune(
    tensor: torch.Tensor, 
    density: float, 
    method: Literal["magnitude", "random"], 
    rescale: bool = False
) -> torch.Tensor:
    """
    Prune a tensor by keeping only the top-k values.
    
    Args:
        tensor: The tensor to prune
        density: Fraction of values to preserve [0,1]
        method: "magnitude" or "random"
        rescale: Whether to rescale after pruning
    """
    if density >= 1.0:
        return tensor
    
    if tensor.numel() == 0:
        return tensor
    
    k = int(tensor.numel() * density)
    if k == 0:
        return tensor * 0
    
    if method == "magnitude":
        # Keep top-k by absolute magnitude
        threshold_idx = tensor.numel() - k + 1
        threshold = torch.kthvalue(tensor.abs().flatten(), threshold_idx).values
        mask = tensor.abs() >= threshold
    elif method == "random":
        # Keep random k values
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        indices = torch.randperm(tensor.numel())[:k]
        mask.view(-1)[indices] = True
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    pruned = tensor * mask
    
    if rescale and method == "random":
        # DARE rescaling
        pruned = pruned / density
    
    return pruned


def calculate_majority_sign_mask(
    tensor: torch.Tensor, 
    method: Literal["total", "frequency"] = "total"
) -> torch.Tensor:
    """
    Calculate the majority sign mask across task tensors.
    
    Args:
        tensor: Stacked task tensors [num_tasks, ...]
        method: "total" (sum-based) or "frequency" (voting-based)
    """
    if method == "total":
        # Sign of the sum
        sign_sum = tensor.sum(dim=0)
        majority_sign = torch.sign(sign_sum)
    elif method == "frequency":
        # Most frequent sign
        sign_tensor = torch.sign(tensor)
        majority_sign = torch.sign(sign_tensor.sum(dim=0))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create mask where each tensor agrees with majority
    sign_tensor = torch.sign(tensor)
    majority_sign_mask = (sign_tensor == majority_sign.unsqueeze(0))
    
    return majority_sign_mask


def disjoint_merge(
    task_tensors: torch.Tensor, 
    majority_sign_mask: torch.Tensor
) -> torch.Tensor:
    """
    Merge task tensors using disjoint merge.
    Only average values that agree with the majority sign.
    """
    mixed = (task_tensors * majority_sign_mask).sum(dim=0)
    num_preserved = majority_sign_mask.sum(dim=0)
    return mixed / torch.clamp(num_preserved, min=1.0)


def reshape_weight_task_tensors(
    task_tensors: torch.Tensor, 
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Reshape weights to match task tensor dimensions.
    """
    while len(weights.shape) < len(task_tensors.shape):
        weights = weights.unsqueeze(-1)
    return weights


# =========================================================
# Merging Algorithms
# =========================================================

def task_arithmetic(
    task_tensors: List[torch.Tensor], 
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Simple weighted average of task vectors.
    """
    task_tensors = torch.stack(task_tensors, dim=0)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted = task_tensors * weights
    return weighted.sum(dim=0)


def ties_merging(
    task_tensors: List[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """
    TIES Merging: Trim, Elect Sign, Disjoint Merge
    
    Args:
        task_tensors: List of task vectors (delta from base)
        weights: Weights for each task
        density: Fraction of parameters to keep
        majority_sign_method: How to determine majority sign
    """
    # 1. Trim: Keep only top-k parameters by magnitude
    task_tensors = [prune(t, density, method="magnitude") for t in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    
    # 2. Elect Sign: Find majority sign
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    
    # 3. Weight task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted = task_tensors * weights
    
    # 4. Disjoint Merge: Average only values with majority sign
    merged = disjoint_merge(weighted, majority_sign_mask)
    
    return merged


def dare_ties_merging(
    task_tensors: List[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """
    DARE-TIES: Drop And REscale with TIES merging
    
    Uses random pruning with rescaling instead of magnitude-based.
    """
    # 1. DARE: Random pruning with rescaling
    task_tensors = [prune(t, density, method="random", rescale=True) for t in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    
    # 2. Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    
    # 3. Weight and merge
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted = task_tensors * weights
    merged = disjoint_merge(weighted, majority_sign_mask)
    
    return merged


def dare_linear_merging(
    task_tensors: List[torch.Tensor],
    weights: torch.Tensor,
    density: float,
) -> torch.Tensor:
    """
    DARE-Linear: Simple weighted average after DARE pruning
    """
    # DARE: Random pruning with rescaling
    task_tensors = [prune(t, density, method="random", rescale=True) for t in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    
    # Weighted average
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted = task_tensors * weights
    
    return weighted.sum(dim=0)


# =========================================================
# PaliGemma-Specific Merging
# =========================================================

def merge_paligemma_vlm(
    base_model_path: str,
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    mode: str = "ties",
    alpha: float = 1.0,
    alpha2: float = None,
    density: float = 0.3,
    layerswap_base_layer_num: int = -1,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16
):
    """
    Merge PaliGemma models using various strategies.
    
    Args:
        base_model_path: Base/pretrained model path
        model_a_path: First fine-tuned model
        model_b_path: Second fine-tuned model
        output_path: Where to save merged model
        mode: Merging mode - 'base', 'layerswap', 'ties', 'dareties', 'darelinear'
        alpha: Weight for model A (or scaling factor for TIES)
        alpha2: Weight for model B (if None, uses 1-alpha)
        density: Fraction of parameters to keep for sparse methods
        layerswap_base_layer_num: Layer threshold for layerswap mode
        device: Device to load models on
        dtype: Data type for models
    """
    
    print(f"🔄 Starting VLM Merging (Mode: {mode})")
    print("="*60)
    
    # Load models
    print(f"📥 Loading Base Model: {base_model_path}")
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype=dtype, device_map=device
    )
    
    print(f"📥 Loading Model A: {model_a_path}")
    model_a = PaliGemmaForConditionalGeneration.from_pretrained(
        model_a_path, torch_dtype=dtype, device_map=device
    )
    
    print(f"📥 Loading Model B: {model_b_path}")
    model_b = PaliGemmaForConditionalGeneration.from_pretrained(
        model_b_path, torch_dtype=dtype, device_map=device
    )
    
    # Get state dicts
    base_sd = base_model.state_dict()
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    
    # Free memory
    del model_a, model_b
    
    # Define keys to exclude from merging
    excluded_keys = {
        'language_model.model.embed_tokens.weight',
        'language_model.lm_head.weight'
    }
    
    # Prepare weights
    if alpha2 is None:
        weights = torch.tensor([alpha, 1 - alpha])
    else:
        weights = torch.tensor([alpha, alpha2])
    
    print(f"\n⚙️  Merging Parameters:")
    print(f"   Mode: {mode}")
    print(f"   Weights: [{weights[0]:.2f}, {weights[1]:.2f}]")
    if mode in ['ties', 'dareties', 'darelinear']:
        print(f"   Density: {density}")
    print("="*60)
    
    merged_sd = {}
    all_keys = list(base_sd.keys())
    
    print(f"\n🔄 Processing {len(all_keys)} tensors...")
    
    with torch.no_grad():
        for idx, key in enumerate(all_keys):
            if (idx + 1) % 100 == 0:
                print(f"   Progress: {idx + 1}/{len(all_keys)}")
            
            # Skip vision encoder and projector - keep from Model B
            if "vision_tower" in key or "multi_modal_projector" in key:
                if key in sd_b:
                    merged_sd[key] = sd_b[key].clone()
                else:
                    merged_sd[key] = base_sd[key].clone()
                continue
            
            # Skip excluded keys
            if any(excl in key for excl in excluded_keys):
                merged_sd[key] = base_sd[key].clone()
                continue
            
            # Check if key exists in all models
            if key not in sd_a or key not in sd_b:
                merged_sd[key] = base_sd[key].clone()
                continue
            
            # Convert to float for computation
            tv_base = base_sd[key].float()
            tv_a = sd_a[key].float()
            tv_b = sd_b[key].float()
            
            # === Merging Logic ===
            if mode == "base":
                # Simple weighted average
                merged_sd[key] = (
                    weights[0] * tv_a + weights[1] * tv_b
                ).to(dtype)
                
            elif mode == "layerswap":
                # Layer-based swapping
                layer_num = extract_layer_number(key)
                if layer_num is not None and layer_num <= layerswap_base_layer_num:
                    merged_sd[key] = tv_a.to(dtype)
                else:
                    merged_sd[key] = (
                        weights[0] * tv_a + weights[1] * tv_b
                    ).to(dtype)
                    
            elif mode in ["ties", "dareties", "darelinear"]:
                # Task vector-based methods
                delta_a = tv_a - tv_base
                delta_b = tv_b - tv_base
                
                if mode == "ties":
                    merged_delta = ties_merging(
                        [delta_a, delta_b], weights, density
                    )
                elif mode == "dareties":
                    merged_delta = dare_ties_merging(
                        [delta_a, delta_b], weights, density
                    )
                elif mode == "darelinear":
                    merged_delta = dare_linear_merging(
                        [delta_a, delta_b], weights, density
                    )
                
                merged_sd[key] = (tv_base + merged_delta).to(dtype)
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
    
    print(f"\n✅ Merging complete!")
    print(f"📦 Applying merged weights to base model...")
    
    # Load merged weights into base model
    base_model.load_state_dict(merged_sd)
    
    # Save merged model
    print(f"💾 Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    base_model.save_pretrained(output_path)
    
    # Save processor from base
    print(f"💾 Saving processor...")
    processor = PaliGemmaProcessor.from_pretrained(base_model_path, use_fast=True)
    processor.save_pretrained(output_path)
    
    print(f"\n✅ VLM Merging Complete!")
    print(f"📁 Output: {output_path}")
    

def extract_layer_number(key: str) -> int:
    """Extract layer number from parameter key."""
    import re
    match = re.search(r'\.(\d+)\.', key)
    if match:
        return int(match.group(1))
    return None


# =========================================================
# Main Function
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="VLM Merging for PaliGemma (ICML 2025 Method)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # TIES Merging (recommended)
  python merge_vlm.py \\
    --base_model google/paligemma-3b-pt-224 \\
    --model_a NYUAD-ComNets/FaceScanPaliGemma_Race \\
    --model_b NYUAD-ComNets/FaceScanPaliGemma_Age \\
    --output ./weights/merged_ties \\
    --mode ties --alpha 1.0 --alpha2 1.0 --density 0.3

  # DARE-TIES Merging
  python merge_vlm.py \\
    --base_model google/paligemma-3b-pt-224 \\
    --model_a NYUAD-ComNets/FaceScanPaliGemma_Race \\
    --model_b NYUAD-ComNets/FaceScanPaliGemma_Age \\
    --output ./weights/merged_dare \\
    --mode dareties --alpha 1.2 --density 0.2

  # Simple weighted average
  python merge_vlm.py \\
    --base_model google/paligemma-3b-pt-224 \\
    --model_a NYUAD-ComNets/FaceScanPaliGemma_Race \\
    --model_b NYUAD-ComNets/FaceScanPaliGemma_Age \\
    --output ./weights/merged_base \\
    --mode base --alpha 0.5
        """
    )
    
    # Required arguments
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base/pretrained model path")
    parser.add_argument("--model_a", type=str, required=True,
                       help="First fine-tuned model path")
    parser.add_argument("--model_b", type=str, required=True,
                       help="Second fine-tuned model path")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for merged model")
    
    # Merging configuration
    parser.add_argument("--mode", type=str, default="ties",
                       choices=["base", "layerswap", "ties", "dareties", "darelinear"],
                       help="Merging mode (default: ties)")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Weight for model A (default: 1.0)")
    parser.add_argument("--alpha2", type=float, default=None,
                       help="Weight for model B (default: 1-alpha)")
    parser.add_argument("--density", type=float, default=0.3,
                       help="Density for sparse methods (default: 0.3)")
    parser.add_argument("--layerswap_layer", type=int, default=-1,
                       help="Layer threshold for layerswap mode")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (default: cpu)")
    
    args = parser.parse_args()
    
    # Run merging
    merge_paligemma_vlm(
        base_model_path=args.base_model,
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        output_path=args.output,
        mode=args.mode,
        alpha=args.alpha,
        alpha2=args.alpha2,
        density=args.density,
        layerswap_base_layer_num=args.layerswap_layer,
        device=args.device,
        dtype=torch.bfloat16
    )


if __name__ == "__main__":
    main()
