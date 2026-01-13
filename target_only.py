import json
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from random import sample
import argparse

def load_data(member_similarity_file, non_member_similarity_file):
    """Load similarity data from JSON files."""
    with open(member_similarity_file, 'r') as file:
        member_data_all = json.load(file)
    with open(non_member_similarity_file, 'r') as file:
        non_member_data_all = json.load(file)
    return member_data_all, non_member_data_all

def perform_z_test(group1, group2):
    """Perform z-test between two groups."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_se = np.sqrt(std1**2/n1 + std2**2/n2)
    z = (mean1 - mean2) / pooled_se
    p_value = norm.sf(z)
    return p_value

def target_only_inference(member_data_all, non_member_data_all, granularity, temperature_low, temperature_high, metric):
    """
    Perform target-only membership inference attack.
    
    Args:
        member_data_all: List of member similarity data
        non_member_data_all: List of non-member similarity data
        granularity: Number of samples per test
        temperature_low: Low temperature value
        temperature_high: High temperature value
        metric: Similarity metric to use
    
    Returns:
        AUC score
    """
    all_indices_member = range(len(member_data_all))
    all_indices_non_member = range(len(non_member_data_all))
    p_list = []
    label_list = []
    
    for _ in range(1000):
        member_sampled_indices = sample(all_indices_member, granularity)
        member_low = [member_data_all[index][f'similarity_{temperature_low}'][metric]  for index in member_sampled_indices]
        member_high = [member_data_all[index][f'similarity_{temperature_high}'][metric]  for index in member_sampled_indices]
        p_member = perform_z_test(member_low, member_high)
        p_list.append(p_member)
        label_list.append(0)

        non_member_sampled_indices = sample(all_indices_non_member, granularity)
        non_member_low = [non_member_data_all[index][f'similarity_{temperature_low}'][metric]  for index in non_member_sampled_indices]
        non_member_high = [non_member_data_all[index][f'similarity_{temperature_high}'][metric]  for index in non_member_sampled_indices]
        p_non_member = perform_z_test(non_member_low, non_member_high)
        p_list.append(p_non_member)
        label_list.append(1)
        
    auc = roc_auc_score(label_list, p_list)
    return auc


def main(args):
    """Main function for single model evaluation."""
    print("="*60)
    print("🎯 Target-Only Membership Inference Attack")
    print("="*60)
    
    # Load data
    print(f"\n📂 Loading data...")
    print(f"  Member file: {args.member_similarity_file}")
    print(f"  Non-member file: {args.non_member_similarity_file}")
    
    member_data_all, non_member_data_all = load_data(
        args.member_similarity_file, 
        args.non_member_similarity_file
    )
    
    print(f"\n📊 Dataset info:")
    print(f"  Member samples: {len(member_data_all)}")
    print(f"  Non-member samples: {len(non_member_data_all)}")
    
    print(f"\n⚙️  Attack parameters:")
    print(f"  Granularity: {args.granularity}")
    print(f"  Temperature range: [{args.temperature_low}, {args.temperature_high}]")
    print(f"  Similarity metric: {args.similarity_metric}")
    print(f"  Number of runs: {args.num_runs}")
    
    # Run attack multiple times
    print(f"\n🔄 Running attack...")
    aucs = []
    for run_idx in range(args.num_runs):
        auc = target_only_inference(
            member_data_all, 
            non_member_data_all, 
            args.granularity, 
            args.temperature_low, 
            args.temperature_high, 
            args.similarity_metric
        )
        aucs.append(auc)
        print(f"  Run {run_idx + 1}/{args.num_runs}: AUC = {auc:.4f}")

    # Calculate statistics
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    min_auc = min(aucs)
    max_auc = max(aucs)
    
    # Print results
    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print("="*60)
    print(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"Min AUC:     {min_auc:.4f}")
    print(f"Max AUC:     {max_auc:.4f}")
    print("="*60)
    
    # Save results if requested
    if args.output:
        results = {
            'member_file': args.member_similarity_file,
            'non_member_file': args.non_member_similarity_file,
            'parameters': {
                'granularity': args.granularity,
                'temperature_low': args.temperature_low,
                'temperature_high': args.temperature_high,
                'similarity_metric': args.similarity_metric,
                'num_runs': args.num_runs
            },
            'results': {
                'aucs': aucs,
                'avg_auc': float(avg_auc),
                'std_auc': float(std_auc),
                'min_auc': float(min_auc),
                'max_auc': float(max_auc)
            }
        }
        
        print(f"\n💾 Saving results to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print("✅ Results saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Target-Only Membership Inference Attack for Vision-Language Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  # Example usage:
  python target_only.py \\
    --member_similarity_file data/merged_member_similarity.json \\
    --non_member_similarity_file data/merged_non_member_similarity.json \\
    --output results_merged.json
        """
    )
    
    # Required arguments
    parser.add_argument('--member_similarity_file', type=str, required=True,
                       help='Path to member similarity JSON file')
    parser.add_argument('--non_member_similarity_file', type=str, required=True,
                       help='Path to non-member similarity JSON file')
    
    # Attack parameters
    parser.add_argument('--granularity', type=int, default=50,
                       help='Number of samples per test (default: 50)')
    parser.add_argument('--temperature_high', type=float, default=1.6,
                       help='High temperature value (default: 1.6)')
    parser.add_argument('--temperature_low', type=float, default=0.1,
                       help='Low temperature value (default: 0.1)')
    parser.add_argument('--similarity_metric', type=str, default='rouge2_f',
                       help='Similarity metric to use (default: rouge2_f)')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of evaluation runs (default: 5)')
    
    # Output
    parser.add_argument('--output', type=str,
                       help='Output JSON file to save results')

    args = parser.parse_args()
    main(args)
