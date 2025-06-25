#!/usr/bin/env python3
"""
Compare performance of different models on MIND dataset.
This script collects results from RecFormer fine-tuning and baseline models.
"""

import os
import json
import re
import argparse
from pathlib import Path
import pandas as pd

def extract_recformer_results(log_file):
    """Extract RecFormer results from the mind_finetuning_log.md file"""
    if not os.path.exists(log_file):
        print(f"RecFormer log file not found: {log_file}")
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Look for test results in the log
    results = {}
    
    # Pattern to match metric results
    patterns = {
        'NDCG@10': r'NDCG@10[:\s]+([0-9.]+)',
        'Recall@10': r'Recall@10[:\s]+([0-9.]+)',
        'NDCG@20': r'NDCG@20[:\s]+([0-9.]+)',
        'Recall@20': r'Recall@20[:\s]+([0-9.]+)',
        'MRR': r'MRR[:\s]+([0-9.]+)'
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            results[metric] = float(match.group(1))
    
    return results if results else None

def extract_baseline_results(checkpoint_dir, model_name):
    """Extract baseline results from RecBole checkpoint directory"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Look for log files in the checkpoint directory
    log_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.log') or 'log' in file.lower():
            log_files.append(os.path.join(checkpoint_dir, file))
    
    if not log_files:
        print(f"No log files found in {checkpoint_dir}")
        return None
    
    # Read the most recent log file
    latest_log = max(log_files, key=os.path.getmtime)
    
    with open(latest_log, 'r') as f:
        content = f.read()
    
    results = {}
    
    # Pattern to match RecBole test results
    patterns = {
        'NDCG@10': r'NDCG@10\s*:\s*([0-9.]+)',
        'Recall@10': r'Recall@10\s*:\s*([0-9.]+)',
        'NDCG@20': r'NDCG@20\s*:\s*([0-9.]+)',
        'Recall@20': r'Recall@20\s*:\s*([0-9.]+)',
        'MRR': r'MRR\s*:\s*([0-9.]+)'
    }
    
    # Look for test results (usually after "test result:")
    test_section = re.search(r'test result:.*?(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
    if test_section:
        test_content = test_section.group(0)
        for metric, pattern in patterns.items():
            match = re.search(pattern, test_content, re.IGNORECASE)
            if match:
                results[metric] = float(match.group(1))
    
    return results if results else None

def create_comparison_table(results_dict):
    """Create a formatted comparison table"""
    if not results_dict:
        print("No results to compare")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results_dict).T
    
    # Ensure all metrics are present (fill with NaN if missing)
    expected_metrics = ['NDCG@10', 'Recall@10', 'NDCG@20', 'Recall@20', 'MRR']
    for metric in expected_metrics:
        if metric not in df.columns:
            df[metric] = float('nan')
    
    # Reorder columns
    df = df[expected_metrics]
    
    # Format numbers to 4 decimal places
    df = df.round(4)
    
    print("\n" + "="*80)
    print("MIND Dataset - Model Performance Comparison")
    print("="*80)
    print(df.to_string())
    print("="*80)
    
    # Find best performing model for each metric
    print("\nBest performing model for each metric:")
    for metric in expected_metrics:
        if not df[metric].isna().all():
            best_model = df[metric].idxmax()
            best_score = df[metric].max()
            print(f"{metric}: {best_model} ({best_score:.4f})")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Compare MIND dataset results across models")
    parser.add_argument('--recformer_log', type=str, default='mind_finetuning_log.md',
                        help='Path to RecFormer finetuning log file')
    parser.add_argument('--baseline_dir', type=str, default='saved_baselines',
                        help='Directory containing baseline model results')
    parser.add_argument('--output_file', type=str, default='mind_results_comparison.csv',
                        help='Output CSV file for results')
    
    args = parser.parse_args()
    
    results = {}
    
    # Extract RecFormer results
    print("Extracting RecFormer results...")
    recformer_results = extract_recformer_results(args.recformer_log)
    if recformer_results:
        results['RecFormer'] = recformer_results
        print(f"Found RecFormer results: {recformer_results}")
    else:
        print("No RecFormer results found")
    
    # Extract baseline results
    baseline_models = [
        ('SASRec', 'mind_sasrec'),
        ('FDSA', 'mind_fdsa'),
        ('UniSRec', 'mind_unisrec'),
        ('BERT4Rec', 'mind')
    ]
    
    for model_name, dir_name in baseline_models:
        print(f"Extracting {model_name} results...")
        checkpoint_dir = os.path.join(args.baseline_dir, dir_name)
        baseline_results = extract_baseline_results(checkpoint_dir, model_name)
        if baseline_results:
            results[model_name] = baseline_results
            print(f"Found {model_name} results: {baseline_results}")
        else:
            print(f"No {model_name} results found")
    
    # Create comparison table
    if results:
        df = create_comparison_table(results)
        
        # Save to CSV
        if df is not None:
            df.to_csv(args.output_file)
            print(f"\nResults saved to: {args.output_file}")
    else:
        print("No results found for comparison")

if __name__ == "__main__":
    main() 