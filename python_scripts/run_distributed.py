#!/usr/bin/env python
"""
Simplified distributed training script for RecBole using native distributed support
"""

import os
import sys
import torch
import argparse
import datetime
from recbole.quick_start import run_recbole


def monitor_gpu_usage():
    """Monitor and report GPU usage"""
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Print GPU memory info for each available GPU
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({memory_total:.1f} GB)")
            
            # Print memory usage if GPU is active
            try:
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  Memory allocated: {memory_allocated:.2f} GB")
                print(f"  Memory cached: {memory_cached:.2f} GB")
            except:
                print(f"  Memory info not available for GPU {i}")


def main():
    """Main function using RecBole's native distributed training"""
    parser = argparse.ArgumentParser(description="Distributed RecBole training")
    parser.add_argument('--config_file', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--nproc', type=int, default=4,
                       help='Number of processes (GPUs) to use')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run distributed training.")
        sys.exit(1)
    
    # Check if we have enough GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < args.nproc:
        print(f"Requested {args.nproc} GPUs but only {num_gpus} available.")
        args.nproc = num_gpus
        print(f"Using {args.nproc} GPUs instead.")
    
    print(f"=== Pre-Training GPU Status ===")
    monitor_gpu_usage()
    
    print(f"\n=== Starting RecBole Distributed Training ===")
    print(f"Config file: {args.config_file}")
    print(f"Number of processes: {args.nproc}")
    print(f"Available GPUs: {num_gpus}")
    print(f"Start time: {datetime.datetime.now()}")
    
    try:
        # Let RecBole handle distributed training natively
        # RecBole will initialize the process group internally
        run_recbole(
            config_file_list=[args.config_file],
            config_dict={
                'nproc': args.nproc,
                'gpu_id': ','.join(str(i) for i in range(args.nproc))
            },
            saved=True
        )
        
        print(f"\n=== Training Completed Successfully ===")
        print(f"End time: {datetime.datetime.now()}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    print(f"\n=== Post-Training GPU Status ===")
    monitor_gpu_usage()


if __name__ == '__main__':
    main() 