#!/usr/bin/env python
"""
Distributed training script for RecBole using proper distributed setup
Based on RecBole documentation: https://recbole.io/docs/get_started/distributed_training.html
"""

import os
import sys
import torch
import signal
import threading
import time

# CRITICAL: Apply PyTorch 2.7 compatibility patch BEFORE any RecBole imports
# This must be done first to ensure all torch.load calls use weights_only=False
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """
    Patched version of torch.load that automatically sets weights_only=False
    for RecBole checkpoint files to maintain compatibility with PyTorch 2.7+
    """
    # Always set weights_only=False for RecBole compatibility unless explicitly overridden
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply the monkey patch globally
torch.load = patched_torch_load

# Set environment variable to ensure all subprocesses also get the fix
os.environ['RECBOLE_PYTORCH_COMPAT_FIX'] = '1'

print("🔧 Applied PyTorch 2.7 compatibility patch for RecBole")

# Now safe to import other modules
import torch.multiprocessing as mp
import argparse
import datetime
import shutil
from pathlib import Path
from recbole.quick_start import run_recboles


def check_cuda_compatibility():
    """Check CUDA compatibility and provide recommendations"""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    
    # Check PyTorch version
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    
    print(f"PyTorch version: {torch_version}")
    print(f"CUDA version: {cuda_version}")
    
    # Check GPU capabilities
    gpu_capabilities = []
    for i in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(i)
        gpu_name = torch.cuda.get_device_name(i)
        gpu_capabilities.append((i, gpu_name, capability))
        print(f"GPU {i}: {gpu_name} - Compute Capability: {capability[0]}.{capability[1]}")
    
    # Check for H100 GPUs (compute capability 9.0)
    has_h100 = any(cap[2][0] >= 9 for cap in gpu_capabilities)
    
    if has_h100:
        # Check if PyTorch supports sm_90
        try:
            # Try to create a simple tensor operation to test compatibility
            test_tensor = torch.tensor([1.0], device='cuda:0')
            _ = test_tensor + 1
            return True, "CUDA compatibility OK"
        except RuntimeError as e:
            if "no kernel image is available" in str(e):
                return False, f"PyTorch {torch_version} doesn't support H100 GPUs. Need PyTorch >=2.1 with CUDA 12.x"
            else:
                return False, f"CUDA error: {e}"
    
    return True, "CUDA compatibility OK"


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


def clean_checkpoint_dir(checkpoint_dir):
    """Clean potentially corrupted checkpoint files"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return
    
    print(f"Cleaning checkpoint directory: {checkpoint_dir}")
    
    # Remove any potentially corrupted .pth files
    for pth_file in checkpoint_path.glob("*.pth"):
        try:
            # Try to load the checkpoint to see if it's corrupted
            # Use weights_only=False for RecBole checkpoints (they use pickle protocol 4)
            torch.load(pth_file, map_location='cpu', weights_only=False)
            print(f"  Checkpoint {pth_file.name} is valid")
        except Exception as e:
            print(f"  Removing corrupted checkpoint: {pth_file.name} - {e}")
            pth_file.unlink()


def setup_distributed_training_config(args):
    """Set up distributed training parameters."""
    # Clean checkpoint directory first
    checkpoint_dir = f"./saved_baselines/{args.model.lower()}_test"
    clean_checkpoint_dir(checkpoint_dir)
    
    config_dict = {
        "world_size": args.nproc,
        "ip": "127.0.0.1",
        "port": "12355",
        "nproc": args.nproc,
        "offset": 0,
        "gpu_id": ','.join(str(i) for i in range(args.nproc)),
        "use_gpu": True,
        "checkpoint_dir": checkpoint_dir,
        "save_step": 1,
    }

    # We no longer override eval_args. The settings from the YAML config file will
    # be used, allowing RecBole to run train, validation, and test in one go.
    # The wrapper will handle the expected errors on worker nodes.
    print("  Distributed training configured to use evaluation settings from config file.")
    
    # Add H100 compatibility workarounds
    if args.disable_flops:
        config_dict['benchmark_filename'] = None
    
    return config_dict, checkpoint_dir


def run_distributed_training_wrapper(rank, model, dataset, config_file_list, kwargs):
    """
    Wrapper for distributed training with a barrier to prevent race conditions during evaluation.
    """
    try:
        # Import here to avoid issues with multiprocessing
        from recbole.quick_start import run_recboles
        from recbole.trainer.trainer import Trainer
        import torch.distributed

        # Ensure the monkey patch for torch.load is applied in each subprocess
        if not hasattr(torch.load, '__wrapped__'):
            torch.load = patched_torch_load
            print(f"Process {rank}: Applied PyTorch compatibility patch")

        # Monkey-patch the Trainer.fit method to add a barrier
        _original_fit = Trainer.fit
        
        def patched_fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=True, callback_fn=None):
            # Run the original training process
            result = _original_fit(self, train_data, valid_data, verbose, saved, show_progress, callback_fn)
            
            # After training, and before testing, insert a barrier to synchronize all processes.
            # This ensures that the master process (rank 0) has finished saving the best model
            # before any process attempts to load it for the final test evaluation.
            if self.config['world_size'] > 1:
                print(f"Process {self.config['local_rank']}: Training finished. Synchronizing all processes before test phase.")
                torch.distributed.barrier()
                print(f"Process {self.config['local_rank']}: Synchronization complete. Proceeding to test.")
            
            return result
        
        Trainer.fit = patched_fit
        print(f"Process {rank}: Applied synchronization barrier patch to Trainer.fit.")

        # This function now handles training, validation, and testing in one go.
        return run_recboles(rank, model, dataset, config_file_list, kwargs)
        
    except Exception as e:
        error_msg = str(e)
        
        # Handle H100 compatibility issues
        if "no kernel image is available" in error_msg:
            print(f"Process {rank}: H100 compatibility issue detected.")
            raise RuntimeError("H100 GPU compatibility issue. Need PyTorch >=2.1 with CUDA 12.x")
        
        else:
            import traceback
            print(f"Process {rank}: Unexpected error in wrapper: {error_msg}")
            traceback.print_exc()
            raise


def main():
    """Main function using RecBole's proper distributed training"""
    parser = argparse.ArgumentParser(description="Distributed RecBole training")
    parser.add_argument('--config_file', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--nproc', type=int, default=4,
                       help='Number of processes (GPUs) to use')
    parser.add_argument('--model', type=str, default='SASRec',
                       help='Model name')
    parser.add_argument('--dataset', type=str, default='pretrain_corpus',
                       help='Dataset name')
    parser.add_argument('--disable_flops', action='store_true',
                       help='Disable FLOP calculation to avoid H100 compatibility issues')
    
    args = parser.parse_args()
    
    # Check CUDA compatibility first
    print(f"=== CUDA Compatibility Check ===")
    cuda_ok, cuda_msg = check_cuda_compatibility()
    print(f"CUDA Status: {cuda_msg}")
    
    if not cuda_ok:
        print(f"\n⚠️  CUDA Compatibility Issue Detected!")
        print(f"Recommendation: Update PyTorch to support H100 GPUs")
        print(f"Run: uv pip install 'torch>=2.1.0' --index-url https://download.pytorch.org/whl/cu121")
        print(f"Attempting to continue with workarounds...")
        os.environ['DISABLE_FLOP_CALCULATION'] = '1'
        args.disable_flops = True
    
    # Check if we have enough GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        print("No CUDA GPUs available. Cannot run distributed training.")
        sys.exit(1)
    
    if num_gpus < args.nproc:
        print(f"Requested {args.nproc} GPUs but only {num_gpus} available.")
        args.nproc = num_gpus
        print(f"Using {args.nproc} GPUs instead.")
    
    print(f"\n=== Pre-Training GPU Status ===")
    monitor_gpu_usage()
    
    print(f"\n=== Starting RecBole Distributed Training ===")
    print(f"Config file: {args.config_file}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of processes: {args.nproc}")
    print(f"Available GPUs: {num_gpus}")
    print(f"FLOP calculation: {'Disabled' if args.disable_flops else 'Enabled'}")
    print(f"Start time: {datetime.datetime.now()}")
    
    try:
        # Multi-GPU distributed training using run_recboles
        print(f"Running distributed training with {args.nproc} processes...")
        
        # Set up distributed training parameters
        config_dict, checkpoint_dir = setup_distributed_training_config(args)
        print(f"Distributed config: {config_dict}")
        
        # Set up queue for collecting results (optional)
        queue = mp.get_context('spawn').SimpleQueue()
        
        # Prepare kwargs as per RecBole documentation
        kwargs = {
            "config_dict": config_dict,
            "queue": queue  # Optional
        }
        
        print("Launching distributed processes...")
        
        # Launch distributed training using mp.spawn as per RecBole docs
        mp.spawn(
            run_distributed_training_wrapper,
            args=(args.model, args.dataset, [args.config_file], kwargs),
            nprocs=args.nproc,
            join=True
        )
        
        # Collect results if available
        result = None if queue.empty() else queue.get()
        if result:
            print(f"Final result from main process: {result}")
        else:
            print("Distributed training completed. Check logs from process 0 for test results.")
        
        print(f"\n=== Training and Evaluation Completed Successfully ===")
        print(f"End time: {datetime.datetime.now()}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error during script execution: {error_msg}")
        
        if "no kernel image is available" in error_msg:
            print("\n🔧 H100 Compatibility Fix Required:")
            print("1. Update PyTorch: uv pip install 'torch>=2.1.0' --index-url https://download.pytorch.org/whl/cu121")
            print("2. Or try running with: --disable_flops flag")
        
        # Removed obsolete checkpoint error handling
        raise
    
    print(f"\n=== Post-run GPU Status ===")
    monitor_gpu_usage()


if __name__ == '__main__':
    main() 