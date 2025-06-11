#!/usr/bin/env python
"""
GPU monitoring script to check if all GPUs are being utilized
Run this in parallel with your training to monitor GPU usage
"""

import time
import subprocess
import argparse
import datetime


def run_nvidia_smi():
    """Run nvidia-smi and return the output"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error running nvidia-smi: {e}"


def monitor_gpus(interval=10, duration=None):
    """Monitor GPU usage at regular intervals"""
    start_time = time.time()
    iteration = 0
    
    print(f"Starting GPU monitoring at {datetime.datetime.now()}")
    print(f"Monitoring interval: {interval} seconds")
    if duration:
        print(f"Duration: {duration} seconds")
    print("=" * 80)
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if we should stop
            if duration and elapsed > duration:
                break
            
            iteration += 1
            print(f"\n=== GPU Status Check #{iteration} at {datetime.datetime.now()} ===")
            print(f"Elapsed time: {elapsed:.1f} seconds")
            
            # Run nvidia-smi
            output = run_nvidia_smi()
            print(output)
            
            # Sleep until next check
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped by user at {datetime.datetime.now()}")
    except Exception as e:
        print(f"\nError during monitoring: {e}")
    
    total_time = time.time() - start_time
    print(f"\nGPU monitoring completed after {total_time:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU usage during training")
    parser.add_argument('--interval', type=int, default=30, 
                       help='Monitoring interval in seconds (default: 30)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Total monitoring duration in seconds (default: unlimited)')
    
    args = parser.parse_args()
    
    print("GPU Monitoring Tool")
    print("This script will show GPU usage at regular intervals")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    # Initial GPU check
    print("=== Initial GPU Status ===")
    initial_status = run_nvidia_smi()
    print(initial_status)
    
    # Start monitoring
    monitor_gpus(interval=args.interval, duration=args.duration)


if __name__ == '__main__':
    main() 