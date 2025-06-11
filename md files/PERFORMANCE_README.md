# RecBole Distributed Training Performance Tuning

This document outlines the performance observations and optimizations applied to the RecBole distributed training pipeline for this project.

## Changes from Single-GPU to Multi-GPU Setup

To transition from the original `python_scripts/run.py` single-GPU setup to a distributed 4-GPU H100 configuration, the following key changes were implemented:

### Core Script Changes
- **New Distributed Script**: Created `python_scripts/run_distributed.py` to replace `run.py`, utilizing RecBole's `mp.spawn` and `run_recboles` functions for proper distributed training.
- **Synchronization Barrier**: Added a monkey-patched synchronization barrier in the `Trainer.fit` method to prevent race conditions during checkpoint loading between training and testing phases.
- **PyTorch Compatibility**: Applied a compatibility patch for PyTorch 2.7+ to handle `torch.load` calls with `weights_only=False` for RecBole checkpoints.

### Configuration Updates
- **Distributed Parameters**: Added distributed training settings to the config file including `world_size`, `nproc`, `gpu_id`, and checkpoint management.
- **Job Script**: Updated `jobs/run_baseline_distributed.job` to use 4 H100 GPUs with the new distributed script.

### Hardware Compatibility
- **H100 Support**: Implemented workarounds for H100 GPU compatibility issues, including disabling FLOP calculation via `benchmark_filename: null`.

## Performance Tuning and Observations

### Observation 1: Training on 4 H100s was not faster than on 1 A100.

-   **Reason**: **Communication Overhead.** With 4 GPUs, the `train_batch_size` of `1024` was split, giving each GPU only 256 items per step. For a powerful H100, this is a trivial amount of work. The GPUs spent more time communicating gradients and waiting for each other than doing useful computation.
-   **Solution**: **Increase the workload.** Following the standard practice of linear scaling, we quadrupled the `train_batch_size` to `4096`. This ensures each GPU processes a substantial batch of 1024 items, making the computation time significant relative to the communication time.
-   **Result**: The training time for one epoch on the full dataset dropped by **over 50%** (from ~36 minutes to ~17 minutes).

### Observation 2: The number of training steps per epoch decreased from ~20k to ~5k.

-   **Reason**: This is an expected and direct consequence of the larger batch size.
-   **Formula**: `Number of Steps = Total Training Samples / Batch Size`
-   Since we made the batch size 4x larger, the number of steps required to complete an epoch became 4x smaller. The progress bar now shows the global progress of the entire 4-GPU system.
