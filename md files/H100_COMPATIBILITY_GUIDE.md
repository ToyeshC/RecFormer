# H100 GPU Compatibility Guide

This guide outlines the necessary changes to make the code compatible with NVIDIA H100 GPUs.

## Required Changes

### 1. Python Environment
- Use Python 3.10 or higher
- Create a new virtual environment with Python 3.10:
  ```bash
  module load 2023
  module load Python/3.10.4-GCCcore-11.3.0
  python -m venv recformer_env
  source recformer_env/bin/activate
  ```

### 2. Dependencies
Update the dependencies in `pyproject.toml`:
```toml
dependencies = [
    "torch>=2.1.0",          # Required for H100 support
    "pytorch-lightning>=2.0.0",
    "transformers==4.28.0",
    "deepspeed>=0.10.0",     # Updated for H100 compatibility
    "recbole",
    "numpy<2.0.0",           # To avoid bool8 deprecation
    "ray[tune]<2.7.0",       # To ensure compatibility with numpy
    "tensorboardX"           # Required for ray tune logging
]
```

Install the dependencies:
```bash
uv pip install -e .
```

### 3. Running the Code
The code can be run using the distributed training script:
```bash
sbatch jobs/run_baseline_distributed.job
```

The job script is already configured for H100 GPUs with the correct SLURM settings:
```bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
```

## Notes
- PyTorch 2.1.0 or higher is required for H100 support
- The code should now work with H100 GPUs without any additional configuration
- All necessary changes have been integrated into the main codebase 