# RecFormer: Text-Based Sequential Recommendation

This repository contains the implementation of RecFormer, a text-based sequential recommendation model, along with baseline models and evaluation scripts.

## Project Structure

```
.
├── configs/           # Configuration files for models
├── data/             # Data processing utilities
├── dataset/          # Dataset storage and processing scripts
├── jobs/             # SLURM job scripts
├── models/           # Model implementations
├── python_scripts/   # Main Python implementation scripts
├── RESULTS/          # Experimental results and outputs
├── .gitignore       # Git ignore file
├── LICENSE          # Project license
├── pyproject.toml   # Project configuration and dependencies
├── REPRO.md         # Reproduction instructions
└── requirements.txt # Python package dependencies
```

## Environment Setup

1. Create and activate a virtual environment:
```bash
python -m venv recformer_env
source recformer_env/bin/activate
```

2. Install required packages (requirements.txt will be provided)

## Available Models

1. **RecFormer**
   - Text-based sequential recommendation model
   - Supports both cached and on-the-fly item encoding

2. **Baseline Models**
   - SASRec
   - FDSA
   - UniSRec

## Datasets

### 1. Amazon Product Reviews
Located in `datasets/downstream/`:
- Arts
- Instruments
- Office
- Pantry
- Scientific
and more...

Each dataset contains:
- `*.inter`: User-item interaction data
- `*.item2index`: Item ID to index mapping
- `*.user2index`: User ID to index mapping
- `*.text`: Item text data
- `*.feat1CLS`: Feature data

### 2. MIND (Microsoft News Dataset)
Located in `datasets/MIND_mini/`

## Running Experiments

### 1. Running Baseline Models

```bash
python python_scripts/run.py \
    --config_file=configs/<model_config>.yaml \
    --nproc=1 \
    --model_name=<model_name>
```

Available model configurations:
- `configs/SASRec.yaml`
- `configs/FDSA.yaml`
- `configs/UniSRec.yaml`

### 2. Fine-tuning RecFormer

#### a. Cached Embeddings Approach
```bash
python python_scripts/finetune_optimized.py \
    --pretrain_ckpt pretrained_models/recformer_seqrec_ckpt.bin \
    --data_path finetune_data/<dataset> \
    --num_train_epochs 50 \
    --batch_size 32 \
    --fp16 \
    --multi_gpu \
    --gpu_ids "0,1,2,3" \
    --cache_item_embeddings
```

#### b. On-the-Fly Encoding Approach
```bash
python python_scripts/finetune_recformer_mind_optimized_onthefly.py \
    --batch_size 32 \
    --num_epochs 5 \
    --lr 1e-4 \
    --max_len 50 \
    --cache_size 10000
```

## SLURM Job Scripts

The `jobs/` directory contains SLURM job scripts for different experiments:

1. `finetune_recformer_amazon_optimized.job`: Fine-tuning RecFormer on Amazon datasets
   - Uses 4 H100 GPUs
   - 240GB memory
   - 32 CPUs

2. `finetune_recformer_mind_onthefly.job`: Fine-tuning with on-the-fly encoding
   - Uses 4 H100 GPUs
   - 320GB memory
   - Optimized for memory efficiency

3. `run_baselines_mind.job`: Running baseline models on MIND dataset
   - Supports SASRec, FDSA, and UniSRec
   - Includes data conversion to RecBole format

4. `run_baseline_mig.job` and `run_baseline_mig_diversity.job`: Running baseline models
   - Single GPU setup
   - Supports different baseline models

## Key Features

1. **Smart Caching Strategy**
   - LRU (Least Recently Used) cache for item embeddings
   - Configurable cache size
   - Dynamic encoding for cache misses

2. **Multi-GPU Support**
   - Gradient accumulation for effective batch size control
   - Optimized memory management
   - FP16 training support

3. **Custom Metrics**
   - Standard metrics (NDCG, Recall)
   - Gini Coefficient for recommendation diversity

## Performance Optimization

1. Memory Management:
   - Configurable cache sizes
   - Gradient accumulation
   - FP16 training

2. Training Efficiency:
   - Multi-GPU support
   - Optimized data loading
   - Smart caching strategies

## Notes

- For large datasets, use the on-the-fly encoding approach for better model performance
- Adjust batch size and cache size based on available GPU memory
- Monitor GPU memory usage when working with large datasets
- Use gradient accumulation to maintain effective batch sizes while managing memory 