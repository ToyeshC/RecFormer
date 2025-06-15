# Guide: Fine-Tuning RecFormer on the MIND Dataset

This document outlines the complete pipeline for downloading the MIND dataset and fine-tuning the RecFormer model, including the optimized multi-GPU implementation for Snellius H100 infrastructure.

### 1. Data Acquisition

The MINDsmall dataset is downloaded and extracted into a specific directory structure.

1.  **Download the data:**
    ```bash
    # Create the target directory
    mkdir -p datasets/mind
    cd datasets/mind

    # Download the zipped train and validation sets
    wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip
    wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip
    ```
2.  **Unzip the data:** The archives are extracted into their respective folders.
    ```bash
    unzip MINDsmall_train.zip -d train/
    unzip MINDsmall_dev.zip -d dev/
    ```
This results in the following directory structure:
```
datasets/mind/
├── train/
│   ├── behaviors.tsv
│   └── news.tsv
├── dev/
│   ├── behaviors.tsv
│   └── news.tsv
├── behaviors.tsv (combined)
├── news.tsv (combined)
└── [other files...]
```

### 2. Data Format Transformation

The `process_mind.py` script transforms the raw MIND dataset from TSV format into JSON format that the RecFormer fine-tuning script expects. Here's how the files correspond:

#### File Mapping Table

| **Output JSON File** | **Source Data** | **Purpose** | **Equivalent in datasets/downstream/Arts** |
|---------------------|-----------------|-------------|-------------------------------------------|
| `train.json` | `datasets/mind/behaviors.tsv` (training interactions) | User-item interaction sequences for training | `Arts.train.inter` |
| `val.json` | `datasets/mind/behaviors.tsv` (validation interactions) | User-item interaction sequences for validation | `Arts.valid.inter` |
| `test.json` | `datasets/mind/behaviors.tsv` (test interactions) | User-item interaction sequences for testing | `Arts.test.inter` |
| `meta_data.json` | `datasets/mind/news.tsv` (item metadata) | News article titles, categories, and text content | `Arts.text` |
| `smap.json` | Generated mapping | Maps original news IDs to sequential indices | `Arts.item2index` |

**Key Differences:**
- **MIND**: Uses JSON format with rich text content (titles, abstracts, categories)
- **Arts**: Uses simpler TSV format with basic item features
- **MIND**: News recommendation domain with temporal sequences
- **Arts**: Product recommendation domain with purchase sequences

### 3. The Fine-Tuning Pipeline

The process is orchestrated by three key scripts that work in sequence:

1.  `python_scripts/process_mind.py`: This script is responsible for **data transformation**. It reads the raw `.tsv` files (the item catalog from `news.tsv` and the interaction logs from `behaviors.tsv`) and converts them into a set of `.json` files (`train.json`, `val.json`, `test.json`, `meta_data.json`, `smap.json`). This JSON format is what the model's fine-tuning script is designed to consume.

2.  `python_scripts/finetune.py` (Original): This is the core **model training script** provided by the authors. It loads the pre-trained RecFormer checkpoint (`recformer_seqrec_ckpt.bin`) and continues its training (fine-tunes it) on the processed MIND dataset from the previous step. It handles item encoding, training epochs, and evaluation. **Note**: We've created an optimized version (`finetune_optimized.py`) for multi-GPU H100 training on Snellius.

3.  `jobs/finetune_recformer_mind.job`: This is the master **Slurm job script**. It automates the entire process by calling the two Python scripts in the correct order with the correct parameters.

### 4. How to Run

#### Original Version (Single GPU)
The original pipeline can be executed with:
```bash
sbatch jobs/finetune_recformer_mind.job
```

#### Optimized Version (Multi-GPU H100 on Snellius)
For faster training with 4x H100 GPUs and optimized performance:
```bash
sbatch jobs/finetune_recformer_mind_optimized.job
```

### 5. Key Concepts & Interpretation

**Fine-Tuning vs. Pre-Training**
This process is **fine-tuning**, not pre-training.
*   **Pre-Training** was done by the original authors on a massive, general dataset to teach the model about language and recommendations. We start with their result (`recformer_seqrec_ckpt.bin`).
*   **Fine-Tuning** is what we do: we take that expert model and train it for a little longer on our specific MIND dataset to adapt its expertise to the new domain of news recommendations.

**Interpreting a Successful Run**
A successful job output log will show three main phases:
1.  **`Encode all items`**: The model converts the text of every news article into a numerical vector (embedding) that it can understand.
2.  **`Evaluate` (Baseline)**: Before training, the model is tested to get a baseline score. This score is expected to be low.
3.  **`Training` & `Evaluate` (The Loop)**: The model iterates through the training data to learn, and after each epoch, it is evaluated on a validation set to track progress and save the best version. The final run on the test set provides the official performance metrics for the fine-tuned model.

#### Performance Results Example

Here's an example of the dramatic improvement achieved through fine-tuning on the MIND dataset:

| **Metric** | **Before Fine-tuning (Baseline)** | **After 1 Epoch Fine-tuning** | **Improvement Factor** |
|------------|-----------------------------------|-------------------------------|------------------------|
| NDCG@10 | 5.31e-05 | 0.000705 | **13.3x** |
| Recall@10 | 0.000130 | 0.001449 | **11.1x** |
| NDCG@50 | 0.000159 | 0.002065 | **13.0x** |
| Recall@50 | 0.000649 | 0.008123 | **12.5x** |
| MRR | 0.000144 | 0.001329 | **9.2x** |
| AUC | 0.499 | 0.800 | **1.6x** |

**Key Observations:**
- The baseline scores are extremely low (near zero), confirming that the pre-trained model has no knowledge of news recommendations
- After just 1 epoch of fine-tuning, all metrics show substantial improvement (9-13x better)
- AUC improves from random performance (0.5) to good performance (0.8)
- These results demonstrate the effectiveness of transfer learning from the pre-trained RecFormer model

---

## 6. Snellius H100 Multi-GPU Optimization

### Problem Identification with Original Implementation

The authors' original `finetune.py` script had several performance bottlenecks when running on Snellius H100 infrastructure:

1. **Early Stopping**: ✅ Already implemented (5 epochs patience in stage 1, 3 epochs in stage 2)
2. **Multi-GPU Support**: ❌ Missing - only using 1 GPU out of 4 available H100s
3. **Critical Bottlenecks**:
   - `dataloader_num_workers = 0` (major I/O bottleneck)
   - `batch_size = 8` (too small for powerful H100 GPUs)
   - `preprocessing_num_workers = 8` (underutilizing 32 CPU cores)
   - Item embeddings re-encoded every epoch (expensive operation)
   - No performance optimizations for modern PyTorch versions

### Our Optimized Implementation

We created `python_scripts/finetune_optimized.py` based on the authors' original script with the following enhancements:

#### 1. **Multi-GPU Support (4x H100s)**
- Added `DataParallel` wrapper for model distributed across 4 GPUs
- **Fixed Critical Bug**: Added loss reduction for multi-GPU training
  ```python
  # Handle DataParallel loss reduction
  if isinstance(loss, torch.Tensor) and loss.dim() > 0:
      loss = loss.mean()
  ```
- Proper multi-GPU model state handling for save/load operations
- Compatible with older PyTorch versions on Snellius:
  - Uses `torch.cuda.amp.GradScaler()` instead of `torch.amp.GradScaler('cuda')`
  - Uses `autocast()` instead of `autocast(device_type='cuda')`

#### 2. **Data Loading Optimization**
- **Before**: `dataloader_num_workers = 0` (single-threaded I/O)
- **After**: `dataloader_num_workers = 16` (parallel data loading)
- **Before**: `preprocessing_num_workers = 8`
- **After**: `preprocessing_num_workers = 32` (utilizing all CPU cores)
- Added `pin_memory = True` for faster GPU transfers
- Optimized chunking: `max(1, len(item_meta_list) // (args.preprocessing_num_workers * 8))`

#### 3. **Batch Size & Training Configuration**
- **Before**: `batch_size = 8`, `gradient_accumulation_steps = 8`
- **After**: `batch_size = 32`, `gradient_accumulation_steps = 2`
- **Effective batch size**: 64 per step (optimal for H100s)
- **Total effective batch size**: 256 across 4 GPUs
- Increased default epochs from 16 to 50 for better convergence

#### 4. **Smart Caching & Performance**
- Added `--cache_item_embeddings` to avoid re-encoding items every epoch
- Only re-encode items every 5 epochs instead of every epoch
- Cache tokenized items across epochs
- Optimized item embedding initialization for multi-GPU

#### 5. **Enhanced Two-Stage Training**
The optimized version maintains the authors' sophisticated two-stage approach:

**Stage 1: Initial Training (patience=5)**
- Up to 50 epochs with early stopping after 5 consecutive non-improvements
- Saves best model based on NDCG@10 validation performance
- Model saved to `best_model.bin` (default) or custom `--ckpt` path

**Stage 2: Fine-tuning (patience=3)**
- Loads best model from Stage 1 automatically
- Continues training with tighter early stopping (patience=3)
- Further refines the model with more selective improvements
- Updates the same checkpoint file when validation improves

#### 6. **Resource Allocation**
- **CPUs**: 32 cores (matched preprocessing workers)
- **Memory**: 240GB (sufficient for large batches)
- **GPUs**: 4x H100 (maximum utilization)
- **Time**: 24 hours (increased from 12 for safety)

### Performance Improvements

#### Speed Improvements:
1. **4x GPU parallelization**: ~3-4x faster training per epoch
2. **16x data loading parallelization**: Eliminates I/O bottlenecks  
3. **4x larger effective batch size**: Better convergence, fewer steps per epoch
4. **Item embedding caching**: ~5x faster item encoding (cached every 5 epochs)
5. **32 CPU cores for preprocessing**: ~4x faster tokenization

#### **Estimated Total Speedup: 10-15x faster**

### Hardware Specifications

```bash
# Optimized for Snellius H100 Node:
#SBATCH --gpus-per-node=4      # 4x H100 GPUs
#SBATCH --cpus-per-task=32     # All 32 CPU cores  
#SBATCH --mem=240G             # 240GB RAM
#SBATCH --partition=gpu_h100   # H100 partition
#SBATCH --time=24:00:00        # 24 hour time limit
```

### Usage Examples

**Basic optimized run:**
```bash
sbatch jobs/finetune_recformer_mind_optimized.job
```

**Custom configuration:**
```bash
python python_scripts/finetune_optimized.py \
    --pretrain_ckpt pretrained_models/recformer_seqrec_ckpt.bin \
    --data_path "downstream_datasets/MIND_mini_json" \
    --num_train_epochs 50 \
    --batch_size 32 \
    --fp16 \
    --multi_gpu \
    --gpu_ids "0,1,2,3" \
    --verbose 1 \
    --cache_item_embeddings
```

### Key Files in Optimized Implementation

- `python_scripts/finetune_optimized.py` - Multi-GPU optimized training script
- `jobs/finetune_recformer_mind_optimized.job` - H100 job script with optimized resources
- `best_model.bin` - Saved checkpoint (best model from both training stages)

### Monitoring and Debugging

The optimized version includes comprehensive logging:
- GPU utilization across all 4 devices
- Batch processing throughput per epoch
- Early stopping progression (patience counters)
- Memory usage optimization
- Training convergence metrics
- Stage transitions and model loading confirmations

This optimization transforms the authors' original single-GPU implementation into a production-ready, high-performance training pipeline suitable for modern HPC environments like Snellius.