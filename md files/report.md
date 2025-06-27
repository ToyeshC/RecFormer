# Research Report: Reproducibility and Extension of RecFormer

This document provides a detailed account of the research conducted to reproduce the results of the RecFormer model and extend its application to a new domain, the MIND dataset.

## Introduction

The primary goal of this research is twofold:
1.  **Reproducibility**: To reproduce the results of the RecFormer model on the Amazon datasets, as described in the original paper, "Text Is All You Need: Learning Language Representations for Sequential Recommendation".
2.  **Extension**: To evaluate the performance of RecFormer on a new, out-of-domain dataset, the Microsoft News Dataset (MIND), and compare it against established baseline models.

This report details the experimental setup, including dataset preparation, model configurations, and the scripts used to run the experiments.

---

## Part 1: Reproducing RecFormer on Amazon Datasets

This section describes the steps taken to reproduce the results on the Amazon datasets.

### 1.1 Datasets

The experiments were conducted on several Amazon product review datasets. These datasets are located in the `datasets/downstream/` directory. Each subdirectory represents a different dataset category.

The directory structure is as follows:
```
datasets/downstream/
├── Arts/
├── Instruments/
├── Office/
├── Pantry/
├── Scientific/
...
```

Each dataset directory contains files in a format suitable for the baseline models (RecBole). For example, the `datasets/downstream/Instruments/` directory contains:
```
datasets/downstream/Instruments/
├── Instruments.feat1CLS
├── Instruments.item2index
├── Instruments.test.inter
├── Instruments.text
├── Instruments.train.inter
├── Instruments.user2index
└── Instruments.valid.inter
```
- `*.inter` files contain the user-item interaction data.
- `*.item2index` and `*.user2index` contain mappings from raw IDs to integer indices.

### 1.2. Running Baselines

A suite of baseline models was run on the Amazon datasets to establish a performance benchmark. The execution is handled by `python_scripts/run.py`, which acts as a wrapper for the RecBole library.

**Execution Script**: `python_scripts/run.py`

This script is responsible for:
- Initializing a RecBole `Config` object from a given YAML file.
- Creating the dataset and preparing the data splits (train, validation, test).
- Handling both standard RecBole models and custom models like `FDSA`.
- Training the model using the specified trainer.
- Evaluating the model on the test set to get standard metrics (e.g., NDCG, Recall).
- **Custom Metric**: It includes a custom implementation to calculate the **Gini Coefficient** for the recommended items' categories, providing a measure of recommendation diversity.
- **Fine-tuning Mode**: Supports a `--load_model` flag which, when used with a `pretrained_model_path` in the config, allows for fine-tuning a pre-trained baseline model.

**Job Script**: `jobs/run_baseline_mig.job`

This script was used to launch the baseline model experiments. While the provided script is configured for the `UniSRec` model on a single dataset, it was adapted to run other models and on other datasets by modifying the configuration.

**Script Content**:
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=output_%A.out
#SBATCH --output=job_output/run_pet_unisrec_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
NPROC=1

# activate venv
mkdir -p jobs/job_output

source ../../recformer_env/bin/activate

# Run the main script with the config file
echo "Run UnisRec on Instruments dataset"
python python_scripts/run.py \
    --config_file=configs/UniSRec.yaml \
    --nproc=$NPROC \
    --model_name=UniSRec

# Other models run by uncommenting and modifying:
# echo "Run FDSA on Instruments dataset"
# python python_scripts/run.py \
#     --config_file=configs/FDSA.yaml \
#     --nproc=$NPROC \
#     --model_name=FDSA

# echo "Run SASRec on Instruments dataset"
# python python_scripts/run.py \
#     --config_file=configs/SASRec.yaml \
#     --nproc=$NPROC \
#     --model_name=SASRec
```

**Execution Details**:
- The script runs `python_scripts/run.py`.
- The `--config_file` argument points to a YAML file containing the model's hyperparameters.
- The `--model_name` argument specifies the model to run (`UniSRec`, `FDSA`, `SASRec`).
- To run on different datasets, the `dataset` parameter within the respective `.yaml` config files was modified.

### 1.3. Fine-tuning RecFormer

Two primary approaches were explored for fine-tuning the RecFormer model. The initial method used pre-computed, cached embeddings for efficiency, while the final, improved method uses an on-the-fly encoding strategy to maximize performance.

#### 1.3.1. Initial Approach: Cached Item Embeddings

The first approach focused on minimizing computational overhead and preventing potential out-of-memory (OOM) errors on large datasets. This was achieved by pre-encoding all items in the dataset and caching their embeddings.

**Methodology**:
The `python_scripts/finetune_optimized.py` script implements this logic. Before the training process begins, it iterates through every item in the dataset, computes its embedding using the RecFormer model, and stores it in memory. During training and evaluation, the model retrieves these pre-computed embeddings instead of performing redundant encoding operations.

**Trade-offs**:
-   **Advantage**: This method is highly efficient. By encoding each item only once per experiment, it significantly reduces the computational load within the training loop, leading to faster epochs.
-   **Disadvantage**: The primary drawback is a potential reduction in model performance. Since the item embeddings are computed only once with the initial pre-trained model, they do not evolve as the model's weights are updated during fine-tuning. These "stale" embeddings can limit the model's ability to learn nuanced representations based on the downstream task, as the item encoder part of the model is not being trained jointly with the sequence recommendation head.

**Job Script**: `jobs/finetune_recformer_amazon_optimized.job`
This job script was used to launch experiments with the cached embedding approach.

#### 1.3.2. Improved Approach: On-the-Fly Item Encoding

To address the performance limitations of stale embeddings, a more dynamic approach was developed. This "on-the-fly" method ensures that item embeddings are always generated using the most up-to-date model weights.

**Methodology**:
This approach is implemented in `python_scripts/finetune_recformer_mind_optimized_onthefly.py`. Instead of pre-computing all embeddings, it uses a **smart caching** strategy (`SmartItemCache`).
1.  For each training batch, the script identifies the unique items required.
2.  It attempts to retrieve the embeddings for these items from an in-memory LRU (Least Recently Used) cache.
3.  If an item's embedding is not in the cache (a "miss"), the item is encoded in real-time using the current state of the model.
4.  The newly generated embedding is then used for the forward pass and stored in the cache for potential reuse in subsequent steps.

**Trade-offs**:
-   **Advantage**: This method leads to superior model performance. Because the item encoder is actively used during training, its weights are updated, and the generated embeddings are always current. This allows the model to learn more task-specific and effective item representations.
-   **Disadvantage**: This approach is more computationally intensive than the cached method, as it involves performing encoding operations throughout the training process. The `SmartItemCache` mitigates some of this overhead but does not eliminate it entirely.

**Job Script**: `jobs/finetune_recformer_mind_onthefly.job`
While named for the MIND dataset, this job script template illustrates the launch process for the on-the-fly methodology. It first converts data to a suitable format and then calls the on-the-fly training script.

**Key Script Arguments (`finetune_recformer_mind_optimized_onthefly.py`)**
- `cache_size`: The maximum number of item embeddings to store in the LRU cache.
- `max_len`: The maximum sequence length for user histories.

**Job Script**: `jobs/finetune_recformer_amazon_optimized.job`

**Script Content**:
```bash
#!/bin/bash
#SBATCH --job-name=finetune_recformer_mind_optimized
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=240G
#SBATCH --time=15:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=job_output/finetune_recformer_instruments_2_%j.out

# ... (environment setup) ...

python python_scripts/finetune_optimized.py \
    --pretrain_ckpt pretrained_models/recformer_seqrec_ckpt.bin \
    --data_path finetune_data/Instruments \
    --num_train_epochs 50 \
    --batch_size 32 \
    --fp16 \
    --multi_gpu \
    --gpu_ids "0,1,2,3" \
    --finetune_negative_sample_size -1 \
    --verbose 1 \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 32 \
    --pin_memory \
    --cache_item_embeddings \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --min_inter 0
```
*Note: The job name in the script is `finetune_recformer_mind_optimized`, but it was used for Amazon datasets by changing `--data_path`.*

**Execution Details**:
- The job script launches `python_scripts/finetune_optimized.py`.
- This script was run for each Amazon dataset by changing the `--data_path` argument (e.g., `finetune_data/Instruments`, `finetune_data/Scientific`, etc.).
- **Pretrained Checkpoint**: `pretrained_models/recformer_seqrec_ckpt.bin`
- **Key Hyperparameters**:
    - `num_train_epochs`: 50
    - `batch_size`: 32 (per GPU)
    - `gradient_accumulation_steps`: 2 (Effective batch size: 32 * 4 GPUs * 2 = 256)
    - `learning_rate`: 5e-5

---

## Part 2: Extending RecFormer to the MIND Dataset

This section details the extension of the research to the MIND dataset.

### 2.1. MIND Dataset

The MIND dataset was used to evaluate RecFormer's performance in a different domain (news recommendations).

**Raw Data Location**: `datasets/mind/`

The raw dataset has the following structure:
```
datasets/mind/
├── behaviors.tsv
├── dev/
│   ├── behaviors.tsv
│   └── news.tsv
├── entity_embedding.vec
├── news.tsv
├── relation_embedding.vec
└── train/
    ├── behaviors.tsv
    └── news.tsv
```
- `news.tsv`: Contains metadata for the news articles (items).
- `behaviors.tsv`: Contains user interaction data (clicks).

### 2.2. Processing MIND for RecFormer

The raw MIND dataset was first processed into a format suitable for RecFormer fine-tuning. Two processing pipelines were used, corresponding to the two fine-tuning approaches.

#### 2.2.1. Processing for Cached Fine-Tuning

**Processing Script**: `python_scripts/process_mind.py`
This script converts the raw `behaviors.tsv` and `news.tsv` files into a set of JSON files (`train.json`, `val.json`, `test.json`). It performs a chronological split for each user, creating training sequences and corresponding validation/test targets.

#### 2.2.2. Processing for On-the-Fly Fine-Tuning

**Processing Script**: `python_scripts/convert_mind_to_amazon.py`
The on-the-fly training script is optimized for data in the "Amazon format" (where each line is a `user_id`, `item_id_list`, and `target_item_id`). This script converts the RecBole-formatted MIND data into this more efficient structure.

### 2.3. Fine-tuning RecFormer on MIND

Both fine-tuning strategies were applied to the MIND dataset.

#### 2.3.1. Initial Approach: Cached Item Embeddings

The cached embedding approach, as described in Section 1.3.1, was run on the JSON-formatted MIND data.

**Job Script**: `jobs/finetune_recformer_mind_optimized.job`
This script first calls `python_scripts/process_mind.py` to prepare the data and then launches `python_scripts/finetune_optimized.py`.

#### 2.3.2. Improved Approach: On-the-Fly Item Encoding

The superior on-the-fly approach, as described in Section 1.3.2, was run on the Amazon-formatted MIND data to achieve the best performance.

**Job Script**: `jobs/finetune_recformer_mind_onthefly.job`

**Script Content**:
```bash
#!/bin/bash
#SBATCH --job-name=finetune_recformer_mind_onthefly
# ... (resource allocation) ...

# --- Step 1: Convert MIND dataset to Amazon format ---
python python_scripts/convert_mind_to_amazon.py

# --- Step 2: Run On-The-Fly RecFormer fine-tuning ---
python python_scripts/finetune_recformer_mind_optimized_onthefly.py \
    --batch_size 32 \
    --num_epochs 5 \
    --lr 1e-4 \
    --max_len 50 \
    --cache_size 10000 \
    --output_dir "output/recformer_mind_mini_onthefly"
```

**Execution Details**:
- The job first runs the conversion script to prepare data in the Amazon format.
- It then launches the on-the-fly fine-tuning script (`finetune_recformer_mind_optimized_onthefly.py`) with specific hyperparameters for the MIND dataset, such as `cache_size` and `max_len`.
- This represents the final and most effective method for fine-tuning RecFormer on the MIND dataset.

### 2.4. Preparing MIND for Baselines

To run the baseline models on the MIND dataset, it was necessary to convert the data into the RecBole format.

**Conversion Script**: `python_scripts/convert_mind_to_recbole.py`

This script takes the JSON-formatted MIND data (produced by `process_mind.py`) and converts it into RecBole's standard file formats:
- **Interaction File (`.inter`)**: It reads `train.json`, `val.json`, and `test.json` and transforms them into the RecBole format: `user_id:token\titem_id:token\ttimestamp:float`. A synthetic timestamp is generated based on the item's position in the user's interaction sequence.
- **Item File (`.item`)**: It reads `meta_data.json` and creates a tab-separated `.item` file with headers like `item_id:token`, `title:token_seq`, etc.

The conversion is initiated from the `jobs/run_baselines_mind.job` script.

**Output Location**: `recbole_data/MIND/`

The resulting directory structure is:
```
recbole_data/MIND/
├── MIND.inter
├── MIND.item
├── MIND.test.inter
├── MIND.train.inter
└── MIND.valid.inter
```

### 2.5. Running Baselines on MIND

The baseline models were then run on the converted MIND dataset using the `python_scripts/run.py` script.

**Job Script**: `jobs/run_baselines_mind.job`

**Script Content**:
```bash
#!/bin/bash
#SBATCH --job-name=baselines_mind
#SBATCH --output=job_output/baselines_mind_%A.out
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=240G

# ... (environment and data conversion steps) ...

# Step 2: Run Baseline Models on MIND
MODELS=("SASRec" "FDSA" "UniSRec")

for MODEL in "${MODELS[@]}"; do
    echo "=== Running $MODEL on MIND dataset ==="
    
    # Determine config file based on model
    if [ "$MODEL" == "SASRec" ]; then
        CONFIG_FILE="configs/mind_sasrec_config.yaml"
    elif [ "$MODEL" == "FDSA" ]; then
        CONFIG_FILE="configs/mind_fdsa_config.yaml"
    elif [ "$MODEL" == "UniSRec" ]; then
        CONFIG_FILE="configs/mind_unisrec_simple.yaml"
    fi
    
    # Run the model
    python python_scripts/run.py \
        --config_file="$CONFIG_FILE" \
        --nproc=$NPROC \
        --model_name="$MODEL"
done
```

**Execution Details**:
- The script first calls `python_scripts/convert_mind_to_recbole.py` to prepare the data.
- It then iterates through a list of models (`SASRec`, `FDSA`, `UniSRec`) and launches `python_scripts/run.py` for each.
- Each model uses a dedicated configuration file (e.g., `configs/mind_sasrec_config.yaml`) that points to the converted MIND dataset in `recbole_data/MIND/`.

---

## Part 3: Experimental Results

### 3.1. Amazon Dataset Reproducibility Results

The following table presents a comparison between the original results reported in the paper and the results reproduced in this study. (O) denotes Original results, and (R) denotes Reproduced results.

**Table 1: Comparison of Original and Reproduced Results**
| Dataset | Metric | SASR (O) | SASR (R) | FDSA (O) | FDSA (R) | UniSR (O) | UniSR (R) | RecF (O) | RecF (R) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scientific** | **NDCG@10** | 0.0797 | **0.0598** | 0.0716 | **0.0606** | 0.0862 | **0.0685** | 0.1027 | **0.1040** |
| | **Recall@10** | 0.1305 | **0.1137** | 0.0967 | **0.0945** | 0.1255 | **0.1388** | 0.1448 | **0.1469** |
| | **MRR** | 0.0696 | **0.0431** | 0.0692 | **0.0505** | 0.0786 | **0.0467** | 0.0951 | **0.0965** |
| **Instruments** | **NDCG@10** | 0.0634 | **0.0758** | 0.0731 | **0.0947** | 0.0785 | **0.0823** | 0.0830 | **0.0805** |
| | **Recall@10** | 0.0995 | **0.1342** | 0.1006 | **0.1322** | 0.1119 | **0.1462** | 0.1052 | **0.1012** |
| | **MRR** | 0.0577 | **0.0574** | 0.0748 | **0.0832** | 0.0740 | **0.0522** | 0.0807 | **0.0785** |
| **Arts** | **NDCG@10** | 0.0848 | **0.0656** | 0.0994 | **0.0775** | 0.0894 | **0.0709** | 0.1252 | **0.1243** |
| | **Recall@10** | 0.1342 | **0.1209** | 0.1209 | **0.1153** | 0.1333 | **0.1386** | 0.1614 | **0.1603** |
| | **MRR** | 0.0742 | **0.0483** | 0.0941 | **0.0659** | 0.0798 | **0.0498** | 0.1189 | **0.1182** |
| **Office** | **NDCG@10** | 0.0832 | **0.0747** | 0.0922 | **0.0957** | 0.0919 | **0.0811** | 0.1141 | **0.1274** |
| | **Recall@10** | 0.1196 | **0.1246** | 0.1285 | **0.1286** | 0.1262 | **0.1419** | 0.1403 | **0.1613** |
| | **MRR** | 0.0751 | **0.0588** | 0.0972 | **0.0855** | 0.0848 | **0.0617** | 0.1089 | **0.1207** |
| **Games** | **NDCG@10** | 0.0547 | **0.0520** | 0.0600 | **0.0588** | 0.0580 | **0.0576** | 0.0684 | **0.0679** |
| | **Recall@10** | 0.0953 | **0.1124** | 0.0931 | **0.1075** | 0.0923 | **0.1249** | 0.1039 | **0.1034** |
| | **MRR** | 0.0505 | **0.0337** | 0.0546 | **0.0441** | 0.0552 | **0.0372** | 0.0650 | **0.0643** |
| **Pet** | **NDCG@10** | 0.0569 | **0.0740** | 0.0673 | **0.0813** | 0.0702 | **0.0808** | 0.0972 | **0.0977** |
| | **Recall@10** | 0.0881 | **0.1142** | 0.0949 | **0.1137** | 0.0933 | **0.1242** | 0.1162 | **0.1176** |
| | **MRR** | 0.0507 | **0.0616** | 0.0650 | **0.0632** | 0.0650 | **0.0674** | 0.0940 | **0.0944** |

### 3.2. MIND Dataset Extension Results

The following table shows the performance of the baseline models and RecFormer on the MIND dataset.

**Table 2: Test Set Performance Metrics for SASRec (50 Epochs), FDSA (50 Epochs), and RecFormer (11 Epochs).**
*Notably, RecFormer outperforms the other models despite training for significantly fewer epochs.*

| Metric | SASRec | FDSA | RecFormer |
| :--- | :--- | :--- | :--- |
| **Recall@10** | 0.1534 | 0.1506 | 0.3249 |
| **NDCG@10** | 0.0777 | 0.0786 | 0.2171 |
| **MRR@10** | 0.0549 | 0.0568 | 0.1952 |


---

## Part 4: Core Technologies and Libraries

This research leverages several key open-source libraries to implement the data processing pipelines, baseline models, and the RecFormer architecture. This section details the specific tools and techniques used.

### 4.1. RecBole

The RecBole library formed the backbone for running all baseline experiments (`SASRec`, `FDSA`, `UniSRec`). Its high-level APIs enabled rapid and standardized evaluation of these models.

-   **Experiment Execution**: The primary entry point was `recbole.quick_start.run_recbole`, which handles the entire training and evaluation pipeline based on a configuration file.
-   **Configuration Management**: Model hyperparameters, dataset paths, and training settings were managed using RecBole's `recbole.config.Config` class, which parses the project's YAML configuration files.
-   **Data Handling**: The `recbole.data.create_dataset` and `recbole.data.data_preparation` functions were used to load data from the specified format and create the necessary data loaders for training and evaluation.

### 4.2. PyTorch

PyTorch was the fundamental deep learning framework used for implementing and training the custom RecFormer model.

-   **Model Architecture**: The `RecformerForSeqRec` model is a custom `torch.nn.Module`, allowing for a flexible implementation of the paper's architecture.
-   **Custom Training Loop**: Unlike the RecBole baselines, the RecFormer fine-tuning was performed using a manual training loop. This provided granular control over the training process, including manual gradient accumulation, optimizer steps (`optimizer.step()`), and learning rate scheduler updates (`scheduler.step()`).
-   **Performance Optimization**:
    -   **Mixed-Precision Training**: `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler` were employed to enable FP16 training, which reduces memory consumption and speeds up computation on compatible hardware (e.g., A100, H100 GPUs).
    -   **Multi-GPU Training**: The initial cached-based fine-tuning script utilized `torch.nn.DataParallel` to distribute training across multiple GPUs.
-   **Utilities**: The `pytorch_lightning.seed_everything` function was used to ensure bit-for-bit reproducibility across runs.

### 4.3. Hugging Face Transformers

The `transformers` library was central to the RecFormer model, providing the underlying language model architecture and tokenization capabilities.

-   **Base Model**: The RecFormer model is built upon the `allenai/longformer-base-4096` model. This choice is critical, as the Longformer architecture is specifically designed to handle long input sequences, which is highly relevant for modeling extensive user interaction histories.
-   **Tokenization**: The project's custom `RecformerTokenizer` wraps the fast, Rust-based tokenizers from the `transformers` library, used to convert raw item text attributes into the input IDs and token type IDs required by the model.

### 4.4. Pandas

The `pandas` library was essential for the data ingestion and manipulation pipelines, particularly for converting the MIND dataset.

-   **Data Ingestion**: `pandas.read_csv` was used to efficiently read the large, tab-separated `.tsv` files (`behaviors.tsv`, `news.tsv`) from the raw MIND dataset.
-   **Data Transformation**: The data processing scripts (`process_mind.py`, `convert_mind_to_amazon.py`) heavily rely on Pandas DataFrames to group interactions by user, construct chronological sequences, and transform the data into the required JSON or Amazon-style formats for fine-tuning.

## Conclusion

This report has outlined the comprehensive methodology used to reproduce and extend the RecFormer model. By detailing the purpose and key functionalities of the datasets, execution scripts, core libraries, and hyperparameters, this work provides a clear path for others to replicate these findings. The experiments cover both the original Amazon datasets and the new MIND dataset, providing a thorough evaluation of the RecFormer model's capabilities. 