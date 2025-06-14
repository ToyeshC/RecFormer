# Guide: Fine-Tuning RecFormer on the MIND Dataset

This document outlines the successful, final pipeline for downloading the MIND dataset and fine-tuning the RecFormer model.

### 1. Data Acquisition

The MINDsmall dataset is downloaded and extracted into a specific directory structure.

1.  **Download the data:**
    ```bash
    # Create the target directory
    mkdir -p data
    cd data

    # Download the zipped train and validation sets
    wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip
    wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip
    ```
2.  **Unzip the data:** The archives are extracted into their respective folders.
    ```bash
    unzip MINDsmall_train.zip -d MINDsmall_train/
    unzip MINDsmall_dev.zip -d MINDsmall_dev/
    ```
This results in the following directory structure:
```
data/
├── MINDsmall_train/
│   ├── behaviors.tsv
│   └── news.tsv
└── MINDsmall_dev/
    ├── behaviors.tsv
    └── news.tsv
```

### 2. The Fine-Tuning Pipeline

The process is orchestrated by three key scripts that work in sequence:

1.  `python_scripts/process_mind.py`: This script is responsible for **data transformation**. It reads the raw `.tsv` files (the item catalog from `news.tsv` and the interaction logs from `behaviors.tsv`) and converts them into a set of `.json` files (`train.json`, `val.json`, `test.json`, `meta_data.json`, `smap.json`). This JSON format is what the model's fine-tuning script is designed to consume.

2.  `python_scripts/finetune.py`: This is the core **model training script**. It loads the pre-trained RecFormer checkpoint (`recformer_seqrec_ckpt.bin`) and continues its training (fine-tunes it) on the processed MIND dataset from the previous step. It handles item encoding, training epochs, and evaluation.

3.  `jobs/finetune_recformer_mind.job`: This is the master **Slurm job script**. It automates the entire process by calling the two Python scripts in the correct order with the correct parameters.

### 3. How to Run

The entire pipeline can be executed with a single command, which submits the master job script to the Slurm scheduler:

```bash
sbatch jobs/finetune_recformer_mind.job
```

### 4. Key Concepts & Interpretation

**Fine-Tuning vs. Pre-Training**
This process is **fine-tuning**, not pre-training.
*   **Pre-Training** was done by the original authors on a massive, general dataset to teach the model about language and recommendations. We start with their result (`recformer_seqrec_ckpt.bin`).
*   **Fine-Tuning** is what we do: we take that expert model and train it for a little longer on our specific MIND dataset to adapt its expertise to the new domain of news recommendations.

**Interpreting a Successful Run**
A successful job output log will show three main phases:
1.  **`Encode all items`**: The model converts the text of every news article into a numerical vector (embedding) that it can understand.
2.  **`Evaluate` (Baseline)**: Before training, the model is tested to get a baseline score. This score is expected to be low.
3.  **`Training` & `Evaluate` (The Loop)**: The model iterates through the training data to learn, and after each epoch, it is evaluated on a validation set to track progress and save the best version. The final run on the test set provides the official performance metrics for the fine-tuned model. 