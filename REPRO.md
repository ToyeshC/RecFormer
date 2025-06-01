# 🔁 Reproducibility Instructions

This document provides the full set of instructions to reproduce our project results from scratch, including data setup, environment configuration, training, and evaluation.

---

## 🧱 Project Structure

```bash
recformer_reproduction_repo/
├── .gitignore
├── LICENSE
├── README.md
├── REPRO.md                
├── pyproject.toml          # Python project configuration
├── requirements.txt
│
├── configs/                # Experiment configuration files
│   ├── recbole_baseline_config.yaml
│   └── recbole_finetune_config.yaml
│
├── jobs/                   
│   ├── job_output/
│   ├── run_baseline.job
│   └── run_finetune.job
│
├── python_scripts/         
│   ├── convert_data....py  
│   ├── convert_fine....py
│   └── run.py              │
└── saved_baselines/        
    ├── SASRec-Jun-01-20....pth
    └── ...

---

## ⚙️ Environment Setup


Setup project by running the following commands:



```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd recformer_reproduction_repo
uv venv recformer_env --python 3.10
source recformer_env/bin/activate
uv pip sync requirements.txt
```

---

## 📂 Download & Prepare Datasets

I downloaded the datasets from the following link: https://github.com/AaronHeee/RecFormer?tab=readme-ov-file

and used the scripts in `python_scripts' to convert them to the RecBole format.

NOTE, For my tests, I've just ran on the train_set and split that one up into train, valid, test. If you want to run properly, you'll need to figure out how to properly use new datasets in RecBole.

---

## ⚙️ Configuration

Set your parameters in the config file before training. They're set for training and finetuning

---

## 🚀 Training

### Baselines

Execute the following slurm jobs:

```bash
sbatch run_baseline.job 
sbatch run_finetune.job
```
Please note the numbers of gpus need to be changed in the job script and the config file. The parameters I used are the ones you see in my job scripts and config files.
---

## 📈 Evaluation

Evaluation happens after training regardless, but you can run a 'evaluate' run by setting the epochs to 0 in the config file.

Note, I completely vibecoded the gini metric in `run.py`; it works, but I don't stand for the code :p

---


## 📦 Dependencies / References

This project repository uses the following frameworks / refers to the following papers:

- https://arxiv.org/pdf/2305.13731
- https://dl.acm.org/doi/pdf/10.1145/3534678.3539381 # the settings for the baselines are taken from this paper


