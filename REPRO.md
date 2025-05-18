# 🔁 Reproducibility Instructions

This document provides the full set of instructions to reproduce our project results from scratch, including data setup, environment configuration, training, and evaluation.

---

## 🧱 Project Structure

```bash
├── recformer_reproduction_repo/  
│   ├── recformer_env/            
│   ├── data/                     
│   │   └── amazon-electronics/   # Example dataset
│   ├── job_output/               # SLURM output
│   ├── saved_baselines/          # Checkpoints for the RecBole baseline model
│   ├── run.py                    # Script to run RecBole
│   ├── recbole_baseline_config.yaml # Configuration for RecBole
│   ├── requirements.txt       
│   ├── recformer_job.sbatch      # Example job script
│   ├── README.md                 # Main project README
│   └── REPRO.md             
```

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

Datasets are downloaded by RecBole.

---

## ⚙️ Configuration

Set your parameters in the config file before training.

---

## 🚀 Training

### Baselines

Run the following command to train the baseline:

```bash
python run.py --config_file recbole_baseline_config.yaml
```

To perform inference:

```bash
python XXXX
```

Alternatively, execute the following slurm jobs:

```bash
sbatch 
sbatch job_scripts/infer_xxxxx.job
```

---

## 📈 Evaluation

After training, evaluate all models with:

```bash
python XXXX
```

---


## 📎 Misc. Notes (optional)

---

## 📦 Dependencies / References

This project repository uses the following frameworks / refers to the following papers:

- XXX
- XXX


