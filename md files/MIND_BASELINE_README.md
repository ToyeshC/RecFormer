# Running Baselines on MIND Dataset

This guide explains how to run baseline models (SASRec, FDSA, UniSRec, BERT4Rec) on the MIND dataset and compare their performance with RecFormer.

## Dataset Format Differences

### Amazon Dataset (e.g., Arts)
- **Format**: RecBole format with structured files
- **Files**: `.train.inter`, `.test.inter`, `.valid.inter`, `.item`, `.feat1CLS`
- **Structure**: `user_id`, `item_id_list`, `item_id` with sequential interactions
- **Text Features**: Pre-computed BERT embeddings in `.feat1CLS` files

### MIND Dataset  
- **Format**: JSON format for news recommendation
- **Files**: `train.json`, `val.json`, `test.json`, `meta_data.json`, `smap.json`
- **Structure**: `[user_id, item_id, interaction_type]` tuples
- **Text Features**: Raw news article text in `meta_data.json`

## Prerequisites

1. **Environment**: Ensure RecFormer environment is activated
2. **RecFormer Results**: You should have already run RecFormer fine-tuning on MIND using `jobs/finetune_recformer_mind_optimized.job`
3. **MIND Dataset**: MIND_mini_json dataset should be in `downstream_datasets/MIND_mini_json/`

## Step-by-Step Instructions

### 1. Convert MIND Dataset to RecBole Format

First, convert the MIND JSON format to RecBole format that baseline models can use:

```bash
python python_scripts/convert_mind_to_recbole.py \
    --input_dir downstream_datasets/MIND_mini_json \
    --output_dir recbole_data/MIND \
    --dataset_name MIND
```

### 2. Run All Baselines (Recommended)

Use the comprehensive job script to run all baseline models:

```bash
sbatch jobs/run_baselines_mind.job
```

This will:
- Convert MIND dataset to RecBole format
- Run SASRec, FDSA, UniSRec, and BERT4Rec sequentially
- Save results in respective directories

### 3. Run Individual Baselines (Optional)

If you prefer to run models individually:

```bash
# Run SASRec
sbatch jobs/run_sasrec_mind.job

# Run FDSA  
sbatch jobs/run_fdsa_mind.job

# Run UniSRec
sbatch jobs/run_unisrec_mind.job

# Run BERT4Rec (using existing config)
sbatch jobs/run_bert4rec_mind.job  # Create this if needed
```

### 4. Compare Results

After all models have finished, compare their performance:

```bash
python python_scripts/compare_mind_results.py \
    --recformer_log mind_finetuning_log.md \
    --baseline_dir saved_baselines \
    --output_file mind_results_comparison.csv
```

## Configuration Files

Each baseline model has a dedicated config file for MIND:

- `configs/mind_sasrec_config.yaml` - SASRec configuration
- `configs/mind_fdsa_config.yaml` - FDSA configuration  
- `configs/mind_unisrec_config.yaml` - UniSRec configuration
- `configs/mind_bert4rec_config.yaml` - BERT4Rec configuration (existing)

## Expected Output Locations

Results will be saved in:
- **SASRec**: `./saved_baselines/mind_sasrec/`
- **FDSA**: `./saved_baselines/mind_fdsa/`
- **UniSRec**: `./saved_baselines/mind_unisrec/`
- **BERT4Rec**: `./saved_baselines/mind/`
- **RecFormer**: Results documented in `mind_finetuning_log.md`

## Key Differences in Configuration

### FDSA Adaptations for MIND
- Uses `title` field instead of `feat1CLS` (Amazon's pre-computed embeddings)
- Adapter layers configured for text processing: `[768, 300]`
- Selected features: `[title]` instead of `[class]`

### Text Processing
- Amazon datasets use pre-computed BERT embeddings (`.feat1CLS`)
- MIND dataset uses raw text that gets processed during training
- News articles typically have longer text content than product descriptions

## Training Settings

All baseline models use consistent settings:
- **Epochs**: 50 (with early stopping after 10 epochs without improvement)
- **Batch Size**: 512 for both training and evaluation
- **Learning Rate**: 0.001
- **Metrics**: Recall@10/20, NDCG@10/20, MRR
- **Hardware**: 4 H100 GPUs with 32 CPUs and up to 240GB memory

## Pre-trained Models - Important Note

**You do NOT need pre-trained models for baselines.** Unlike RecFormer which uses a pre-trained checkpoint (`pretrained_models/recformer_seqrec_ckpt.bin`), the baseline models (SASRec, FDSA, UniSRec, BERT4Rec) train from scratch on the MIND dataset.

The `--load_model` flag in existing Amazon scripts is for fine-tuning already-trained RecBole models, not for loading pre-trained representations.

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size in config files if you encounter OOM errors
2. **Missing Dependencies**: Ensure all RecBole dependencies are installed
3. **Path Issues**: Check that `downstream_datasets/MIND_mini_json/` exists and contains the required JSON files

### Monitoring Progress

Check job outputs in `jobs/job_output/` directory:
- `baselines_mind_[JobID].out` - Combined baseline job
- `sasrec_mind_[JobID].out` - Individual SASRec job
- `fdsa_mind_[JobID].out` - Individual FDSA job
- `unisrec_mind_[JobID].out` - Individual UniSRec job

### Expected Runtime

- **Individual models**: 2-6 hours each
- **All baselines**: 12-24 hours total
- **Data conversion**: 5-10 minutes

## Comparison with RecFormer

After running all baselines, you can directly compare:

1. **Generalization**: How well do Amazon-trained baselines adapt to news domain?
2. **Performance**: Which model performs best on MIND dataset?
3. **Efficiency**: Training time and resource usage comparison

The comparison script will generate a formatted table showing all metrics across models, making it easy to identify the best-performing approach for news recommendation.

## Next Steps

1. Run the baseline experiments using the provided scripts
2. Analyze the comparison results
3. Consider domain-specific adaptations if baselines underperform
4. Explore the impact of different text processing approaches

For questions or issues, refer to the individual script documentation or the main RecFormer repository. 