from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.utils import init_seed, get_model, get_trainer, init_logger, set_color
from recbole.data import create_dataset, data_preparation
import datetime
import argparse
import torch
import numpy as np
from collections import Counter
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unisrec import UniSRec
from models.fdsa import FDSA
from data.dataset import UniSRecDataset
from logging import getLogger
from itertools import combinations

"""
Download dataset unisrec repo way (scientific)
Download data for metadata and other things
In the same folder create preprocessing
Take utils and preprocess_amazone in that folder and run them.
Change preprocess_amazone file
"""
# GiniCoefficient class
class GiniCoefficient:
    def gini_coefficient(self, values):
        arr = np.array(values, dtype=float)
        if arr.sum() == 0: return 0.0
        n = arr.size
        if n <= 1 or np.all(arr == arr[0]): return 0.0
        arr = np.sort(arr)
        mu = arr.mean()
        if mu == 0: return 0.0
        index = np.arange(1, n + 1)
        gini_val = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        return gini_val

    def calculate_list_gini(self, articles, key="category"):
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        if not freqs: return 0.0
        return self.gini_coefficient(list(freqs.values()))

# --- Gini Calculation Function ---
def calculate_gini_for_categories_from_recs(
    recommended_item_ids_tensor, k_for_gini, 
    base_dataset_object, 
    config_obj, # Added config_obj
    existing_results_dict
):
    category_field_name_in_dataset = 'categories'
    if not hasattr(base_dataset_object, 'item_feat') or base_dataset_object.item_feat is None:
        print("Warning (Gini): Item features (item_feat) not loaded in base_dataset_object. Cannot calculate Gini.")
        existing_results_dict[f'gini_{category_field_name_in_dataset}@{k_for_gini}'] = "Error: item_feat missing"
        return existing_results_dict
    if category_field_name_in_dataset not in base_dataset_object.item_feat:
        print(f"Warning (Gini): Category field '{category_field_name_in_dataset}' not found in item_feat of base_dataset_object. Cannot calculate Gini.")
        existing_results_dict[f'gini_{category_field_name_in_dataset}@{k_for_gini}'] = f"Error: {category_field_name_in_dataset} missing"
        return existing_results_dict

    all_item_categories_tensor = base_dataset_object.item_feat[category_field_name_in_dataset]
    
    padding_id_for_categories = 0 
    pad_token_str = config_obj['PAD_TOKEN'] 

    if category_field_name_in_dataset in base_dataset_object.field2token_id and \
       pad_token_str in base_dataset_object.field2token_id[category_field_name_in_dataset]:
        padding_id_for_categories = base_dataset_object.field2token_id[category_field_name_in_dataset][pad_token_str]
    else:
        print(f"Warning (Gini): Padding token '{pad_token_str}' not found in vocab for field '{category_field_name_in_dataset}'. Defaulting padding ID to 0. This might be incorrect if 0 is a valid category token ID.")
        
    all_category_tokens_in_recommendations = []
    for user_recs_item_ids_1_indexed in recommended_item_ids_tensor:
        for internal_item_id_1_indexed in user_recs_item_ids_1_indexed.tolist():
            if internal_item_id_1_indexed == 0: continue
            internal_item_id_0_indexed = internal_item_id_1_indexed - 1
            if not (0 <= internal_item_id_0_indexed < all_item_categories_tensor.shape[0]): continue
            item_specific_category_token_ids = all_item_categories_tensor[internal_item_id_0_indexed].tolist()
            for cat_token_id in item_specific_category_token_ids:
                if cat_token_id != padding_id_for_categories:
                    all_category_tokens_in_recommendations.append(cat_token_id)

    metric_name = f'gini_{category_field_name_in_dataset}@{k_for_gini}'
    if not all_category_tokens_in_recommendations:
        print(f"Warning (Gini): No valid category tokens for Gini. Setting {metric_name} to 0.0.")
        existing_results_dict[metric_name] = 0.0
        return existing_results_dict

    category_token_counts = Counter(all_category_tokens_in_recommendations)
    gini_calculator = GiniCoefficient()
    gini_val = gini_calculator.gini_coefficient(list(category_token_counts.values()))
    existing_results_dict[metric_name] = gini_val
    print(f"Gini Coefficient ({category_field_name_in_dataset}@{k_for_gini}): {gini_val:.4f}")
    return existing_results_dict

def calculate_coverage(recommended_item_ids_tensor, total_num_items, existing_results_dict, k):
    recommended_items = recommended_item_ids_tensor.flatten().tolist()
    recommended_set = set(item_id for item_id in recommended_items if item_id > 0)
    coverage = len(recommended_set) / total_num_items
    metric_name = f'coverage@{k}'
    existing_results_dict[metric_name] = coverage
    print(f"Coverage@{k}: {coverage:.4f}")
    return existing_results_dict

def jaccard_distance(set1, set2):
    if not set1 and not set2: return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - intersection / union if union != 0 else 0.0

def calculate_intra_list_diversity(recommended_item_ids_tensor, item_feature_dict, existing_results_dict, k):
    ild_scores = []
    for rec_list in recommended_item_ids_tensor:
        rec_items = rec_list.tolist()
        rec_items = [item for item in rec_items if item > 0]
        if len(rec_items) < 2:
            ild_scores.append(0.0)
            continue


        distances = []
        for i, j in combinations(rec_items, 2):
            feat_i = item_feature_dict.get(i - 1, set())
            feat_j = item_feature_dict.get(j - 1, set())
            dist = jaccard_distance(feat_i, feat_j)
            distances.append(dist)
        ild_scores.append(sum(distances) / len(distances) if distances else 0.0)
   
    ild_value = sum(ild_scores) / len(ild_scores)
    metric_name = f'ild@{k}'
    existing_results_dict[metric_name] = ild_value
    print(f"Intra-List Diversity@{k}: {ild_value:.4f}")
    return existing_results_dict

def evaluate_diversity(config, trainer, test_data, dataset_obj=None):
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    model = trainer.model
    model.eval()

    print("Calculating Gini coefficient for categories (main process using separate prediction)...")

    k = config['topk'][0] if isinstance(config['topk'], list) else config['topk']
    if dataset_obj is None:
        dataset_obj = create_dataset(config)

    all_topk_item_indices_list = []
    with torch.no_grad():
        for batch_idx, interaction_batch in enumerate(test_data):
            interaction = interaction_batch[0] if isinstance(interaction_batch, tuple) else interaction_batch
            interaction = interaction.to(model.device)

            if hasattr(model, 'full_sort_predict'):
                scores = model.full_sort_predict(interaction)
            elif hasattr(model, 'predict'):
                scores = model.predict(interaction)
                print(f"Warning: Using `predict()` for batch {batch_idx}, which may not be full sort.")
            else:
                print(f"Error: Model lacks `predict` and `full_sort_predict`. Skipping batch {batch_idx}.")
                all_topk_item_indices_list.append(torch.empty(0, k, dtype=torch.long).cpu())
                continue

            if scores.dim() == 1:
                if interaction[config['USER_ID_FIELD']].shape[0] == 1:
                    scores = scores.unsqueeze(0)
                else:
                    print(f"Error: 1D scores but batch size > 1. Skipping batch {batch_idx}.")
                    all_topk_item_indices_list.append(torch.empty(0, k, dtype=torch.long).cpu())
                    continue

            if scores.dim() != 2:
                print(f"Error: Unexpected score dimension {scores.shape}. Skipping.")
                all_topk_item_indices_list.append(torch.empty(0, k, dtype=torch.long).cpu())
                continue

            current_k = min(scores.shape[1], k)
            if current_k == 0:
                all_topk_item_indices_list.append(torch.empty(scores.shape[0], 0, dtype=torch.long).cpu())
                continue

            _, topk = torch.topk(scores, k=current_k, dim=1)
            if current_k < k:
                pad = torch.zeros((topk.shape[0], k - current_k), dtype=torch.long)
                topk = torch.cat((topk.cpu(), pad), dim=1)
            else:
                topk = topk.cpu()

            all_topk_item_indices_list.append(topk + 1)

    # Gini / ILD / Coverage
    if all_topk_item_indices_list:
        valid_tensors = [t for t in all_topk_item_indices_list if t.shape[0] > 0 and t.shape[1] > 0]
        if valid_tensors:
            final_recs = torch.cat(valid_tensors, dim=0)

            test_result = calculate_gini_for_categories_from_recs(
                final_recs, k, dataset_obj, config, test_result
            )

            total_items = dataset_obj.item_num
            pad_token = config['PAD_TOKEN']
            cat_field = 'categories'

            pad_id = dataset_obj.field2token_id.get(cat_field, {}).get(pad_token, 0)

            item_feat_dict = {}
            for item_id, cat_ids in enumerate(dataset_obj.item_feat[cat_field]):
                item_feat_dict[item_id] = set(c for c in cat_ids.tolist() if c != pad_id)

            test_result = calculate_coverage(final_recs, total_items, test_result, k)
            test_result = calculate_intra_list_diversity(final_recs, item_feat_dict, test_result, k)

    return test_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run RecBole experiments.")
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args() 

    # if args.model_name != 'UniSRec':
    #     config = Config(config_file_list=[args.config_file]) # config is the config_obj
    if args.nproc > 1: config['nproc'] = args.nproc

    print(f"Start Time: {datetime.datetime.now()}")
    start_time = datetime.datetime.now()

    if args.load_model and 'pretrained_model_path' in config and config['pretrained_model_path']:
        init_seed(config['seed'], config['reproducibility'])
        dataset_obj = create_dataset(config) 
        train_data, valid_data, test_data = data_preparation(config, dataset_obj)
        
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        
        checkpoint = torch.load(config['pretrained_model_path'], map_location=config['device'])
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        keys_to_delete_from_pretrained = []
        for key in list(pretrained_state_dict.keys()):
            if key in model_state_dict:
                if pretrained_state_dict[key].shape != model_state_dict[key].shape:
                    print(f"Size mismatch for {key}: pretrained {pretrained_state_dict[key].shape} vs current {model_state_dict[key].shape}")
                    if 'embedding' in key:
                        min_dim0_size = min(pretrained_state_dict[key].shape[0], model_state_dict[key].shape[0])
                        current_slice = [slice(None)] * model_state_dict[key].dim()
                        pretrained_slice = [slice(None)] * pretrained_state_dict[key].dim()
                        current_slice[0] = slice(0, min_dim0_size)
                        pretrained_slice[0] = slice(0, min_dim0_size)
                        for i in range(1, model_state_dict[key].dim()):
                            if pretrained_state_dict[key].shape[i] != model_state_dict[key].shape[i]:
                                min_dim_i_size = min(pretrained_state_dict[key].shape[i], model_state_dict[key].shape[i])
                                current_slice[i] = slice(0, min_dim_i_size); pretrained_slice[i] = slice(0, min_dim_i_size)
                        model_state_dict[key][tuple(current_slice)] = pretrained_state_dict[key][tuple(pretrained_slice)]
                        print(f"Copied overlapping part for {key}")
                    keys_to_delete_from_pretrained.append(key)
            else: keys_to_delete_from_pretrained.append(key)
        for key_to_del in keys_to_delete_from_pretrained:
            if key_to_del in pretrained_state_dict: del pretrained_state_dict[key_to_del]
        model.load_state_dict(pretrained_state_dict, strict=False)
        print(f"Loaded pretrained model into main model instance from {config['pretrained_model_path']} (strict=False)")

        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        
        print("Starting fine-tuning...")
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=config['show_progress']
        )
        print(f"Fine-tuning complete. Best validation result: {best_valid_result}")

        is_main_process = True
        if hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
            is_main_process = trainer.accelerator.is_main_process
        
        test_result_std_metrics = {}
        if is_main_process:
            print("Evaluating on test set for standard metrics (main process)...")
            test_result_std_metrics = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
            
            final_model_for_gini = trainer.model
            final_model_for_gini.eval()

            print("Calculating Gini coefficient for categories (main process using separate prediction)...")
            
            k_for_gini_metric = config['topk'][0] if isinstance(config['topk'], list) else config['topk']

            all_topk_item_indices_list = []
            with torch.no_grad():
                for batch_idx, batched_interaction_tuple_or_obj in enumerate(test_data):
                    actual_interaction = batched_interaction_tuple_or_obj[0] if isinstance(batched_interaction_tuple_or_obj, tuple) else batched_interaction_tuple_or_obj
                    interaction_on_device = actual_interaction.to(final_model_for_gini.device)
                    
                    if hasattr(final_model_for_gini, 'full_sort_predict'):
                        scores = final_model_for_gini.full_sort_predict(interaction_on_device)
                    elif hasattr(final_model_for_gini, 'predict'):
                        scores = final_model_for_gini.predict(interaction_on_device)
                        print(f"Warning (Gini): Model lacks 'full_sort_predict'. Using 'predict()'. Output shape: {scores.shape}. This may not be suitable if scores are not for all items or 1D for multi-user batches.")
                    else:
                        print(f"Error (Gini): Model has neither full_sort_predict nor predict method. Skipping Gini for batch {batch_idx}.")
                        all_topk_item_indices_list.append(torch.empty(0, k_for_gini_metric, dtype=torch.long).cpu())
                        continue
                    
                    if scores.dim() == 1:
                        if interaction_on_device[config['USER_ID_FIELD']].shape[0] == 1:
                            scores = scores.unsqueeze(0)
                        else:
                            print(f"Error (Gini): Scores tensor is 1D ({scores.shape}) but effective batch size is > 1. Cannot reliably perform topk. Skipping batch {batch_idx} for Gini.")
                            all_topk_item_indices_list.append(torch.empty(0, k_for_gini_metric, dtype=torch.long).cpu())
                            continue
                    
                    if scores.dim() != 2:
                        print(f"Error (Gini): Scores tensor has unexpected dimension {scores.dim()} ({scores.shape}) after potential reshape. Expected 2D. Skipping batch {batch_idx} for Gini.")
                        all_topk_item_indices_list.append(torch.empty(0, k_for_gini_metric, dtype=torch.long).cpu())
                        continue
                    
                    if scores.shape[1] < k_for_gini_metric:
                        print(f"Warning (Gini): Not enough items ({scores.shape[1]}) to take top {k_for_gini_metric}. Taking all available items for batch {batch_idx}.")
                        current_k = scores.shape[1]
                    else:
                        current_k = k_for_gini_metric

                    if current_k == 0 :
                         all_topk_item_indices_list.append(torch.empty(scores.shape[0], 0, dtype=torch.long).cpu())
                         continue

                    _, topk_indices_batch = torch.topk(scores, k=current_k, dim=1)
                    if current_k < k_for_gini_metric:
                         padding = torch.zeros((topk_indices_batch.shape[0], k_for_gini_metric - current_k), dtype=torch.long)
                         topk_indices_batch = torch.cat((topk_indices_batch.cpu(), padding), dim=1)

                    all_topk_item_indices_list.append(topk_indices_batch.cpu() + 1)

            if all_topk_item_indices_list:
                valid_tensors = [t for t in all_topk_item_indices_list if t.shape[0] > 0 and t.shape[1] > 0]
                if valid_tensors:
                    final_recommended_item_ids_tensor = torch.cat(valid_tensors, dim=0)
                    test_result_std_metrics = calculate_gini_for_categories_from_recs(
                        final_recommended_item_ids_tensor,
                        k_for_gini_metric,
                        dataset_obj, 
                        config, # Pass the config object
                        test_result_std_metrics
                    )
                else:
                    print("Warning (Gini): No valid recommendations generated across all batches for Gini calculation.")
                    test_result_std_metrics[f'gini_categories@{k_for_gini_metric}'] = "Error: No valid recs for Gini"
            else:
                print("Warning (Gini): No recommendations generated from manual pass for Gini calculation (all_topk_item_indices_list is empty).")
                test_result_std_metrics[f'gini_categories@{k_for_gini_metric}'] = "Error: No recs for Gini (manual pass, list empty)"
            
            print(f"Final Test Result (including Gini): {test_result_std_metrics}")
        else:
            print("Skipping test evaluation and Gini calculation (not on main DDP process).")

    else:
        if args.model_name != 'UniSRec' and args.model_name != 'FDSA':
            print("Running standard RecBole experiment (Gini coefficient not calculated in this path).")
            print(f"Using config file: {args.config_file}")
            props = [args.config_file, 'configs/finetune.yaml']            
            # recbole is black box so to get the model and eval with own metrics I just use trainer.fit
            # run_recbole(model=args.model_name, config_file_list=props, config_dict={'nproc': args.nproc} if args.nproc > 1 else {}, saved=True)
        
            config = Config(model=args.model_name, config_file_list=props)
            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)
            logger = getLogger()
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
            model_class = get_model(args.model_name)
            model = model_class(config, train_data.dataset).to(config['device'])
            trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

            # Train the model
            best_valid_score, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=config['show_progress']
            )

            # Run the shared evaluation
            test_result = evaluate_diversity(config, trainer, test_data)
        
        elif args.model_name == 'UniSRec' or args.model_name == 'FDSA':
            print("Running UniSRec experiment.")
            print(f"Using config file: {args.config_file}")
            props = [args.config_file, 'configs/finetune.yaml']
            print(props)

            if args.model_name == 'UniSRec':
                config = Config(model=UniSRec, config_file_list=props)
            else:
                config = Config(model=FDSA, config_file_list=props)

            init_seed(config['seed'], config['reproducibility'])
            # logger initialization
            init_logger(config)
            logger = getLogger()
            logger.info(config)
            # dataset filtering
            dataset = UniSRecDataset(config)
            logger.info(dataset)
            # dataset splitting
            train_data, valid_data, test_data = data_preparation(config, dataset)
            # model loading and initialization
            if args.model_name == 'UniSRec':
                model = UniSRec(config, train_data.dataset).to(config['device'])
            elif args.model_name == 'FDSA':
                model = FDSA(config, train_data.dataset).to(config['device'])
            # Load pre-trained model
            if args.load_model and 'pretrained_model_path' in config and config['pretrained_model_path']:
                checkpoint = torch.load(config['pretrained_model_path'])
                logger.info(f'Loading from {config["pretrained_model_path"]}')
                logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                if fix_enc:
                    logger.info(f'Fix encoder parameters.')
                    for _ in model.position_embedding.parameters():
                        _.requires_grad = False
                    for _ in model.trm_encoder.parameters():
                        _.requires_grad = False
            logger.info(model)
            # trainer loading and initialization
            trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

            # model training
            best_valid_score, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=config['show_progress']
            )

            test_result = evaluate_diversity(config, trainer, test_data)

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

    end_time = datetime.datetime.now()
    print(f"--- Experiment Finished ---")
    print(f"End Time: {end_time}")
    print(f"Total Duration: {end_time - start_time}")
