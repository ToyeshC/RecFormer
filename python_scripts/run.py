from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.utils import init_seed, get_model, get_trainer
from recbole.data import create_dataset, data_preparation
import datetime
import argparse
import torch
import numpy as np
from collections import Counter

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run RecBole experiments.")
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args()

    config = Config(config_file_list=[args.config_file]) # config is the config_obj
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
        print("Running standard RecBole experiment (Gini coefficient not calculated in this path).")
        run_recbole(config_file_list=[args.config_file], config_dict={'nproc': args.nproc} if args.nproc > 1 else {}, saved=True)

    end_time = datetime.datetime.now()
    print(f"--- Experiment Finished ---")
    print(f"End Time: {end_time}")
    print(f"Total Duration: {end_time - start_time}")