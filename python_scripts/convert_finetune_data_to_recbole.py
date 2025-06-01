import json
import os

def create_recbole_item_file(meta_data_path, output_dir, dataset_name):
    """
    Creates the .item file for RecBole using the original simpler cleaning.
    meta_data.json format: {"item_id_1": {"title": "t1", "brand": "b1", "category": "c1"}, ...}
    """
    print(f"Processing item metadata from: {meta_data_path}")
    try:
        with open(meta_data_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Meta data file not found at {meta_data_path}")
        return 0
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {meta_data_path}")
        return 0

    output_file_path = os.path.join(output_dir, f"{dataset_name}.item")

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write("item_id:token\ttitle:token_seq\tcategories:token_seq\tbrand:token\n")
        processed_items = 0
        for item_id, data in meta_data.items():
            if not isinstance(data, dict):
                print(f"Warning: Invalid data format for item_id {item_id}. Expected dict, got {type(data)}. Skipping.")
                continue

            title = data.get("title", "")
            category_raw = data.get("category", "")
            brand = data.get("brand", "")

            title = title.replace("\t", " ").replace("\n", " ").replace('"', '')

            if isinstance(category_raw, list):
                category_str = " ".join([str(c) for c in category_raw if c])
                category = category_str.replace("\t", " ").replace("\n", " ").replace('"', '')
            elif isinstance(category_raw, str):
                category = category_raw.replace("\t", " ").replace("\n", " ").replace('"', '')
            else:
                # Ensure category is an empty string if category_raw is not list or str (e.g. None, number)
                category = "" # Initialize as empty string
                category = category.replace("\t", " ").replace("\n", " ").replace('"', '') # Clean the empty string (no-op but consistent)


            brand = brand.replace("\t", " ").replace("\n", " ").replace('"', '')

            f_out.write(f"{item_id}\t{title}\t{category}\t{brand}\n")
            processed_items += 1
    print(f"Successfully created item file: {output_file_path}")
    print(f"Total items processed for {dataset_name}: {processed_items}")
    return processed_items

def create_recbole_inter_file_for_split(interaction_data_path, output_dir, dataset_name, split_name, user_id_prefix="user_"):
    """
    Creates the .inter file for RecBole for a specific split (train, valid, test).
    Interaction file format: JSON object where keys are user IDs and values are lists of item IDs.
    e.g., {"user_str_id1": [item_id1, item_id2, ...], "user_str_id2": [item_id_a, item_id_b, ...]}
    """
    print(f"Processing {split_name} interaction data from: {interaction_data_path}")
    interaction_data_dict = None
    try:
        with open(interaction_data_path, 'r', encoding='utf-8') as f:
            interaction_data_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: Interaction file not found at {interaction_data_path}")
        return 0, 0
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {interaction_data_path}. Details: {e}")
        return 0,0

    if not isinstance(interaction_data_dict, dict):
        print(f"Error: Expected a dictionary from {interaction_data_path}, but got {type(interaction_data_dict)}. Cannot process.")
        return 0,0

    # MODIFIED: Use underscore between dataset_name and split_name for the filename
    output_file_path = os.path.join(output_dir, f"{dataset_name}_{split_name}.inter")
    # OLD: output_file_path = os.path.join(output_dir, f"{dataset_name}.{split_name}.inter")


    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write("user_id:token\titem_id:token\ttimestamp:float\n")

        recbole_user_id_counter = 0
        total_interactions_processed = 0

        for original_user_id_key, user_sequence in interaction_data_dict.items():
            if not isinstance(user_sequence, list):
                print(f"Warning: For user key '{original_user_id_key}', expected a list for sequence, but got {type(user_sequence)}. Skipping this user.")
                continue

            recbole_user_id_str = f"{user_id_prefix}{recbole_user_id_counter}"
            timestamp = 0.0
            interactions_in_sequence = 0
            for item_id in user_sequence:
                if not item_id and item_id != 0: 
                    continue

                item_id_str = str(item_id).strip() 
                if not item_id_str and item_id_str != "0": 
                    continue

                f_out.write(f"{recbole_user_id_str}\t{item_id_str}\t{timestamp}\n")
                timestamp += 1.0
                total_interactions_processed += 1
                interactions_in_sequence += 1

            if interactions_in_sequence > 0:
                recbole_user_id_counter += 1 

    print(f"Successfully created {split_name} interaction file: {output_file_path}")
    print(f"Total RecBole users (sequences with interactions) for {split_name} in {dataset_name}: {recbole_user_id_counter}")
    print(f"Total interactions written for {split_name} in {dataset_name}: {total_interactions_processed}")
    return recbole_user_id_counter, total_interactions_processed

def combine_split_files_for_recbole(output_dir, dataset_name):
    """
    Combines the train, valid, test .inter files (now named with _) into a single .inter file for RecBole,
    removing duplicate headers and ensuring proper format.
    """
    print(f"Combining split files for {dataset_name}...")

    # MODIFIED: Use underscore in filenames
    train_file = os.path.join(output_dir, f"{dataset_name}_train.inter")
    valid_file = os.path.join(output_dir, f"{dataset_name}_valid.inter")
    test_file = os.path.join(output_dir, f"{dataset_name}_test.inter")
    # OLD:
    # train_file = os.path.join(output_dir, f"{dataset_name}.train.inter")
    # valid_file = os.path.join(output_dir, f"{dataset_name}.valid.inter")
    # test_file = os.path.join(output_dir, f"{dataset_name}.test.inter")
    
    combined_file = os.path.join(output_dir, f"{dataset_name}.inter") # Combined file still uses a dot before .inter

    split_files = [train_file, valid_file, test_file]
    missing_files = [f for f in split_files if not os.path.exists(f)]

    if missing_files:
        print(f"Warning: Missing split files for {dataset_name}: {missing_files}. Cannot combine.")
        return False

    try:
        with open(combined_file, 'w', encoding='utf-8') as f_out:
            with open(train_file, 'r', encoding='utf-8') as f_train:
                header = f_train.readline()
                f_out.write(header)
                for line in f_train:
                    f_out.write(line)

            with open(valid_file, 'r', encoding='utf-8') as f_valid:
                next(f_valid) 
                for line in f_valid:
                    f_out.write(line)

            with open(test_file, 'r', encoding='utf-8') as f_test:
                next(f_test)  
                for line in f_test:
                    f_out.write(line)

        print(f"Successfully combined split files into: {combined_file}")
        with open(combined_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Combined file has {len(lines)} total lines (including header)")
            print(f"Header: {lines[0].strip()}")
            if len(lines) > 1: print(f"First data line: {lines[1].strip()}")
            if len(lines) > 2: print(f"Last data line: {lines[-1].strip()}")
        return True
    except Exception as e:
        print(f"Error combining files for {dataset_name}: {e}")
        return False

# --- Configuration for Fine-tuning Data Processing ---
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

BASE_FINETUNE_INPUT_DIR = os.path.join(repo_root, "raw_data")
RECDATA_BASE_OUTPUT_DIR = os.path.join(repo_root, "recbole_data")

print(f"Script directory: {script_dir}")
print(f"Repository root: {repo_root}")
print(f"Input directory: {BASE_FINETUNE_INPUT_DIR}")
print(f"Output directory: {RECDATA_BASE_OUTPUT_DIR}")

try:
    target_domains_original_names = [d for d in os.listdir(BASE_FINETUNE_INPUT_DIR) if os.path.isdir(os.path.join(BASE_FINETUNE_INPUT_DIR, d))]
except FileNotFoundError:
    print(f"Error: Base fine-tune input directory not found at {BASE_FINETUNE_INPUT_DIR}")
    target_domains_original_names = []

if not target_domains_original_names:
    print(f"No domain subdirectories found in {BASE_FINETUNE_INPUT_DIR}. Please check the path and structure.")
else:
    print(f"Found target domains: {target_domains_original_names}")

for domain_name_original in target_domains_original_names:
    domain_name_clean = domain_name_original.lower().replace(" ", "_").replace("-", "_")

    print(f"\n===== Processing domain: {domain_name_original} (as {domain_name_clean}) =====")
    domain_input_path = os.path.join(BASE_FINETUNE_INPUT_DIR, domain_name_original)

    recbole_dataset_name = f"finetune_{domain_name_clean}"
    domain_recbole_output_path = os.path.join(RECDATA_BASE_OUTPUT_DIR, recbole_dataset_name)
    os.makedirs(domain_recbole_output_path, exist_ok=True)
    print(f"Output directory for {recbole_dataset_name}: {domain_recbole_output_path}")

    meta_file_path = os.path.join(domain_input_path, "meta_data.json")
    create_recbole_item_file(meta_file_path, domain_recbole_output_path, recbole_dataset_name)

    domain_user_prefix = f"{domain_name_clean}_user_"
    splits_to_process = {
        "train": "train.json",
        "valid": "val.json",
        "test": "test.json"
    }

    for split_key, json_filename in splits_to_process.items():
        interaction_file_path = os.path.join(domain_input_path, json_filename)
        create_recbole_inter_file_for_split(
            interaction_file_path,
            domain_recbole_output_path,
            recbole_dataset_name,
            split_key,
            user_id_prefix=domain_user_prefix
        )
    
    combine_split_files_for_recbole(domain_recbole_output_path, recbole_dataset_name)

print("\n===== All specified fine-tuning datasets processed. =====")