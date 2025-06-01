import json
import os
import re # For cleaning text

def clean_text(text_string):
    """Cleans text by removing tabs, newlines, and double quotes, and normalizes spaces."""
    if not isinstance(text_string, str):
        return "" # Return empty string if not a string (e.g. None or NaN)
    text_string = text_string.replace("\t", " ").replace("\n", " ").replace('"', '')
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    text_string = re.sub(r'\s+', ' ', text_string).strip()
    return text_string

def create_recbole_item_file(meta_data_path, output_dir, dataset_name):
    """
    Creates the .item file for RecBole, ensuring consistent cleaning.
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
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write("item_id:token\ttitle:token_seq\tcategories:token_seq\tbrand:token\n")
        processed_items = 0
        for item_id, data in meta_data.items():
            if not isinstance(data, dict):
                print(f"Warning: Invalid data format for item_id {item_id}. Expected dict, got {type(data)}. Skipping.")
                continue

            title = clean_text(data.get("title", ""))
            
            category_raw = data.get("category", "")
            if isinstance(category_raw, list):
                # Clean each category string in the list and join them with a space
                category = " ".join([clean_text(c) for c in category_raw if c]) 
            else:
                # If it's already a string, just clean it
                category = clean_text(category_raw)
            
            brand = clean_text(data.get("brand", ""))
            
            f_out.write(f"{item_id}\t{title}\t{category}\t{brand}\n")
            processed_items +=1
    print(f"Successfully created item file: {output_file_path}")
    print(f"Total items processed for {dataset_name}: {processed_items}")
    return processed_items


def create_recbole_inter_file(interaction_data_path, output_dir, dataset_name, user_id_prefix="user_"):
    """
    Creates the .inter file for RecBole.
    Interaction file format: List of lists, e.g., [["item_id_1", "item_id_2", ...], ["item_id_a", "item_id_b", ...]]
    """
    print(f"Processing interaction data from: {interaction_data_path}")
    try:
        with open(interaction_data_path, 'r', encoding='utf-8') as f:
            interaction_data = json.load(f) # This should be a list of sequences
    except FileNotFoundError:
        print(f"Error: Interaction file not found at {interaction_data_path}")
        return 0, 0
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {interaction_data_path}")
        return 0,0
        
    output_file_path = os.path.join(output_dir, f"{dataset_name}.inter") # Standard .inter name for RecBole internal splitting
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write("user_id:token\titem_id:token\ttimestamp:float\n")

        user_counter = 0
        total_interactions_processed = 0
        for user_sequence in interaction_data:
            if not isinstance(user_sequence, list):
                print(f"Warning: Expected a list for user sequence, but got {type(user_sequence)}. Skipping this entry in {interaction_data_path}.")
                continue
            
            user_id_str = f"{user_id_prefix}{user_counter}"
            timestamp = 0.0
            interactions_in_sequence = 0
            for item_id in user_sequence:
                if not item_id or not isinstance(item_id, str) or item_id.strip() == "":
                    # print(f"Warning: Invalid or empty item_id found for user {user_id_str}. Skipping item.")
                    continue
                f_out.write(f"{user_id_str}\t{item_id.strip()}\t{timestamp}\n")
                timestamp += 1.0
                total_interactions_processed += 1
                interactions_in_sequence += 1
            
            if interactions_in_sequence > 0: # Only increment user counter if they had valid interactions
                user_counter += 1
                
    print(f"Successfully created interaction file: {output_file_path}")
    print(f"Total users processed for {dataset_name}: {user_counter}")
    print(f"Total interactions written for {dataset_name}: {total_interactions_processed}")
    return user_counter, total_interactions_processed

# --- Configuration for Pre-training Data Processing ---
BASE_INPUT_PATH_PRETRAIN = "./pretrain_data" # Or the full path
PRETRAIN_META_FILE = os.path.join(BASE_INPUT_PATH_PRETRAIN, "meta_data.json")
PRETRAIN_TRAIN_INTER_FILE = os.path.join(BASE_INPUT_PATH_PRETRAIN, "train.json") # Main 7 categories
PRETRAIN_DEV_INTER_FILE = os.path.join(BASE_INPUT_PATH_PRETRAIN, "dev.json")   # "CDs and Vinyl"

# Output directory for RecBole formatted pre-training data
RECDATA_PRETRAIN_CORPUS_OUTPUT_DIR = "./recbole_data/pretrain_corpus"  # For main 7 categories training data
RECDATA_PRETRAIN_VALID_OOD_OUTPUT_DIR = "./recbole_data/pretrain_valid_ood" # For "CDs and Vinyl" OOD validation data

# --- Process Pre-training Data ---
print("===== Processing Pre-training Data =====")

# 1. Create the .item file.
#    This meta_data.json is assumed to cover items for ALL pre-training categories (main 7 + CDs and Vinyl).
#    We will place it in the main pretrain_corpus directory.
#    For the OOD validation set, we will link or copy this item file as it shares the same item universe.
if os.path.exists(PRETRAIN_META_FILE):
    create_recbole_item_file(PRETRAIN_META_FILE, RECDATA_PRETRAIN_CORPUS_OUTPUT_DIR, "pretrain_corpus")

    # Ensure the OOD validation output directory exists
    os.makedirs(RECDATA_PRETRAIN_VALID_OOD_OUTPUT_DIR, exist_ok=True)
    
    # Create a symbolic link (or copy if linking fails or is not preferred)
    # to the item file for the OOD validation dataset.
    source_item_file = os.path.join(RECDATA_PRETRAIN_CORPUS_OUTPUT_DIR, "pretrain_corpus.item")
    target_item_file = os.path.join(RECDATA_PRETRAIN_VALID_OOD_OUTPUT_DIR, "pretrain_valid_ood.item") # Match dataset name

    if os.path.exists(source_item_file):
        if os.path.exists(target_item_file):
            os.remove(target_item_file) # Remove if exists to avoid error on link
        try:
            os.link(source_item_file, target_item_file) # Create a hard link
            print(f"Linked item file to {target_item_file}")
        except OSError as e: # Fallback to copy if linking fails (e.g. different filesystems, permissions)
            import shutil
            shutil.copy2(source_item_file, target_item_file)
            print(f"Copied item file to {target_item_file} (linking failed: {e})")
    else:
        print(f"Warning: Source item file {source_item_file} not found for linking/copying.")
else:
    print(f"Warning: Pre-train meta data file {PRETRAIN_META_FILE} not found. Item file not created.")


# 2. Create the .inter file for the main 7 pre-training categories (pretrain_corpus)
#    RecBole will split this internally for its own train/valid/test during pre-training run.
if os.path.exists(PRETRAIN_TRAIN_INTER_FILE):
    create_recbole_inter_file(
        PRETRAIN_TRAIN_INTER_FILE, 
        RECDATA_PRETRAIN_CORPUS_OUTPUT_DIR, 
        "pretrain_corpus", 
        user_id_prefix="user_pretrain_"
    )
else:
    print(f"Warning: Pre-train train interaction file {PRETRAIN_TRAIN_INTER_FILE} not found.")

# 3. Create the .inter file for the "CDs and Vinyl" OOD validation data (pretrain_valid_ood)
#    This dataset will be used *exclusively* as a validation set in a RecBole config.
if os.path.exists(PRETRAIN_DEV_INTER_FILE):
    create_recbole_inter_file(
        PRETRAIN_DEV_INTER_FILE, 
        RECDATA_PRETRAIN_VALID_OOD_OUTPUT_DIR, 
        "pretrain_valid_ood",  # Dataset name for RecBole
        user_id_prefix="user_ood_valid_" # Distinct user prefix
    )
else:
    print(f"Warning: Pre-train dev interaction file {PRETRAIN_DEV_INTER_FILE} not found.")


print("\n===== Pre-training data processing complete. =====")
print(f"Main pre-training data for RecBole is in: {RECDATA_PRETRAIN_CORPUS_OUTPUT_DIR}")
print(f"OOD validation data for RecBole is in: {RECDATA_PRETRAIN_VALID_OOD_OUTPUT_DIR}")