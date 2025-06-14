import os
import json
import pandas as pd
from tqdm import tqdm
import argparse
from collections import defaultdict

def process_mind_to_json(input_path, output_path):
    """
    Processes the MIND dataset into the JSON format expected by finetune.py,
    with user-based temporal splits.
    """
    print("Starting MIND to JSON processing...")
    os.makedirs(output_path, exist_ok=True)

    # --- 1. Read news data to create metadata and item mapping (same as before) ---
    print("Processing news data...")
    train_news_path = os.path.join(input_path, 'train', 'news.tsv')
    dev_news_path = os.path.join(input_path, 'dev', 'news.tsv')
    
    news_df_train = pd.read_csv(train_news_path, sep='\\t', header=None, engine='python', names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
    news_df_dev = pd.read_csv(dev_news_path, sep='\\t', header=None, engine='python', names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
    
    news_df = pd.concat([news_df_train, news_df_dev]).drop_duplicates(subset=['id'])
    
    meta_data = {}
    for _, row in news_df.iterrows():
        meta_data[row['id']] = {'text': str(row['title']) + ' ' + str(row['abstract'])}
    
    with open(os.path.join(output_path, 'meta_data.json'), 'w') as f:
        json.dump(meta_data, f)
    print(f"Saved meta_data.json to {output_path}")

    all_item_ids = news_df['id'].unique()
    item2id = {item_id: i for i, item_id in enumerate(all_item_ids)}
    with open(os.path.join(output_path, 'smap.json'), 'w') as f:
        json.dump(item2id, f)
    print(f"Saved smap.json to {output_path}")

    # --- 2. Load and Combine Behavior Data ---
    print("Processing behavior data...")
    train_behaviors_path = os.path.join(input_path, 'train', 'behaviors.tsv')
    dev_behaviors_path = os.path.join(input_path, 'dev', 'behaviors.tsv')

    behaviors_df_train = pd.read_csv(train_behaviors_path, sep='\\t', header=None, engine='python', names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    behaviors_df_dev = pd.read_csv(dev_behaviors_path, sep='\\t', header=None, engine='python', names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    
    all_behaviors_df = pd.concat([behaviors_df_train, behaviors_df_dev])

    # --- 3. Construct and Split User Histories ---
    print("Constructing and splitting user histories...")
    user_histories = defaultdict(list)
    for _, row in tqdm(all_behaviors_df.iterrows(), total=all_behaviors_df.shape[0]):
        user_id = row['user_id']
        
        history = []
        if pd.notna(row['history']):
            history = str(row['history']).split()
        
        # Get clicked items from the current impression
        impressions = str(row['impressions']).split()
        clicked_items = [imp.split('-')[0] for imp in impressions if imp.endswith('-1')]
        
        # Combine history with current clicks
        full_history = history + clicked_items
        user_histories[user_id].extend(full_history)

    # Remove duplicates while preserving order
    for user_id in user_histories:
        seen = set()
        user_histories[user_id] = [x for x in user_histories[user_id] if not (x in seen or seen.add(x))]

    train_data = []
    val_data = []
    test_data = []

    for user_id, history in tqdm(user_histories.items()):
        if len(history) < 2:  # A user must have at least 2 items to create a sequence
            continue

        if len(history) == 2:
            # If only 2 items, use one for train, one for val. No test item.
            val_item = history[-1]
            train_items = history[:-1]
            val_data.append([user_id, item2id[val_item], 0])
            for item_id in train_items:
                train_data.append([user_id, item2id[item_id], 0])
        else:
            # Original logic for histories of 3 or more
            test_item = history[-1]
            val_item = history[-2]
            train_items = history[:-2]

            test_data.append([user_id, item2id[test_item], 0])
            val_data.append([user_id, item2id[val_item], 0])
            if train_items:
                for item_id in train_items:
                    train_data.append([user_id, item2id[item_id], 0])

    # --- 4. Write Output JSON Files ---
    print("Writing output JSON files...")
    with open(os.path.join(output_path, 'train.json'), 'w') as f:
        json.dump(train_data, f)
    print(f"Saved train.json to {output_path}")

    with open(os.path.join(output_path, 'val.json'), 'w') as f:
        json.dump(val_data, f)
    print(f"Saved val.json to {output_path}")
    
    with open(os.path.join(output_path, 'test.json'), 'w') as f:
        json.dump(test_data, f)
    print(f"Saved test.json to {output_path}")

    print("MIND data processing to JSON format finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="datasets/mind/", help="Path to the raw MIND dataset directory.")
    parser.add_argument("--output_path", default="downstream_datasets/MIND_json/", help="Path to save the processed JSON dataset.")
    args = parser.parse_args()
    
    process_mind_to_json(args.input_path, args.output_path) 