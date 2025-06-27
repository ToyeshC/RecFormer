#!/usr/bin/env python3
"""
Convert MIND dataset from JSON format to RecBole format for baseline model training.
This script specifically handles the MIND dataset structure.
"""

import json
import os
import argparse
from pathlib import Path

def convert_mind_interactions_to_recbole(input_file, output_file, split_name):
    """
    Convert MIND JSON interaction file to RecBole .inter format
    
    Input format: [[user_id, item_id, interaction_type], ...]
    Output format: user_id:token\titem_id:token\ttimestamp:float
    """
    print(f"Converting {split_name} interactions from {input_file} to {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        interactions = json.load(f)
    
    # Group interactions by user
    user_interactions = {}
    for interaction in interactions:
        user_id, item_id, interaction_type = interaction
        if user_id not in user_interactions:
            user_interactions[user_id] = []
        # Include all interactions (MIND uses 0 for positive, 1 for negative)
        # For sequential recommendation, we treat all clicked items as positive
        user_interactions[user_id].append(item_id)
    
    # Write to RecBole format
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        
        total_interactions = 0
        for user_id, items in user_interactions.items():
            # Sort items to maintain order (assuming they're already in chronological order)
            for timestamp, item_id in enumerate(items):
                f.write(f"{user_id}\t{item_id}\t{float(timestamp)}\n")
                total_interactions += 1
    
    print(f"Converted {len(user_interactions)} users with {total_interactions} interactions for {split_name}")
    return len(user_interactions), total_interactions

def convert_mind_items_to_recbole(meta_file, smap_file, output_file):
    """
    Convert MIND metadata to RecBole .item format
    
    Input: meta_data.json and smap.json
    Output: item_id:token\ttitle:token_seq\tcategories:token_seq\tbrand:token
    """
    print(f"Converting item metadata from {meta_file} to {output_file}")
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    with open(smap_file, 'r', encoding='utf-8') as f:
        smap = json.load(f)
    
    # Create reverse mapping from item_id to internal_id
    id_to_item = {v: k for k, v in smap.items()}
    
    import csv
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        # Write header
        writer.writerow(['item_id:token', 'title:token_seq', 'categories:token_seq', 'brand:token'])
        
        total_items = 0
        for internal_id in range(len(smap)):
            if internal_id in id_to_item:
                item_key = id_to_item[internal_id]
                if item_key in metadata:
                    item_data = metadata[item_key]
                    # Clean and truncate the title text to avoid CSV parsing issues
                    title = item_data.get('text', '')
                    title = title.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                    # Truncate very long titles to 200 characters to avoid parsing issues
                    title = title[:200] if len(title) > 200 else title
                    title = title.strip()
                    # For news articles, we don't have categories/brand, so use empty strings
                    categories = ""
                    brand = ""
                    writer.writerow([internal_id, title, categories, brand])
                    total_items += 1
    
    print(f"Converted {total_items} items")
    return total_items

def combine_splits_to_single_file(output_dir, dataset_name):
    """Combine train, valid, test splits into a single .inter file"""
    splits = ['train', 'valid', 'test']
    combined_file = os.path.join(output_dir, f"{dataset_name}.inter")
    
    with open(combined_file, 'w', encoding='utf-8') as f_out:
        header_written = False
        
        for split in splits:
            split_file = os.path.join(output_dir, f"{dataset_name}.{split}.inter")
            if os.path.exists(split_file):
                with open(split_file, 'r', encoding='utf-8') as f_in:
                    lines = f_in.readlines()
                    if not header_written:
                        f_out.write(lines[0])  # Write header
                        header_written = True
                    # Write data lines (skip header)
                    for line in lines[1:]:
                        f_out.write(line)
                print(f"Added {len(lines)-1} interactions from {split}")
            else:
                print(f"Warning: {split_file} not found")
    
    print(f"Combined file created: {combined_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert MIND dataset to RecBole format")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Input directory containing MIND JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for RecBole format files')
    parser.add_argument('--dataset_name', type=str, default='MIND',
                        help='Dataset name for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths
    input_files = {
        'train': os.path.join(args.input_dir, 'train.json'),
        'val': os.path.join(args.input_dir, 'val.json'), 
        'test': os.path.join(args.input_dir, 'test.json'),
        'meta': os.path.join(args.input_dir, 'meta_data.json'),
        'smap': os.path.join(args.input_dir, 'smap.json')
    }
    
    # Check if all required files exist
    for file_type, file_path in input_files.items():
        if not os.path.exists(file_path):
            print(f"Error: Required file {file_path} not found")
            return
    
    print(f"Converting MIND dataset from {args.input_dir} to {args.output_dir}")
    
    # Convert interaction files
    total_users = 0
    total_interactions = 0
    
    for split in ['train', 'val', 'test']:
        split_name = 'valid' if split == 'val' else split
        output_file = os.path.join(args.output_dir, f"{args.dataset_name}.{split_name}.inter")
        users, interactions = convert_mind_interactions_to_recbole(
            input_files[split], output_file, split_name)
        total_users += users
        total_interactions += interactions
    
    # Convert item metadata
    item_file = os.path.join(args.output_dir, f"{args.dataset_name}.item")
    total_items = convert_mind_items_to_recbole(
        input_files['meta'], input_files['smap'], item_file)
    
    # Combine splits into single file
    combine_splits_to_single_file(args.output_dir, args.dataset_name)
    
    print(f"\nConversion complete!")
    print(f"Total users: {total_users}")
    print(f"Total interactions: {total_interactions}")
    print(f"Total items: {total_items}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 