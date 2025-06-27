#!/usr/bin/env python3
"""
Convert MIND dataset from RecBole format to Amazon format for efficient RecFormer training.
Uses the existing RecBole MIND data instead of processing from scratch.
"""

import os
import pandas as pd
from collections import defaultdict
from pathlib import Path


def convert_recbole_to_amazon_format(recbole_dir, output_dir):
    """
    Convert MIND dataset from RecBole format to Amazon user dictionary format.
    
    RecBole format: user_id:token   item_id:token   timestamp:float
    Amazon format: user_id:token   item_id_list:token_seq  item_id:token
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        input_file = os.path.join(recbole_dir, f'MIND.{split}.inter')
        output_file = os.path.join(output_dir, f'MIND.{split}.inter')
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
            
        print(f"Converting {split} split...")
        
        # Read RecBole format with error handling
        try:
            df = pd.read_csv(input_file, sep='\t')
            print(f"  Loaded {len(df)} interactions from {split}")
            
            if len(df) == 0:
                print(f"  No data in {split}, skipping...")
                continue
                
        except Exception as e:
            print(f"  Error reading {input_file}: {e}")
            continue
        
        # Group by user and sort by timestamp
        user_interactions = defaultdict(list)
        for _, row in df.iterrows():
            user_id = row['user_id:token']
            item_id = row['item_id:token']
            timestamp = row.get('timestamp:float', 0.0)
            
            user_interactions[user_id].append((timestamp, item_id))
        
        # Sort interactions by timestamp for each user
        for user_id in user_interactions:
            user_interactions[user_id].sort(key=lambda x: x[0])
            user_interactions[user_id] = [item for _, item in user_interactions[user_id]]
        
        print(f"  Found {len(user_interactions)} unique users")
        
        # Create sequences
        sequences = []
        
        for user_id, items in user_interactions.items():
            if len(items) == 1:
                # For single interactions (common in valid/test), create a minimal sequence
                if split in ['valid', 'test']:
                    # Use the single item as both history and target (for evaluation)
                    sequences.append({
                        'user_id:token': user_id,
                        'item_id_list:token_seq': str(items[0]),
                        'item_id:token': items[0]
                    })
                else:
                    # Skip single interactions in training
                    continue
            else:
                # For multiple interactions, create sequences with history
                for i in range(1, len(items)):
                    history = items[:i]
                    target = items[i]
                    
                    sequences.append({
                        'user_id:token': user_id,
                        'item_id_list:token_seq': ' '.join(map(str, history)),
                        'item_id:token': target
                    })
        
        print(f"  Created {len(sequences)} sequences for {len(user_interactions)} users")
        
        # Write Amazon format
        if sequences:
            amazon_df = pd.DataFrame(sequences)
            amazon_df.to_csv(output_file, sep='\t', index=False)
            print(f"  Created {len(sequences)} sequences for {len(user_interactions)} users")
        else:
            print(f"  No sequences created for {split}")
    
    # Copy item metadata if it exists
    item_file = os.path.join(recbole_dir, 'MIND.item')
    if os.path.exists(item_file):
        output_item_file = os.path.join(output_dir, 'MIND.text')
        
        # Read and convert item metadata to Amazon text format
        item_df = pd.read_csv(item_file, sep='\t')
        
        # Convert to Amazon text format
        text_data = []
        for _, row in item_df.iterrows():
            item_id = row['item_id:token']
            title = row.get('title:token_seq', '')
            categories = row.get('categories:token_seq', '')
            brand = row.get('brand:token', '')
            
            # Combine metadata into text
            text_parts = []
            if title:
                text_parts.append(str(title))
            if categories:
                text_parts.append(str(categories))
            if brand:
                text_parts.append(str(brand))
            
            text = ' '.join(text_parts)
            text_data.append({
                'item_id:token': item_id,
                'text:token_seq': text
            })
        
        if text_data:
            text_df = pd.DataFrame(text_data)
            text_df.to_csv(output_item_file, sep='\t', index=False)
            print(f"Converted item metadata for {len(text_data)} items")
    
    print(f"\nConversion complete! Amazon format data saved to: {output_dir}")


def main():
    # Default paths - using MIND_mini for testing
    recbole_dir = "recbole_data/MIND_mini"
    output_dir = "datasets/downstream/MIND_mini"
    
    print("Converting MIND_mini from RecBole to Amazon format...")
    print(f"Input directory: {recbole_dir}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(recbole_dir):
        print(f"Error: Input directory {recbole_dir} not found!")
        return
    
    convert_recbole_to_amazon_format(recbole_dir, output_dir)


if __name__ == "__main__":
    main() 