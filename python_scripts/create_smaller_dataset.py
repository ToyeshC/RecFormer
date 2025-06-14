import pandas as pd
import numpy as np
import os
import shutil

def create_smaller_dataset(inter_file, item_file, feat_file, output_dir, sample_ratio=0.1):
    """
    Create smaller versions of interaction and item datasets.
    
    Args:
        inter_file (str): Path to interaction file
        item_file (str): Path to item file
        feat_file (str): Path to feature file
        output_dir (str): Directory to save smaller datasets
        sample_ratio (float): Ratio of data to sample (default: 0.1 for 10%)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the files with the correct column names
    print("Reading interaction file...")
    inter_df = pd.read_csv(inter_file, sep='\t')
    # Clean column names by removing the data type suffix
    inter_df.columns = [col.split(':')[0] for col in inter_df.columns]
    
    print("Reading item file...")
    item_df = pd.read_csv(item_file, sep='\t')
    # Clean column names by removing the data type suffix
    item_df.columns = [col.split(':')[0] for col in item_df.columns]
    
    # Sample users from interaction file
    unique_users = inter_df['user_id'].unique()
    sampled_users = np.random.choice(unique_users, size=int(len(unique_users) * sample_ratio), replace=False)
    
    # Filter interactions for sampled users
    sampled_inter_df = inter_df[inter_df['user_id'].isin(sampled_users)]
    
    # Get unique items from sampled interactions
    sampled_items = sampled_inter_df['item_id'].unique()
    
    # Filter item features for sampled items
    sampled_item_df = item_df[item_df['item_id'].isin(sampled_items)]
    
    # Save smaller datasets
    inter_output = os.path.join(output_dir, 'small_dataset.inter')
    item_output = os.path.join(output_dir, 'small_dataset.item')
    feat_output = os.path.join(output_dir, 'small_dataset.feat1CLS')
    
    print(f"Saving smaller interaction file to {inter_output}")
    # Add back the data type suffixes when saving
    inter_columns = {
        'user_id': 'user_id:token',
        'item_id': 'item_id:token',
        'timestamp': 'timestamp:float'
    }
    sampled_inter_df.columns = [inter_columns[col] for col in sampled_inter_df.columns]
    sampled_inter_df.to_csv(inter_output, sep='\t', index=False)
    
    print(f"Saving smaller item file to {item_output}")
    # Add back the data type suffixes when saving
    item_columns = {
        'item_id': 'item_id:token',
        'title': 'title:token_seq',
        'categories': 'categories:token_seq',
        'brand': 'brand:token'
    }
    sampled_item_df.columns = [item_columns[col] for col in sampled_item_df.columns]
    sampled_item_df.to_csv(item_output, sep='\t', index=False)
    
    # Copy the feature file
    print(f"Copying feature file to {feat_output}")
    shutil.copy2(feat_file, feat_output)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Original interactions: {len(inter_df)}")
    print(f"Sampled interactions: {len(sampled_inter_df)}")
    print(f"Original items: {len(item_df)}")
    print(f"Sampled items: {len(sampled_item_df)}")
    print(f"Original users: {len(unique_users)}")
    print(f"Sampled users: {len(sampled_users)}")

if __name__ == "__main__":
    # Define paths
    data_dir = "recbole_data/pretrain_valid_ood"
    output_dir = "recbole_data/small_dataset"
    
    inter_file = os.path.join(data_dir, "pretrain_valid_ood.inter")
    item_file = os.path.join(data_dir, "pretrain_valid_ood.item")
    feat_file = os.path.join(data_dir, "pretrain_valid_ood.feat1CLS")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create smaller dataset
    create_smaller_dataset(inter_file, item_file, feat_file, output_dir) 