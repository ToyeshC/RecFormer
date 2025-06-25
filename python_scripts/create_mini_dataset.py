import os
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def create_mini_dataset(root_path, output_path, train_samples=1000, dev_samples=200):
    """
    Creates a small, consistent subset of the MIND dataset.

    Args:
        root_path (str): The path to the full MIND dataset (e.g., 'data/mind').
        output_path (str): The path to save the mini dataset (e.g., 'data/mind_mini').
        train_samples (int): The number of user behaviors to take for the training set.
        dev_samples (int): The number of user behaviors to take for the dev set.
    """
    root_path = Path(root_path)
    output_path = Path(output_path)

    print(f"Creating mini dataset at {output_path}...")

    all_news_ids = set()

    for split, n_samples in [('train', train_samples), ('dev', dev_samples)]:
        print(f"Processing '{split}' split...")
        
        # Define paths
        behaviors_file = root_path / split / 'behaviors.tsv'
        news_file = root_path / split / 'news.tsv'
        output_split_path = output_path / split
        
        # Create output directory
        output_split_path.mkdir(exist_ok=True, parents=True)

        # Read n_samples from behaviors.tsv
        behaviors_df = pd.read_csv(
            behaviors_file,
            sep='	',
            header=None,
            nrows=n_samples,
            names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
        )
        
        # Collect all news IDs from the subset of behaviors
        # From 'History' column
        behaviors_df['History'].dropna().apply(lambda x: all_news_ids.update(x.split()))

        # From 'Impressions' column
        def extract_impression_ids(impressions):
            if isinstance(impressions, str):
                all_news_ids.update([imp.split('-')[0] for imp in impressions.split()])
        
        behaviors_df['Impressions'].apply(extract_impression_ids)

        # Save the mini behaviors.tsv
        mini_behaviors_path = output_split_path / 'behaviors.tsv'
        behaviors_df.to_csv(mini_behaviors_path, sep='	', header=False, index=False)
        print(f"  Saved {len(behaviors_df)} samples to {mini_behaviors_path}")

    print(f"Collected {len(all_news_ids)} unique news IDs to keep.")

    # Now filter the news.tsv files to only include the collected news IDs
    for split in ['train', 'dev']:
        print(f"Filtering '{split}/news.tsv'...")
        news_file = root_path / split / 'news.tsv'
        output_news_file = output_path / split / 'news.tsv'

        # Read the full news.tsv
        news_df = pd.read_csv(
            news_file,
            sep='	',
            header=None,
            names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
        )

        # Filter and save
        mini_news_df = news_df[news_df['NewsID'].isin(all_news_ids)]
        mini_news_df.to_csv(output_news_file, sep='	', header=False, index=False)
        print(f"  Saved {len(mini_news_df)} articles to {output_news_file}")
    
    print("Mini dataset creation complete.")


if __name__ == "__main__":
    # Configure paths
    ROOT_DATA_DIR = 'datasets/mind'
    MINI_DATA_DIR = 'datasets/mind_mini'

    if not Path(ROOT_DATA_DIR).exists():
        print(f"Error: The source data directory '{ROOT_DATA_DIR}' does not exist.")
        print("Please ensure the full MIND dataset is downloaded and extracted there.")
        sys.exit(1)

    create_mini_dataset(ROOT_DATA_DIR, MINI_DATA_DIR, train_samples=2000, dev_samples=500) 