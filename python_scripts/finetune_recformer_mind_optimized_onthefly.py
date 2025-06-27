#!/usr/bin/env python3
"""
Optimized on-the-fly training for RecFormer on MIND dataset using Amazon format.
This script converts MIND to Amazon format if needed, then uses smart caching for efficient training.
"""

import os
import sys
import subprocess
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from collections import OrderedDict
import time

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pytorch_lightning import seed_everything

from utils import read_json, AverageMeterSet, Ranker
from optimization import create_optimizer_and_scheduler
from models.recformer.models import RecformerModel, RecformerForSeqRec
from models.recformer.tokenization import RecformerTokenizer
from dataloader_amazon import AmazonDataset


class SmartItemCache:
    """LRU cache for item embeddings with configurable size"""
    
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, item_id):
        if item_id in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(item_id)
            self.hits += 1
            return self.cache[item_id]
        self.misses += 1
        return None
    
    def put(self, item_id, embedding):
        if item_id in self.cache:
            self.cache.move_to_end(item_id)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            self.cache[item_id] = embedding
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def size(self):
        return len(self.cache)


def ensure_amazon_format(mind_path="recbole_data/MIND_mini", amazon_path="datasets/downstream/MIND_mini"):
    """Convert MIND_mini to Amazon format if not already done"""
    
    # Check if Amazon format already exists
    amazon_files = [
        os.path.join(amazon_path, "MIND.train.inter"),
        os.path.join(amazon_path, "MIND.valid.inter"),
        os.path.join(amazon_path, "MIND.test.inter")
    ]
    
    if all(os.path.exists(f) for f in amazon_files):
        print("Amazon format data already exists, skipping conversion...")
        return amazon_path
    
    print("Converting MIND_mini to Amazon format for optimal training...")
    
    # Run the conversion script
    conversion_script = os.path.join(project_root, "python_scripts", "convert_mind_to_amazon.py")
    
    try:
        result = subprocess.run([sys.executable, conversion_script], 
                              capture_output=True, text=True, check=True)
        print("Conversion completed successfully!")
        print(result.stdout)
        return amazon_path
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


def encode_batch_items_smart(model, tokenizer, item_cache, batch_items, device):
    """Smart encoding with caching for batch items"""
    
    # Check cache first
    cached_embeddings = {}
    items_to_encode = []
    
    for item_id in batch_items:
        cached = item_cache.get(item_id)
        if cached is not None:
            cached_embeddings[item_id] = cached
        else:
            items_to_encode.append(item_id)
    
    # Encode uncached items
    if items_to_encode:
        # Get item metadata for encoding
        item_texts = []
        for item_id in items_to_encode:
            # Use placeholder text if metadata not available
            item_texts.append(f"Item {item_id}")
        
        # Tokenize and encode
        encoded = tokenizer(
            item_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.item_encoder(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Pool to get item embedding
        
        # Cache the new embeddings
        for i, item_id in enumerate(items_to_encode):
            embedding = embeddings[i:i+1]
            item_cache.put(item_id, embedding.cpu())
            cached_embeddings[item_id] = embedding
    
    return cached_embeddings


class SmartDataCollator:
    """Data collator that efficiently tracks unique items per batch"""
    
    def __init__(self, tokenizer, max_len=50):
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __call__(self, batch):
        # Extract sequences and targets
        sequences = []
        targets = []
        unique_items = set()
        
        for item in batch:
            seq = item['item_id_list'][:self.max_len]
            target = item['item_id']
            
            sequences.append(seq)
            targets.append(target)
            
            # Track unique items
            unique_items.update(seq)
            unique_items.add(target)
        
        # Convert to tensors
        max_seq_len = max(len(seq) for seq in sequences)
        
        padded_sequences = []
        attention_masks = []
        
        for seq in sequences:
            padded = seq + [0] * (max_seq_len - len(seq))  # 0 is padding token
            mask = [1] * len(seq) + [0] * (max_seq_len - len(seq))
            
            padded_sequences.append(padded)
            attention_masks.append(mask)
        
        return {
            'input_ids': torch.tensor(padded_sequences, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.float),
            'labels': torch.tensor(targets, dtype=torch.long),
            'unique_items': list(unique_items)
        }


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MIND_mini', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='datasets/downstream/MIND_mini', 
                       help='Path to Amazon format dataset')
    parser.add_argument('--model_name', type=str, default='RecFormer-AmazonSports')
    parser.add_argument('--output_dir', type=str, default='output/mind_onthefly_optimized')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--cache_size', type=int, default=50000)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--save_steps', type=int, default=2000)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Setup
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ensure Amazon format data exists
    dataset_dir = ensure_amazon_format()
    args.data_dir = dataset_dir
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize smart cache
    item_cache = SmartItemCache(max_size=args.cache_size)
    print(f"Initialized smart cache with size: {args.cache_size}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name)
    model = RecformerForSeqRec.from_pretrained(args.model_name)
    model.to(device)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = AmazonDataset(
        os.path.join(args.data_dir, 'MIND.train.inter'),
        max_len=args.max_len
    )
    
    valid_dataset = AmazonDataset(
        os.path.join(args.data_dir, 'MIND.valid.inter'),
        max_len=args.max_len
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    
    # Create data collator
    collate_fn = SmartDataCollator(tokenizer, max_len=args.max_len)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Setup optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, 
        learning_rate=args.learning_rate,
        num_training_steps=len(train_loader) * args.num_epochs // args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps
    )
    
    # Setup for mixed precision if requested
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    model.train()
    global_step = 0
    best_valid_loss = float('inf')
    
    meters = AverageMeterSet()
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        epoch_start_time = time.time()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            unique_items = batch['unique_items']
            
            # Smart item encoding with caching
            cached_embeddings = encode_batch_items_smart(
                model, tokenizer, item_cache, unique_items, device
            )
            
            # Temporarily update model's item embeddings
            original_embeddings = {}
            for item_id in unique_items:
                if hasattr(model, 'item_embeddings'):
                    original_embeddings[item_id] = model.item_embeddings.weight[item_id].clone()
                    model.item_embeddings.weight[item_id] = cached_embeddings[item_id].squeeze(0)
            
            # Forward pass
            if args.fp16:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = outputs.loss if hasattr(outputs, 'loss') else \
                           nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                    loss = loss / args.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = outputs.loss if hasattr(outputs, 'loss') else \
                       nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
            
            # Restore original embeddings
            for item_id in unique_items:
                if item_id in original_embeddings:
                    model.item_embeddings.weight[item_id] = original_embeddings[item_id]
            
            # Update meters
            meters.update('loss', loss.item() * args.gradient_accumulation_steps)
            meters.update('cache_hit_rate', item_cache.hit_rate())
            meters.update('cache_size', item_cache.size())
            
            # Gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % args.logging_steps == 0:
                    print(f"Step {global_step}: Loss={meters.loss.avg:.4f}, "
                          f"Cache Hit Rate={meters.cache_hit_rate.avg:.3f}, "
                          f"Cache Size={meters.cache_size.avg:.0f}")
                
                # Evaluation
                if global_step % args.eval_steps == 0:
                    print("Running validation...")
                    model.eval()
                    valid_loss = 0
                    valid_steps = 0
                    
                    with torch.no_grad():
                        for valid_batch in tqdm(valid_loader, desc="Validation"):
                            v_input_ids = valid_batch['input_ids'].to(device)
                            v_attention_mask = valid_batch['attention_mask'].to(device)
                            v_labels = valid_batch['labels'].to(device)
                            v_unique_items = valid_batch['unique_items']
                            
                            # Smart encoding for validation
                            v_cached_embeddings = encode_batch_items_smart(
                                model, tokenizer, item_cache, v_unique_items, device
                            )
                            
                            # Temporary embedding update
                            v_original_embeddings = {}
                            for item_id in v_unique_items:
                                if hasattr(model, 'item_embeddings'):
                                    v_original_embeddings[item_id] = model.item_embeddings.weight[item_id].clone()
                                    model.item_embeddings.weight[item_id] = v_cached_embeddings[item_id].squeeze(0)
                            
                            # Forward pass
                            if args.fp16:
                                with autocast():
                                    v_outputs = model(input_ids=v_input_ids, attention_mask=v_attention_mask)
                                    v_loss = v_outputs.loss if hasattr(v_outputs, 'loss') else \
                                             nn.CrossEntropyLoss()(v_outputs.logits.view(-1, v_outputs.logits.size(-1)), v_labels.view(-1))
                            else:
                                v_outputs = model(input_ids=v_input_ids, attention_mask=v_attention_mask)
                                v_loss = v_outputs.loss if hasattr(v_outputs, 'loss') else \
                                         nn.CrossEntropyLoss()(v_outputs.logits.view(-1, v_outputs.logits.size(-1)), v_labels.view(-1))
                            
                            # Restore embeddings
                            for item_id in v_unique_items:
                                if item_id in v_original_embeddings:
                                    model.item_embeddings.weight[item_id] = v_original_embeddings[item_id]
                            
                            valid_loss += v_loss.item()
                            valid_steps += 1
                    
                    valid_loss /= valid_steps
                    print(f"Validation Loss: {valid_loss:.4f}")
                    
                    # Save best model
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        model.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                        print("Saved new best model!")
                    
                    model.train()
                
                # Regular model saving
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"Saved checkpoint at step {global_step}")
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Final cache statistics - Hit rate: {item_cache.hit_rate():.3f}, Size: {item_cache.size()}")
    
    # Final save
    final_save_path = os.path.join(args.output_dir, 'final_model')
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Final cache hit rate: {item_cache.hit_rate():.3f}")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 