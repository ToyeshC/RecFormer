import os
import sys
import json
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from collections import defaultdict

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pytorch_lightning import seed_everything

from utils import read_json, AverageMeterSet, Ranker
from optimization import create_optimizer_and_scheduler
from models.recformer.models import RecformerModel, RecformerForSeqRec, RecformerConfig
from models.recformer.tokenization import RecformerTokenizer
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerDataset


def load_data(args):
    train = read_json(os.path.join(args.data_path, args.train_file))
    val = read_json(os.path.join(args.data_path, args.dev_file))
    test = read_json(os.path.join(args.data_path, args.test_file))
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))
    
    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    id2item = {v:k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train, val, test, item_meta_dict_filted, item2id, id2item


tokenizer_glb: RecformerTokenizer = None
def _par_tokenize_doc(doc):
    item_id, item_attr = doc
    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)
    return item_id, input_ids, token_type_ids


class OnTheFlyDataCollator:
    """
    Custom data collator that keeps track of item IDs in the batch
    for on-the-fly encoding
    """
    def __init__(self, tokenizer, tokenized_items, is_eval=False):
        self.tokenizer = tokenizer
        self.tokenized_items = tokenized_items
        self.is_eval = is_eval
        self.original_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items) if is_eval else FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    
    def __call__(self, batch_data):
        # Extract all unique item IDs before collation
        unique_items = set()
        
        for data_point in batch_data:
            # Extract items from the sequence
            if 'items' in data_point:
                unique_items.update(data_point['items'])
            
            # Extract items from labels/label
            if not self.is_eval and 'labels' in data_point:
                unique_items.update(data_point['labels'])
            elif self.is_eval and 'label' in data_point:
                if isinstance(data_point['label'], list):
                    unique_items.update(data_point['label'])
                else:
                    unique_items.add(data_point['label'])
        
        # Call original collator
        result = self.original_collator(batch_data)
        
        # Add unique items to the batch
        if self.is_eval:
            batch, labels = result
            batch['_unique_items'] = list(unique_items)
            return batch, labels
        else:
            result['_unique_items'] = list(unique_items)
            return result


def encode_batch_items_onthefly(model: RecformerModel, tokenizer: RecformerTokenizer, 
                               batch_item_ids, tokenized_items, args):
    """Encode only the items that appear in the current batch"""
    if not batch_item_ids:
        return torch.empty(0, model.config.hidden_size).to(args.device), []
    
    model.eval()
    
    # Get tokenized representations for batch items only
    batch_items = []
    valid_item_ids = []
    
    for item_id in batch_item_ids:
        if item_id in tokenized_items:
            batch_items.append([tokenized_items[item_id]])
            valid_item_ids.append(item_id)
    
    if not batch_items:
        return torch.empty(0, model.config.hidden_size).to(args.device), []
    
    # Encode in smaller sub-batches to manage memory
    embeddings = []
    sub_batch_size = min(args.onthefly_batch_size, len(batch_items))
    
    with torch.no_grad():
        for i in range(0, len(batch_items), sub_batch_size):
            sub_batch = batch_items[i:i+sub_batch_size]
            
            inputs = tokenizer.batch_encode(sub_batch, encode_item=False)
            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)
            
            # Handle multi-GPU case - use the longformer directly
            if args.multi_gpu and isinstance(model, nn.DataParallel):
                outputs = model.module.longformer(**inputs)
            else:
                outputs = model.longformer(**inputs)
            
            embeddings.append(outputs.pooler_output.detach())
            
            # Clear intermediate tensors
            del inputs, outputs
    
    if embeddings:
        batch_embeddings = torch.cat(embeddings, dim=0)
    else:
        batch_embeddings = torch.empty(0, model.config.hidden_size).to(args.device)
    
    return batch_embeddings, valid_item_ids


def eval_onthefly(model, dataloader, tokenized_items, args):
    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    for batch, labels in tqdm(dataloader, ncols=100, desc='Evaluate (On-the-fly)'):
        # Extract unique items from this batch
        batch_item_ids = batch.pop('_unique_items')
        
        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)
        
        # Encode only items in this batch ON-THE-FLY
        batch_embeddings, valid_item_ids = encode_batch_items_onthefly(
            model, tokenizer_glb, batch_item_ids, tokenized_items, args)
        
        # Create a mapping from item ID to its fresh embedding
        item_to_embedding = {}
        for i, item_id in enumerate(valid_item_ids):
            if i < len(batch_embeddings):
                item_to_embedding[item_id] = batch_embeddings[i]

        # Temporarily replace the item embeddings in the model
        original_embeddings = None
        if hasattr(model, 'module'):
            original_embeddings = model.module.item_embedding.weight.data.clone()
            # Update embeddings for items in this batch
            for item_id, embedding in item_to_embedding.items():
                if item_id < model.module.item_embedding.weight.size(0):
                    model.module.item_embedding.weight.data[item_id] = embedding
        else:
            original_embeddings = model.item_embedding.weight.data.clone()
            for item_id, embedding in item_to_embedding.items():
                if item_id < model.item_embedding.weight.size(0):
                    model.item_embedding.weight.data[item_id] = embedding

        with torch.no_grad():
            scores = model(**batch)
            
        # Restore original embeddings immediately
        if hasattr(model, 'module'):
            model.module.item_embedding.weight.data = original_embeddings
        else:
            model.item_embedding.weight.data = original_embeddings

        res = ranker(scores, labels)

        metrics = {}
        for i, k in enumerate(args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        for k, v in metrics.items():
            average_meter_set.update(k, v)
        
        # Memory cleanup
        del batch, labels, scores, res, batch_embeddings, item_to_embedding, original_embeddings
        torch.cuda.empty_cache()

    average_metrics = average_meter_set.averages()
    return average_metrics


def train_one_epoch_onthefly(model, dataloader, optimizer, scheduler, scaler, tokenized_items, args, epoch):
    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Training (On-the-fly)')):
        # Extract unique items from this batch
        batch_item_ids = batch.pop('_unique_items')
        
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        # Encode only items in this batch ON-THE-FLY
        batch_embeddings, valid_item_ids = encode_batch_items_onthefly(
            model, tokenizer_glb, batch_item_ids, tokenized_items, args)
        
        # Create a mapping from item ID to its fresh embedding
        item_to_embedding = {}
        for i, item_id in enumerate(valid_item_ids):
            if i < len(batch_embeddings):
                item_to_embedding[item_id] = batch_embeddings[i]

        # Temporarily replace the item embeddings in the model
        original_embeddings = None
        if hasattr(model, 'module'):
            original_embeddings = model.module.item_embedding.weight.data.clone()
            # Update embeddings for items in this batch
            for item_id, embedding in item_to_embedding.items():
                if item_id < model.module.item_embedding.weight.size(0):
                    model.module.item_embedding.weight.data[item_id] = embedding
        else:
            original_embeddings = model.item_embedding.weight.data.clone()
            for item_id, embedding in item_to_embedding.items():
                if item_id < model.item_embedding.weight.size(0):
                    model.item_embedding.weight.data[item_id] = embedding

        if args.fp16:
            with autocast():
                loss = model(**batch)
        else:
            loss = model(**batch)

        # Handle DataParallel loss reduction
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Restore original embeddings after backward pass
        if hasattr(model, 'module'):
            model.module.item_embedding.weight.data = original_embeddings
        else:
            model.item_embedding.weight.data = original_embeddings

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                optimizer.zero_grad()

                if optimizer_was_run:
                    scheduler.step()
            else:
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
        
        # Memory cleanup every 50 steps (more frequent due to on-the-fly computation)
        if step % 50 == 0:
            del batch, loss, batch_embeddings, item_to_embedding, original_embeddings
            torch.cuda.empty_cache()
            gc.collect()
        
        # Print memory usage every 200 steps
        if step % 200 == 0:
            print(f"Epoch {epoch}, Step {step}: GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
            print(f"  Batch items encoded: {len(valid_item_ids)} / {len(batch_item_ids)} total unique")


def main():
    parser = ArgumentParser()
    # path and file
    parser.add_argument('--pretrain_ckpt', type=str, default=None, required=True)
    parser.add_argument('--data_path', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt', type=str, default='best_model.bin')
    parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=16, help="Number of workers for preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help="Number of workers for data loading.")

    # model
    parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")

    # train
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=-1)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=1)
    
    # Multi-GPU settings
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs with DataParallel')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='GPU IDs to use (comma-separated)')
    
    # On-the-fly specific settings
    parser.add_argument('--onthefly_batch_size', type=int, default=32, help='Batch size for on-the-fly item encoding')
    
    # Performance optimizations
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Use pinned memory for faster data transfer')

    args = parser.parse_args()
    print(args)
    seed_everything(42)
    
    # Set up device and multi-GPU
    if args.multi_gpu and torch.cuda.device_count() > 1:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        gpu_ids = [x for x in gpu_ids if x < torch.cuda.device_count()]
        args.device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"Using multi-GPU training on GPUs: {gpu_ids}")
    else:
        args.device = torch.device('cuda:{}'.format(args.device)) if args.device>=0 else torch.device('cpu')
        print(f"Using single GPU: {args.device}")

    # Print initial memory status
    print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    
    global tokenizer_glb
    tokenizer_glb = tokenizer
    
    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)

    print('Tokenize all items for on-the-fly encoding...')
    pool = Pool(args.preprocessing_num_workers)
    item_meta_list = list(item_meta_dict.items())
    
    chunk_size = max(1, len(item_meta_list) // (args.preprocessing_num_workers * 4))
    print(f"Using {args.preprocessing_num_workers} workers with chunksize {chunk_size} for tokenization.")

    tokenized_items_list = pool.map(_par_tokenize_doc, item_meta_list, chunksize=chunk_size)
    tokenized_items = {item2id[item[0]]: (item[1], item[2]) for item in tokenized_items_list}
    pool.close()
    pool.join()
    
    # Memory cleanup after tokenization
    del item_meta_list, tokenized_items_list
    gc.collect()
    
    # Custom collators that track item IDs for on-the-fly encoding
    finetune_data_collator = OnTheFlyDataCollator(tokenizer, tokenized_items, is_eval=False)
    eval_data_collator = OnTheFlyDataCollator(tokenizer, tokenized_items, is_eval=True)

    train_data = RecformerDataset(args, train, val, test, mode='train')
    val_data = RecformerDataset(args, train, val, test, mode='val')
    test_data = RecformerDataset(args, train, val, test, mode='test')

    # DataLoaders for on-the-fly approach
    train_loader = DataLoader(train_data, 
                               batch_size=args.batch_size, 
                               shuffle=True, 
                               collate_fn=finetune_data_collator,
                               num_workers=args.dataloader_num_workers,
                               pin_memory=args.pin_memory)
    dev_loader = DataLoader(val_data, 
                             batch_size=args.batch_size, 
                             collate_fn=eval_data_collator,
                             num_workers=args.dataloader_num_workers,
                             pin_memory=args.pin_memory)
    test_loader = DataLoader(test_data, 
                             batch_size=args.batch_size, 
                             collate_fn=eval_data_collator,
                             num_workers=args.dataloader_num_workers,
                             pin_memory=args.pin_memory)

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt, weights_only=False)
    model.load_state_dict(pretrain_ckpt, strict=False)
    model.to(args.device)

    # Set up multi-GPU if requested
    if args.multi_gpu and torch.cuda.device_count() > 1:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        gpu_ids = [x for x in gpu_ids if x < torch.cuda.device_count()]
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            print(f"Model wrapped with DataParallel using GPUs: {gpu_ids}")

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        if isinstance(model, nn.DataParallel):
            for param in model.module.longformer.embeddings.word_embeddings.parameters():
                param.requires_grad = False
        else:
            for param in model.longformer.embeddings.word_embeddings.parameters():
                param.requires_grad = False

    # Initialize with random item embeddings (will be computed on-the-fly)
    print("Initializing model with random item embeddings (will compute on-the-fly)")
    if isinstance(model, nn.DataParallel):
        model.module.init_item_embedding(torch.randn(len(item2id), config.hidden_size).to(args.device))
    else:
        model.init_item_embedding(torch.randn(len(item2id), config.hidden_size).to(args.device))

    model.to(args.device)

    # Memory cleanup after model setup
    torch.cuda.empty_cache()
    gc.collect()
    print(f"After model setup GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Initial evaluation
    test_metrics = eval_onthefly(model, test_loader, tokenized_items, args)
    print(f'Initial Test set: {test_metrics}')
    
    best_target = float('-inf')
    patience_counter = 0
    
    print(f"\n=== Starting On-the-Fly Training ===")
    print(f"Total epochs: {args.num_train_epochs}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"On-the-fly encoding batch size: {args.onthefly_batch_size}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * len(gpu_ids) if args.multi_gpu else args.batch_size * args.gradient_accumulation_steps}")
    print("🚀 ADVANTAGE: Item embeddings are computed fresh for each batch!")
    print("💡 BENEFIT: Lower memory usage, better training dynamics")

    # Stage 1: Initial training with patience=5
    print(f"\n=== Stage 1: On-the-Fly Training (patience=5) ===")
    patient = 5

    for epoch in range(args.num_train_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_train_epochs}")
        
        # No pre-computation needed - everything is on-the-fly!
        train_one_epoch_onthefly(model, train_loader, optimizer, scheduler, scaler, tokenized_items, args, epoch)
        
        # Memory cleanup after each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval_onthefly(model, dev_loader, tokenized_items, args)
            print(f'Epoch: {epoch+1}. Dev set: {dev_metrics}')
            print(f"GPU Memory after evaluation: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 5
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), args.ckpt)
                else:
                    torch.save(model.state_dict(), args.ckpt)
                patience_counter = 0
            else:
                patient -= 1
                patience_counter += 1
                print(f'No improvement. Patience: {patient}/5')
                if patient == 0:
                    print("Early stopping triggered in Stage 1")
                    break
    
    # Stage 2: Fine-tuning with best model and patience=3
    print(f"\n=== Stage 2: On-the-Fly Fine-tuning (patience=3) ===")
    print('Load best model from stage 1.')
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(args.ckpt, weights_only=False))
    else:
        model.load_state_dict(torch.load(args.ckpt, weights_only=False))

    patient = 3

    for epoch in range(args.num_train_epochs):
        print(f"\nStage 2 - Epoch {epoch+1}/{args.num_train_epochs}")
        
        train_one_epoch_onthefly(model, train_loader, optimizer, scheduler, scaler, tokenized_items, args, epoch)
        
        # Memory cleanup after each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval_onthefly(model, dev_loader, tokenized_items, args)
            print(f'Epoch: {epoch+1}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 3
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), args.ckpt)
                else:
                    torch.save(model.state_dict(), args.ckpt)
            else:
                patient -= 1
                print(f'No improvement. Patience: {patient}/3')
                if patient == 0:
                    print("Early stopping triggered in Stage 2")
                    break

    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    print('Test with the best checkpoint.')
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(args.ckpt, weights_only=False))
    else:
        model.load_state_dict(torch.load(args.ckpt, weights_only=False))
    
    test_metrics = eval_onthefly(model, test_loader, tokenized_items, args)
    print(f'Final Test set: {test_metrics}')
    
    print(f"\n=== On-the-Fly Training Completed ===")
    print(f"🎯 Best NDCG@10: {best_target:.6f}")
    print(f"📊 Total patience triggers: {patience_counter}")
    print(f"💾 Final GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    print("✅ ON-THE-FLY TRAINING: Item embeddings computed fresh every batch!")
               
if __name__ == "__main__":
    main() 