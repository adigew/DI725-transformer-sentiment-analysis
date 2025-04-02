import os
import argparse
import torch
import torch.nn as nn
import wandb
import pickle
import numpy as np
from model_sentiment import GPT, GPTConfig

def load_data_file(file_path, dtype):
    try:
        return np.memmap(file_path, dtype=dtype, mode='r')
    except:
        try:
            with open(file_path, 'rb') as f:
                return np.frombuffer(f.read(), dtype=dtype)
        except Exception as e:
            raise RuntimeError(f"Failed to load data file {file_path}: {str(e)}")

def validate_data(block_size, data_dir='data/sentiment/processed'):
    print("Validating dataset...")
    for split in ['train', 'val']:
        data_path = os.path.join(data_dir, f'{split}.bin')
        labels_path = os.path.join(data_dir, f'{split}_labels.pkl')
        
        try:
            data = load_data_file(data_path, np.uint16)
            with open(labels_path, 'rb') as f:
                labels = pickle.load(f)
                
            max_sequences = len(data) // block_size
            print(f"{split} split: {len(data)} tokens, {len(labels)} labels, {max_sequences} possible sequences")
            
            if len(labels) < max_sequences:
                print(f"Warning: {split} split has {len(labels)} labels but {max_sequences} possible sequences")
            elif len(labels) > max_sequences:
                print(f"Warning: {split} split has {len(labels)} labels but only {max_sequences} possible sequences")
                
        except Exception as e:
            print(f"Error validating {split} split: {str(e)}")
            raise

def get_batch(split, block_size, batch_size, device, data_dir='data/sentiment/processed'):
    data_path = os.path.join(data_dir, f'{split}.bin')
    labels_path = os.path.join(data_dir, f'{split}_labels.pkl')
    
    data = load_data_file(data_path, np.uint16)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    
    max_sequences = len(data) // block_size
    max_valid_idx = min(max_sequences - 1, len(labels) - 1)
    
    if max_valid_idx <= 0:
        raise ValueError(f"Not enough data in {split} split (max_valid_idx={max_valid_idx})")
    
    ix = torch.randint(0, max_valid_idx, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i*block_size:(i+1)*block_size].copy().astype(np.int64))
        for i in ix
    ])
    y = torch.tensor([labels[i] for i in ix], dtype=torch.long)
    return x.to(device), y.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--compile', type=str, default='True')
    args = parser.parse_args()

    # Load config from file
    config_dict = {}
    with open(args.config) as f:
        exec(f.read(), config_dict)
    
    # Filter out non-serializable items
    config_dict = {k: v for k, v in config_dict.items() 
                 if not k.startswith('__') 
                 and not isinstance(v, type(os))}
    
    # Create GPTConfig object (model parameters only)
    model_config = GPTConfig(
        n_layer=config_dict.get('n_layer', 6),
        n_head=config_dict.get('n_head', 6),
        n_embd=config_dict.get('n_embd', 384),
        block_size=config_dict.get('block_size', 128),
        dropout=config_dict.get('dropout', 0.1),
        num_classes=config_dict.get('num_classes', 2),
        vocab_size=config_dict.get('vocab_size', 50257)
    )
    
    # Training parameters
    batch_size = config_dict.get('batch_size', 32)
    learning_rate = config_dict.get('learning_rate', 3e-4)
    max_iters = config_dict.get('max_iters', 1000)
    eval_interval = config_dict.get('eval_interval', 100)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    validate_data(model_config.block_size)

    wandb.init(
        project=config_dict.get('wandb_project', 'nanoGPT-sentiment'),
        config={
            'n_layer': model_config.n_layer,
            'n_head': model_config.n_head,
            'n_embd': model_config.n_embd,
            'block_size': model_config.block_size,
            'dropout': model_config.dropout,
            'num_classes': model_config.num_classes,
            'vocab_size': model_config.vocab_size,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_iters': max_iters,
            'eval_interval': eval_interval
        }
    )

    model = GPT(model_config).to(device)
    if args.compile.lower() == 'true' and hasattr(torch, 'compile'):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    try:
        for iter in range(max_iters):
            try:
                x, y = get_batch('train', model_config.block_size, batch_size, device)
                with torch.autocast(device_type=device, dtype=dtype):
                    _, sentiment_logits, loss = model(x, None, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if iter % eval_interval == 0:
                    model.eval()
                    val_loss, val_acc = 0, 0
                    valid_samples = 0
                    for _ in range(100):
                        try:
                            x_val, y_val = get_batch('val', model_config.block_size, batch_size, device)
                            with torch.no_grad():
                                _, val_sentiment_logits, val_loss_batch = model(x_val, None, y_val)
                            val_loss += val_loss_batch.item()
                            val_acc += (val_sentiment_logits.argmax(-1) == y_val).float().mean().item()
                            valid_samples += 1
                        except Exception as e:
                            print(f"Validation error: {str(e)}")
                            continue
                    
                    if valid_samples > 0:
                        val_loss /= valid_samples
                        val_acc /= valid_samples
                        wandb.log({
                            'iter': iter,
                            'train/loss': loss.item(),
                            'val/loss': val_loss,
                            'val/acc': val_acc
                        })
                        print(f"Iter {iter}: train loss {loss.item():.4f}, val loss {val_loss:.4f}, val acc {val_acc:.4f}")

            except Exception as e:
                print(f"Error during iteration {iter}: {str(e)}")
                continue

    finally:
        wandb.finish()
        torch.save(model.state_dict(), 'sentiment_model.pt')
        print("Training complete! Model saved to sentiment_model.pt")

if __name__ == '__main__':
    main()