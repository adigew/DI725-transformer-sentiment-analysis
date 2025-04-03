import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
import numpy as np
import os
import wandb
import pickle
import math
from torch.utils.data import DataLoader, Dataset
import time

# Define base directory
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, 'data', 'sentiment', 'processed')

# Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, split):
        data_path = os.path.join(processed_dir, f'{split}.bin')
        labels_path = os.path.join(processed_dir, f'{split}_labels.pkl')
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r').reshape(-1, block_size)
        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)
        print(f"Loaded {split} dataset: {len(self.labels)} samples", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx].astype(np.int64))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Config
out_dir = 'out-sentiment'
eval_interval = 100
eval_iters = 10
log_interval = 50
batch_size = 32
block_size = 512
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
learning_rate = 3e-4
max_iters = 600  # Reduced since peak performance is around 400-500
lr_decay_iters = 600
min_lr = 3e-5
warmup_iters = 100
num_classes = 3
compile = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(f"Using device: {device}", flush=True)

# DataLoaders
train_dataset = SentimentDataset('train')
val_dataset = SentimentDataset('val')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
print("DataLoaders initialized", flush=True)

# Initialize W&B in online mode
wandb.init(project="nanoGPT-sentiment", config={
    "batch_size": batch_size,
    "block_size": block_size,
    "max_iters": max_iters,
    "learning_rate": learning_rate,
    "n_layer": n_layer,
    "n_head": n_head,
    "n_embd": n_embd,
    "dropout": dropout,
    "num_classes": num_classes,
    "eval_interval": eval_interval,
    "eval_iters": eval_iters,
    "lr_decay_iters": lr_decay_iters,
    "min_lr": min_lr,
    "warmup_iters": warmup_iters,
    "device": device,
})
print("W&B initialized in online mode. View live at: https://wandb.ai/<your-username>/nanoGPT-sentiment", flush=True)

# Model
config = GPTConfig(
    block_size=block_size,
    vocab_size=50304,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    num_classes=num_classes
)
model = GPT(config)
model.to(device)
print("Model moved to device", flush=True)
if compile:
    model = torch.compile(model)

# Optimizer
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, betas=(0.9, 0.95))
print("Optimizer initialized", flush=True)

# Learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Training loop with early stopping
os.makedirs(out_dir, exist_ok=True)
train_iterator = iter(train_loader)
print("Starting training loop", flush=True)
best_val_loss = float('inf')
patience = 2  # Stop after 2 consecutive increases in val loss
patience_counter = 0

try:
    for iter_num in range(max_iters):
        start_time = time.time()

        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Data loading
        data_start = time.time()
        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        data_time = time.time() - data_start

        # Forward pass
        forward_start = time.time()
        logits, lm_loss, sentiment_loss = model(x, labels=y)
        loss = sentiment_loss
        forward_time = time.time() - forward_start

        # Backward pass
        backward_start = time.time()
        preds = torch.argmax(logits, dim=-1)
        train_accuracy = (preds == y).float().mean().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start

        if iter_num % log_interval == 0 or iter_num == 0:
            total_time = time.time() - start_time
            print(f"Iter {iter_num}, Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, "
                  f"Data: {data_time:.3f}s, Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s, Total: {total_time:.3f}s", flush=True)
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": train_accuracy,
                "learning_rate": lr,
                "iteration": iter_num,
                "data_time": data_time,
                "forward_time": forward_time,
                "backward_time": backward_time,
            })

        if iter_num % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                for i, (x_val, y_val) in enumerate(val_loader):
                    if i >= eval_iters:
                        break
                    if device_type == 'cuda':
                        x_val, y_val = x_val.pin_memory().to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                    else:
                        x_val, y_val = x_val.to(device), y_val.to(device)
                    logits, _, val_sentiment_loss = model(x_val, labels=y_val)
                    val_loss += val_sentiment_loss.item()
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == y_val).sum().item()
                    total += y_val.size(0)
                val_loss /= eval_iters
                val_accuracy = correct / total
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}", flush=True)
                wandb.log({
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "iteration": iter_num
                })

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'{out_dir}/best_model.pt')  # Save best model
                else:
                    patience_counter += 1
                    print(f"Patience counter: {patience_counter}/{patience}", flush=True)
                    if patience_counter >= patience:
                        print("Early stopping triggered. Stopping training.", flush=True)
                        break

            model.train()

except Exception as e:
    print(f"Error during training: {e}", flush=True)
    raise

# Save final checkpoint
torch.save(model.state_dict(), f'{out_dir}/final_model.pt')
wandb.finish()
print("Training complete. View results at: https://wandb.ai/<your-username>/nanoGPT-sentiment", flush=True)