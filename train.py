import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
import numpy as np
import os
import wandb
import pickle
import math

# Custom data loader
def get_batch(split):
    data_path = os.path.join('processed', f'{split}.bin')
    labels_path = os.path.join('processed', f'{split}_labels.pkl')
    data = np.memmap(data_path, dtype=np.uint16, mode='r').reshape(-1, block_size)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    ix = torch.randint(len(labels), (batch_size,))
    x = torch.from_numpy(data[ix].astype(np.int64))
    y = torch.tensor([labels[i] for i in ix], dtype=torch.long)
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Config
out_dir = 'out-sentiment'
eval_interval = 500
eval_iters = 20
log_interval = 100
batch_size = 64
block_size = 512  # Matches your max_length
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
learning_rate = 3e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 3e-5
warmup_iters = 100
num_classes = 3
compile = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Initialize W&B
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
})

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
if compile:
    model = torch.compile(model)

# Optimizer
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, betas=(0.9, 0.95))

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

# Training loop
os.makedirs(out_dir, exist_ok=True)
for iter_num in range(max_iters):
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    x, y = get_batch('train')
    logits, lm_loss, sentiment_loss = model(x, labels=y)
    loss = sentiment_loss

    preds = torch.argmax(logits, dim=-1)
    train_accuracy = (preds == y).float().mean().item()

    if iter_num % log_interval == 0:
        print(f"Iter {iter_num}, Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}")
        wandb.log({
            "train_loss": loss.item(),
            "train_accuracy": train_accuracy,
            "learning_rate": lr,
            "iteration": iter_num
        })

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter_num % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for _ in range(eval_iters):
                x_val, y_val = get_batch('val')
                logits, _, val_sentiment_loss = model(x_val, labels=y_val)
                val_loss += val_sentiment_loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
            val_loss /= eval_iters
            val_accuracy = correct / total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "iteration": iter_num
            })
        model.train()

# Save checkpoint
torch.save(model.state_dict(), f'{out_dir}/model.pt')
wandb.finish()