"""
Training script for a character-level GPT model on sentiment analysis.
Incorporates configuration from config/train_sentiment.py.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
import wandb
import pickle
import math
from torch.utils.data import DataLoader, Dataset
import time

# Enable TF32 for better performance on compatible GPUs
torch.set_float32_matmul_precision('high')

# Model definition (nanoGPT-style for character-level)
class LayerNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    def __init__(self, block_size=512, vocab_size=None, n_layer=6, n_head=6, n_embd=384, dropout=0.1, num_classes=3):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.num_classes = num_classes

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.sentiment_head = nn.Linear(config.n_embd, config.num_classes)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, labels=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            lm_loss = None

        sentiment_logits = self.sentiment_head(x[:, -1, :])
        if labels is not None:
            sentiment_loss = F.cross_entropy(sentiment_logits, labels)
        else:
            sentiment_loss = None

        return sentiment_logits, lm_loss, sentiment_loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

# Dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, 'data', 'sentiment', 'processed')

class SentimentDataset(Dataset):
    def __init__(self, split):
        data_path = os.path.join(processed_dir, f'{split}.bin')
        labels_path = os.path.join(processed_dir, f'{split}_labels.pkl')
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r').reshape(-1, 512)
        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)
        print(f"Loaded {split} dataset: {len(self.labels)} samples", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx].astype(np.int64))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Load vocabulary
with open(os.path.join(processed_dir, 'vocab.pkl'), 'rb') as f:
    vocab = pickle.load(f)
vocab_size = vocab['vocab_size']

# Configuration from config/train_sentiment.py
out_dir = 'out-sentiment-scratch'
eval_interval = 500
eval_iters = 20
log_interval = 100
batch_size = 64
block_size = 512
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
compile = False  # Disabled torch.compile to avoid Triton dependency

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}", flush=True)

# DataLoaders
train_dataset = SentimentDataset('train')
val_dataset = SentimentDataset('val')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
print("DataLoaders initialized", flush=True)

# WANDB
wandb.init(project="nanoGPT-sentiment-scratch", config={
    "out_dir": out_dir,
    "eval_interval": eval_interval,
    "eval_iters": eval_iters,
    "log_interval": log_interval,
    "batch_size": batch_size,
    "block_size": block_size,
    "n_layer": n_layer,
    "n_head": n_head,
    "n_embd": n_embd,
    "dropout": dropout,
    "learning_rate": learning_rate,
    "max_iters": max_iters,
    "lr_decay_iters": lr_decay_iters,
    "min_lr": min_lr,
    "warmup_iters": warmup_iters,
    "num_classes": num_classes,
    "vocab_size": vocab_size,
    "device": device,
    "model": "scratch-char-level"
})

# Model
config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    num_classes=num_classes
)
model = GPT(config)
model.to(device)
if compile and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile for faster training...")
    model = torch.compile(model)
print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)

# Optimizer
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, betas=(0.9, 0.95))

# Learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Training loop
os.makedirs(out_dir, exist_ok=True)
train_iterator = iter(train_loader)
best_val_loss = float('inf')
patience = 2
patience_counter = 0

for iter_num in range(max_iters):
    start_time = time.time()
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    try:
        x, y = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_loader)
        x, y = next(train_iterator)
    x, y = x.to(device), y.to(device)

    sentiment_logits, _, sentiment_loss = model(x, labels=y)
    loss = sentiment_loss

    preds = torch.argmax(sentiment_logits, dim=-1)
    train_accuracy = (preds == y).float().mean().item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter_num % log_interval == 0 or iter_num == 0:
        total_time = time.time() - start_time
        print(f"Iter {iter_num}, Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, Time: {total_time:.3f}s", flush=True)
        wandb.log({"train_loss": loss.item(), "train_accuracy": train_accuracy, "learning_rate": lr, "iteration": iter_num})

    if iter_num % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for i, (x_val, y_val) in enumerate(val_loader):
                if i >= eval_iters:
                    break
                x_val, y_val = x_val.to(device), y_val.to(device)
                sentiment_logits, _, val_sentiment_loss = model(x_val, labels=y_val)
                val_loss += val_sentiment_loss.item()
                preds = torch.argmax(sentiment_logits, dim=-1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
            val_loss /= eval_iters
            val_accuracy = correct / total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}", flush=True)
            wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "iteration": iter_num})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'{out_dir}/best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.", flush=True)
                    break
        model.train()

torch.save(model.state_dict(), f'{out_dir}/final_model.pt')
wandb.finish()
print("Training complete.", flush=True)