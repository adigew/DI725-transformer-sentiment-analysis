"""
Fine-tuning script for pre-trained GPT-2 on sentiment analysis task.
References:
1) OpenAI GPT-2: https://github.com/openai/gpt-2/blob/master/src/model.py
2) Hugging Face GPT-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import torch
from torch.nn import functional as F
import numpy as np
import os
import wandb
import pickle
import math
from torch.utils.data import DataLoader, Dataset
import time
from transformers import GPT2Model, GPT2Tokenizer

# Import the GPT model definition
import math
import inspect
from dataclasses import dataclass
import torch.nn as nn

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=True)
        else:
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
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
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
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    num_classes: int = 3

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.sentiment_head = nn.Linear(config.n_embd, config.num_classes)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("Number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

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
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            lm_logits = self.lm_head(x)
            lm_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            lm_logits = self.lm_head(x[:, [-1], :])
            lm_loss = None

        sentiment_logits = self.sentiment_head(x[:, -1, :])
        if labels is not None:
            sentiment_loss = F.cross_entropy(sentiment_logits, labels)
        else:
            sentiment_loss = None

        return sentiment_logits, lm_loss, sentiment_loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained GPT: {model_type}")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        config_args['dropout'] = override_args.get('dropout', 0.1)
        config_args['num_classes'] = 3
        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias') and 'sentiment_head' not in k]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        model.load_state_dict(sd, strict=False)
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'sentiment_head' not in n]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and 'sentiment_head' not in n]
        sentiment_params = [p for n, p in param_dict.items() if 'sentiment_head' in n]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': sentiment_params, 'weight_decay': 0.0, 'lr': learning_rate * 10},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        print(f"Using fused AdamW: {use_fused}")
        return optimizer

# Define base directory
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, 'data', 'sentiment', 'processed')

# Custom Dataset for sentiment analysis
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

# Configuration
out_dir = 'out-sentiment-gpt2'
eval_interval = 100
eval_iters = 10
log_interval = 50
batch_size = 8  # Smaller batch size for GPT-2 due to memory constraints
block_size = 512  # Matches your pre-processed data
dropout = 0.1
learning_rate = 3e-5  # Typical fine-tuning LR
max_iters = 1000
lr_decay_iters = 1000
min_lr = 3e-6
warmup_iters = 100
num_classes = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}", flush=True)

# DataLoaders
train_dataset = SentimentDataset('train')
val_dataset = SentimentDataset('val')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
print("DataLoaders initialized", flush=True)

# Initialize WANDB for experiment tracking
wandb.init(project="nanoGPT-sentiment-gpt2", config={
    "batch_size": batch_size,
    "block_size": block_size,
    "max_iters": max_iters,
    "learning_rate": learning_rate,
    "dropout": dropout,
    "num_classes": num_classes,
    "device": device,
    "model": "gpt2-finetuned"
})
print("WANDB initialized", flush=True)

# Load pre-trained GPT-2 model
model = GPT.from_pretrained('gpt2', override_args={'dropout': dropout})
model.to(device)
print(f"Model loaded and moved to {device}", flush=True)

# Freeze all layers except sentiment head and top transformer layer for fine-tuning
for name, param in model.named_parameters():
    # Unfreeze sentiment head and top layer (h.11) to adapt to sentiment task while preserving lower-layer features
    if 'sentiment_head' not in name and 'transformer.h.11' not in name:
        param.requires_grad = False
print("Froze all layers except sentiment_head and transformer.h.11", flush=True)

# Optimizer with separate learning rate for sentiment head
optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)
print("Optimizer initialized", flush=True)

# Learning rate scheduler
def get_lr(it):
    # Linear warmup for initial iterations
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Minimum LR after decay period
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay between warmup and max decay iterations
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
    # Update learning rate with scheduler
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Load batch
    data_start = time.time()
    try:
        x, y = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_loader)
        x, y = next(train_iterator)
    x, y = x.to(device), y.to(device)
    data_time = time.time() - data_start

    # Forward pass
    forward_start = time.time()
    sentiment_logits, _, sentiment_loss = model(x, labels=y)
    loss = sentiment_loss  # Focus on sentiment loss for fine-tuning
    forward_time = time.time() - forward_start

    # Backward pass
    backward_start = time.time()
    preds = torch.argmax(sentiment_logits, dim=-1)
    train_accuracy = (preds == y).float().mean().item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backward_time = time.time() - backward_start

    # Logging
    if iter_num % log_interval == 0 or iter_num == 0:
        total_time = time.time() - start_time
        print(f"Iter {iter_num}, Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, "
              f"Data: {data_time:.3f}s, Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s, Total: {total_time:.3f}s", flush=True)
        wandb.log({"train_loss": loss.item(), "train_accuracy": train_accuracy, "learning_rate": lr, "iteration": iter_num})

    # Validation
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

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'{out_dir}/best_model.pt')
                print(f"Saved best model at iteration {iter_num}", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered. Stopping training.", flush=True)
                    break
        model.train()

# Save final model
torch.save(model.state_dict(), f'{out_dir}/final_model.pt')
wandb.finish()
print("Training complete.", flush=True)