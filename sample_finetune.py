"""
Sample script to run inference on the test set using the fine-tuned nanoGPT-style GPT-2 model.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pickle
import os
import argparse
import tiktoken
import math

# nanoGPT-style GPT model (copied from train_finetune.py)
class LayerNorm(nn.Module):
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

class GPTConfig:
    def __init__(self, block_size=512, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, dropout=0.1, bias=True, num_classes=3):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
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
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.sentiment_head = nn.Linear(config.n_embd, config.num_classes)
        self.transformer.wte.weight = self.lm_head.weight

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

        sentiment_logits = self.sentiment_head(x[:, -1, :])
        loss = None
        if labels is not None:
            loss = F.cross_entropy(sentiment_logits, labels)
        return sentiment_logits, loss

# Argument parser
parser = argparse.ArgumentParser(description="Run sentiment prediction on test data.")
parser.add_argument('--out_dir', type=str, default='out-sentiment-gpt2', help='Directory where model checkpoint is stored')
parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Model checkpoint file')
args = parser.parse_args()

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, 'data', 'sentiment', 'processed')

# Load model
model_path = os.path.join(args.out_dir, args.checkpoint)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Checkpoint not found at {model_path}")
config = GPTConfig(
    block_size=1024,  # Match the training script
    vocab_size=50257,  # Match the training script
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    bias=True,
    num_classes=3
)
model = GPT(config)
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()
print(f"Model loaded from {model_path} and moved to {device}", flush=True)

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load test data
test_data_path = os.path.join(processed_dir, 'test.bin')
test_labels_path = os.path.join(processed_dir, 'test_labels.pkl')
if not os.path.exists(test_data_path) or not os.path.exists(test_labels_path):
    raise FileNotFoundError(f"Test data not found at {test_data_path} or labels at {test_labels_path}")
test_data = np.memmap(test_data_path, dtype=np.uint16, mode='r').reshape(-1, 512)
with open(test_labels_path, 'rb') as f:
    test_labels = pickle.load(f)
print(f"Loaded test dataset: {len(test_labels)} samples", flush=True)

# Sentiment labels
sentiment_labels = ['Negative', 'Neutral', 'Positive']
num_samples = len(test_labels)
correct = 0

# Inference loop
for sample_idx in range(num_samples):
    tokens = torch.from_numpy(test_data[sample_idx:sample_idx+1].astype(np.int64)).to(device)
    with torch.no_grad():
        logits, _ = model(tokens)
        logits = logits / 2  # Temperature scaling
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    
    true_label = test_labels[sample_idx]
    text = enc.decode([t for t in test_data[sample_idx] if t != 0])
    
    print(f"\nSample {sample_idx + 1}:")
    print(f"Text: {text[:100]}...")
    print(f"Predicted Sentiment: {sentiment_labels[pred]} (Probabilities: {probs[0].tolist()})")
    print(f"True Sentiment: {sentiment_labels[true_label]}")
    if pred == true_label:
        correct += 1

# Summary
accuracy = correct / num_samples
print(f"\nSummary: {correct}/{num_samples} correct, Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")