import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    """Configuration class for GPT model"""
    def __init__(self, **kwargs):
        # Default parameters (modify through kwargs)
        self.n_layer = kwargs.get('n_layer', 6)
        self.n_head = kwargs.get('n_head', 6)
        self.n_embd = kwargs.get('n_embd', 384)
        self.block_size = kwargs.get('block_size', 128)
        self.dropout = kwargs.get('dropout', 0.2)
        self.num_classes = kwargs.get('num_classes', 3)  # For sentiment classification
        self.vocab_size = kwargs.get('vocab_size', 50257)  # GPT-2 vocab size
        
        # Set all other provided kwargs as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    """Multi-head self attention with causal masking"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_size)
        
        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence, embedding
        
        # Project to Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Combine values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """Two-layer feedforward network with GELU activation"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block: attention + MLP with residual connections"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.ln_1(x))
        # Feedforward
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """GPT model for sequence classification"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Classification head
        self.classifier = nn.Linear(config.n_embd, config.num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        print(f"Model initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        # Positional encoding
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward pass
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Classification using last token
        logits = self.classifier(x[:, -1, :])  # (batch, num_classes)
        
        # Loss calculation if targets provided
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        Returns AdamW optimizer with weight decay
        """
        decay_params = [p for n, p in self.named_parameters() if p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)