import time

# Base configuration
out_dir = 'out-sentiment'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

# WandB settings
wandb_log = True
wandb_run_name = f"gpt-sentiment-{time.strftime('%Y%m%d-%H%M%S')}"
wandb_notes = "GPT model for customer sentiment analysis"

# Data parameters
dataset = 'processed'
batch_size = 8
block_size = 512

# Model architecture
n_layer = 6
n_head = 6
n_embd = 512
dropout = 0.2
num_classes = 3

# Training parameters
learning_rate = 2e-5
max_iters = 4000
min_lr = 1e-6
warmup_iters = 200
weight_decay = 1e-2
grad_clip = 1.0

# System config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile = True
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
backend = 'nccl' if torch.cuda.is_available() else 'gloo'
