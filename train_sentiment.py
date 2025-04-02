import os
import time
import math
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model_sentiment import GPT, GPTConfig
import wandb

# Configuration imports
from config import train_sentiment as config
from wandb_config import WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY

# System setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=config.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# WandB initialization
if master_process and config.wandb_log:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=config.wandb_run_name,
        notes=config.wandb_notes,
        config=config.__dict__
    )
    wandb.watch_called = False

# Set random seeds
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in config.device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loading
def get_batch(split):
    data = np.memmap(os.path.join('data/processed', f'{split}.bin'), dtype=np.uint16, mode='r')
    labels = np.load(os.path.join('data/processed', f'{split}_labels.npy'))
    ix = torch.randint(len(data) // config.block_size - 1, (config.batch_size,))
    x = torch.stack([torch.from_numpy(data[i*config.block_size:(i+1)*config.block_size].astype(np.int64)) for i in ix])
    y = torch.from_numpy(labels[ix.numpy()]).long()
    if device_type == 'cuda':
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y

# Model initialization
model_args = dict(
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
    block_size=config.block_size,
    dropout=config.dropout,
    num_classes=config.num_classes,
    vocab_size=50257
)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(config.device)

# Optimizer setup
optimizer = model.configure_optimizers(
    weight_decay=config.weight_decay,
    learning_rate=config.learning_rate,
    betas=(config.beta1, config.beta2),
    device_type=device_type
)
scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

# Distributed training
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Evaluation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses, accs = [], []
        for _ in range(config.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            acc = (logits.argmax(dim=-1) == Y).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())
        out[f'{split}/loss'] = torch.tensor(losses).mean()
        out[f'{split}/acc'] = torch.tensor(accs).mean()
    model.train()
    return out

# Training loop
best_val_loss = float('inf')
X, Y = get_batch('train')
t0 = time.time()

for iter_num in range(config.max_iters):
    # Learning rate decay
    lr = config.learning_rate
    if iter_num < config.warmup_iters:
        lr *= iter_num / config.warmup_iters
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation and logging
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"Step {iter_num}: Train loss {losses['train/loss']:.4f}, Acc {losses['train/acc']:.2f} | Val loss {losses['val/loss']:.4f}, Acc {losses['val/acc']:.2f}")
        
        if config.wandb_log and master_process:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train/loss'],
                "val/loss": losses['val/loss'],
                "train/acc": losses['train/acc'],
                "val/acc": losses['val/acc'],
                "lr": lr,
                "time_per_iter": (time.time()-t0)*1000,
            })
            
        if losses['val/loss'] < best_val_loss:
            best_val_loss = losses['val/loss']
            torch.save(model.state_dict(), os.path.join(config.out_dir, 'best_model.pt'))
            if config.wandb_log and master_process:
                artifact = wandb.Artifact('best_model', type='model')
                artifact.add_file(os.path.join(config.out_dir, 'best_model.pt'))
                wandb.log_artifact(artifact)

    # Training step
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / config.gradient_accumulation_steps
        scaler.scale(loss).backward()
        X, Y = get_batch('train')

    # Gradient clipping
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Log training progress
    if iter_num % config.log_interval == 0 and master_process:
        dt = time.time() - t0
        t0 = time.time()
        lossf = loss.item() * config.gradient_accumulation_steps
        print(f"Iter {iter_num}: Loss {lossf:.4f}, Time {dt*1000:.2f}ms")
        
        if config.wandb_log and master_process:
            wandb.log({
                "iter": iter_num,
                "train/loss_step": lossf,
                "time_per_iter": dt*1000,
            })

# Cleanup
if ddp:
    destroy_process_group()

if config.wandb_log and master_process:
    wandb.finish()
