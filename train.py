import os
import time
import wandb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig

# Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, split):
        with open(f"data/sentiment/processed/{split}_text.txt") as f:
            self.texts = [line.strip() for line in f]
        with open(f"data/sentiment/processed/{split}_labels.txt") as f:
            self.labels = [int(line.strip()) for line in f]
        
        # Tokenization stub (replace with your tokenizer)
        self.vocab = {chr(i): i for i in range(256)}
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Simple character-level tokenization
        tokens = [self.stoi[ch] for ch in text[:self.config.block_size]]
        return torch.tensor(tokens, dtype=torch.long), self.labels[idx]

def get_batch(dataset, batch_size):
    indices = torch.randint(len(dataset), (batch_size,))
    texts, labels = zip(*[dataset[i] for i in indices])
    return torch.stack(texts), torch.tensor(labels)

def evaluate(model, dataset):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            texts, labels = get_batch(dataset, batch_size)
            _, loss = model(texts, labels)
            losses.append(loss.item())
            accs.append((logits.argmax(-1) == labels).float().mean().item())
    model.train()
    return np.mean(losses), np.mean(accs)

def main():
    from config.train_sentiment import config
    
    # Initialize WandB
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        notes=config.wandb_notes,
        config=config
    )
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("val/loss", summary="min")

    # Data
    train_dataset = SentimentDataset("train")
    val_dataset = SentimentDataset("val")

    # Model
    model = GPT(GPTConfig(**config))
    model.to(config.device)
    wandb.watch(model, log="all", log_freq=config.log_interval)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    
    for iter in range(config.max_iters):
        # Training step
        texts, labels = get_batch(train_dataset, config.batch_size)
        logits, loss = model(texts, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log training metrics
        train_losses.append(loss.item())
        if iter % config.log_interval == 0:
            avg_loss = np.mean(train_losses[-config.loss_window_size:])
            wandb.log({
                "iter": iter,
                "train/loss": loss.item(),
                "train/loss_smooth": avg_loss,
                "lr": optimizer.param_groups[0]['lr']
            }, commit=False)

        # Evaluation
        if iter % config.eval_interval == 0:
            val_loss, val_acc = evaluate(model, val_dataset)
            wandb.log({
                "val/loss": val_loss,
                "val/acc": val_acc
            }, commit=True)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")
                wandb.save("best_model.pt")

    # Final save
    torch.save(model.state_dict(), "final_model.pt")
    wandb.finish()

if __name__ == "__main__":
    main()