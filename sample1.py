import torch
from model import GPT, GPTConfig
import tiktoken
import numpy as np
import pickle
import os

# Define base directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, 'data', 'sentiment', 'processed')

# Model config
config = GPTConfig(
    block_size=512,
    vocab_size=50304,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    num_classes=3
)

# Load model
model = GPT(config)
model.load_state_dict(torch.load('out-sentiment/model.pt'))
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load test data for example
test_data = np.memmap(os.path.join(processed_dir, 'test.bin'), dtype=np.uint16, mode='r').reshape(-1, 512)
with open(os.path.join(processed_dir, 'test_labels.pkl'), 'rb') as f:
    test_labels = pickle.load(f)

# Predict sentiment for a test sample
sample_idx = 0  # Change this to test different samples
tokens = torch.from_numpy(test_data[sample_idx:sample_idx+1].astype(np.int64)).to(device)
logits, _ = model(tokens)
probs = torch.softmax(logits, dim=-1)
pred = torch.argmax(probs, dim=-1).item()
true_label = test_labels[sample_idx]
sentiment_labels = ['Negative', 'Neutral', 'Positive']  # Matches your sentiment_map
text = enc.decode([t for t in test_data[sample_idx] if t != 0])  # Remove padding

print(f"Text: {text}")
print(f"Predicted Sentiment: {sentiment_labels[pred]} (Probabilities: {probs[0].tolist()})")
print(f"True Sentiment: {sentiment_labels[true_label]}")