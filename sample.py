import torch
from model import GPT, GPTConfig
import tiktoken
import numpy as np
import pickle
import os
import argparse

parser = argparse.ArgumentParser(description="Run sentiment prediction on test data.")
parser.add_argument('--out_dir', type=str, default='out-sentiment', help='Directory where model checkpoint is stored')
parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Model checkpoint file')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, 'data', 'sentiment', 'processed')

config = GPTConfig(
    block_size=512,
    vocab_size=50304,
    n_layer=5,
    n_head=4,
    n_embd=384,
    dropout=0.3,
    num_classes=3
)

model_path = os.path.join(args.out_dir, args.checkpoint)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Checkpoint not found at {model_path}")
model = GPT(config)
model.load_state_dict(torch.load(model_path))
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Model loaded from {model_path} and moved to {device}", flush=True)

enc = tiktoken.get_encoding("gpt2")

test_data_path = os.path.join(processed_dir, 'test.bin')
test_labels_path = os.path.join(processed_dir, 'test_labels.pkl')
if not os.path.exists(test_data_path) or not os.path.exists(test_labels_path):
    raise FileNotFoundError(f"Test data not found at {test_data_path} or labels at {test_labels_path}")
test_data = np.memmap(test_data_path, dtype=np.uint16, mode='r').reshape(-1, 512)
with open(test_labels_path, 'rb') as f:
    test_labels = pickle.load(f)
print(f"Loaded test dataset: {len(test_labels)} samples", flush=True)

sentiment_labels = ['Negative', 'Neutral', 'Positive']
num_samples = len(test_labels)
correct = 0

for sample_idx in range(num_samples):
    tokens = torch.from_numpy(test_data[sample_idx:sample_idx+1].astype(np.int64)).to(device)
    with torch.no_grad():
        logits, _ = model(tokens)
        logits = logits / 2  # Temperature scaling to soften probabilities
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

accuracy = correct / num_samples
print(f"\nSummary: {correct}/{num_samples} correct, Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")