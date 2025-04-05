"""
Preprocesses customer service conversation data for character-level GPT sentiment analysis.
Outputs: train.bin, val.bin, test.bin (tokenized character sequences), and corresponding labels.
"""

import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# Define file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = script_dir
processed_dir = os.path.join(data_dir, 'processed')
os.makedirs(processed_dir, exist_ok=True)

train_csv = os.path.join(data_dir, 'train.csv')
test_csv = os.path.join(data_dir, 'test.csv')

# Load and process training data
train_df = pd.read_csv(train_csv)
train_df = train_df[['conversation', 'customer_sentiment']].dropna()
train_df = train_df.rename(columns={'customer_sentiment': 'sentiment'})

def clean_text(text):
    """Clean text using training data-derived rules."""
    if pd.isna(text):
        return ""
    text = re.sub(r'Agent:|Customer:', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9 !?\']', ' ', text)
    return text.lower()

train_df['conversation'] = train_df['conversation'].apply(clean_text)

# Define sentiment mapping
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
train_df['sentiment'] = train_df['sentiment'].str.lower().map(sentiment_map)

# Process test data
test_df = pd.read_csv(test_csv)
test_df = test_df[['conversation', 'customer_sentiment']].dropna()
test_df = test_df.rename(columns={'customer_sentiment': 'sentiment'})
test_df['conversation'] = test_df['conversation'].apply(clean_text)
test_df['sentiment'] = test_df['sentiment'].str.lower().map(sentiment_map)

# Validate distributions
print("Train class distribution:")
print(train_df['sentiment'].value_counts())
print("\nTest class distribution:")
print(test_df['sentiment'].value_counts())

# Split training data into train/val
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['conversation'],
    train_df['sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=train_df['sentiment']
)

# Character-level vocabulary
all_text = ''.join(train_texts) + ''.join(val_texts) + ''.join(test_df['conversation'])
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
print(f"Vocabulary size (unique characters): {vocab_size}")
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Save vocabulary for reference
with open(os.path.join(processed_dir, 'vocab.pkl'), 'wb') as f:
    pickle.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char, 'vocab_size': vocab_size}, f)

# Tokenize datasets at character level
def tokenize_to_bin_char_level(texts, labels, max_length=512):
    """Convert texts to character-level token arrays with padding/truncation."""
    tokens = [[char_to_idx[ch] for ch in text[:max_length]] for text in texts]
    padded = [t + [0] * (max_length - len(t)) for t in tokens]
    return np.array(padded, dtype=np.uint16), np.array(labels, dtype=np.uint8)

# Process all datasets
train_tokens, train_labels = tokenize_to_bin_char_level(train_texts, train_labels)
val_tokens, val_labels = tokenize_to_bin_char_level(val_texts, val_labels)
test_tokens, test_labels = tokenize_to_bin_char_level(test_df['conversation'], test_df['sentiment'])

# Save processed data
train_tokens.tofile(os.path.join(processed_dir, 'train.bin'))
val_tokens.tofile(os.path.join(processed_dir, 'val.bin'))
test_tokens.tofile(os.path.join(processed_dir, 'test.bin'))

with open(os.path.join(processed_dir, 'train_labels.pkl'), 'wb') as f:
    pickle.dump(train_labels, f)
with open(os.path.join(processed_dir, 'val_labels.pkl'), 'wb') as f:
    pickle.dump(val_labels, f)
with open(os.path.join(processed_dir, 'test_labels.pkl'), 'wb') as f:
    pickle.dump(test_labels, f)

print("\nData processing complete:")
print(f"- Training samples: {len(train_tokens)}")
print(f"- Validation samples: {len(val_tokens)}")
print(f"- Test samples: {len(test_tokens)}")
print(f"Data saved to: {processed_dir}")

# Verify saved files
print("\nVerifying saved files:")
saved_files = [
    ('train.bin', os.path.getsize(os.path.join(processed_dir, 'train.bin'))),
    ('val.bin', os.path.getsize(os.path.join(processed_dir, 'val.bin'))),
    ('test.bin', os.path.getsize(os.path.join(processed_dir, 'test.bin'))),
    ('train_labels.pkl', os.path.getsize(os.path.join(processed_dir, 'train_labels.pkl'))),
    ('val_labels.pkl', os.path.getsize(os.path.join(processed_dir, 'val_labels.pkl'))),
    ('test_labels.pkl', os.path.getsize(os.path.join(processed_dir, 'test_labels.pkl')))
]
for filename, size in saved_files:
    print(f"- {filename}: {size} bytes")