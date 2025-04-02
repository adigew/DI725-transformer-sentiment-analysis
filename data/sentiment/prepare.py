import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
import tiktoken

# Define file paths
data_dir = os.path.dirname(__file__)
train_csv = os.path.join(data_dir, 'train.csv')
test_csv = os.path.join(data_dir, 'test.csv')

# Load and process training data first
train_df = pd.read_csv(train_csv)
# Select and rename sentiment column
train_df = train_df[['conversation', 'customer_sentiment']].dropna()
train_df = train_df.rename(columns={'customer_sentiment': 'sentiment'})

def clean_text(text):
    """Clean text using only training data-derived rules"""
    if pd.isna(text):
        return ""
    text = re.sub(r'Agent:|Customer:', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9 !?\']', ' ', text)
    return text.lower()

# Learn cleaning from training data only
train_df['conversation'] = train_df['conversation'].apply(clean_text)

# Define label mapping based on training data
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
train_df['sentiment'] = train_df['sentiment'].str.lower().map(sentiment_map)

# Process test data separately using training-derived transformations
test_df = pd.read_csv(test_csv)
# Select and rename sentiment column
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

# Initialize tokenizer (using training data only)
enc = tiktoken.get_encoding("gpt2")

# Tokenize datasets and convert to numpy arrays
def tokenize_to_bin(texts, labels, max_length=512):
    """Convert texts to token arrays with padding/truncation"""
    tokens = [enc.encode_ordinary(text)[:max_length] for text in texts]
    padded = [t + [0]*(max_length - len(t)) for t in tokens]
    return np.array(padded, dtype=np.uint16), np.array(labels, dtype=np.uint8)

# Process all datasets
train_tokens, train_labels = tokenize_to_bin(train_texts, train_labels)
val_tokens, val_labels = tokenize_to_bin(val_texts, val_labels)
test_tokens, test_labels = tokenize_to_bin(test_df['conversation'], test_df['sentiment'])

# Save processed data in original bin format
processed_dir = os.path.join(data_dir, 'processed')
os.makedirs(processed_dir, exist_ok=True)

# Save token arrays as .bin files
train_tokens.tofile(os.path.join(processed_dir, 'train.bin'))
val_tokens.tofile(os.path.join(processed_dir, 'val.bin'))
test_tokens.tofile(os.path.join(processed_dir, 'test.bin'))

# Save labels as .npy files
np.save(os.path.join(processed_dir, 'train_labels.npy'), train_labels)
np.save(os.path.join(processed_dir, 'val_labels.npy'), val_labels)
np.save(os.path.join(processed_dir, 'test_labels.npy'), test_labels)

print("\nData processing complete:")
print(f"- Training samples: {len(train_tokens)}")
print(f"- Validation samples: {len(val_tokens)}")
print(f"- Test samples: {len(test_tokens)}")
print(f"Data saved to: {processed_dir}")