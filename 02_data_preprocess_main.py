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

# Load dataset
train_df = pd.read_csv(train_csv)

# Keep only necessary columns and drop missing values
train_df = train_df[['conversation', 'sentiment']].dropna()

def clean_text(text):
    """Clean conversation text by removing agent/customer tags, punctuation, and extra spaces."""
    if pd.isna(text):
        return ""
    text = re.sub(r'Agent:|Customer:', '', text)  # Remove speaker tags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and newlines
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove punctuation
    return text.lower()  # Convert to lowercase

# Apply text cleaning
train_df['conversation'] = train_df['conversation'].apply(clean_text)

# Encode sentiment labels (0: Negative, 1: Neutral, 2: Positive)
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
train_df['sentiment'] = train_df['sentiment'].str.lower().map(sentiment_map)

# Split into train and validation sets with stratification
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['conversation'], train_df['sentiment'], test_size=0.2, random_state=42, stratify=train_df['sentiment']
)

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Tokenize text data
train_tokens = [enc.encode_ordinary(text) for text in train_texts]
val_tokens = [enc.encode_ordinary(text) for text in val_texts]

# Convert to NumPy arrays
train_tokens = np.array([np.array(tokens, dtype=np.uint16) for tokens in train_tokens], dtype=object)
val_tokens = np.array([np.array(tokens, dtype=np.uint16) for tokens in val_tokens], dtype=object)
train_labels = np.array(train_labels, dtype=np.uint8)
val_labels = np.array(val_labels, dtype=np.uint8)

# Save processed data
processed_dir = os.path.join(data_dir, 'processed')
os.makedirs(processed_dir, exist_ok=True)

np.save(os.path.join(processed_dir, 'train_tokens.npy'), train_tokens)
np.save(os.path.join(processed_dir, 'val_tokens.npy'), val_tokens)
np.save(os.path.join(processed_dir, 'train_labels.npy'), train_labels)
np.save(os.path.join(processed_dir, 'val_labels.npy'), val_labels)

print("Preprocessing complete. Tokenized data and labels saved in 'processed' directory.")
