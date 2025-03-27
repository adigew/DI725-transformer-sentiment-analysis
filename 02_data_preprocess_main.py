import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Initialize Tokenizer (Change model name as needed, e.g., 'bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def clean_text(text):
    """Clean conversation text by removing agent/customer tags, punctuation, and extra spaces."""
    if pd.isna(text):  # Handle missing values
        return ""
    
    text = re.sub(r'Agent:|Customer:', '', text)  # Remove speaker tags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and newlines
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove punctuation
    return text.lower()  # Convert to lowercase

def preprocess_data(train_path, test_path, output_dir="data/processed"):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Select relevant columns and handle missing values
    train_df = train_df[['conversation', 'customer_sentiment']].dropna()
    test_df = test_df[['conversation', 'customer_sentiment']].dropna()

    # Clean text
    train_df['conversation'] = train_df['conversation'].apply(clean_text)
    test_df['conversation'] = test_df['conversation'].apply(clean_text)

    # Normalize sentiment labels
    train_df['customer_sentiment'] = train_df['customer_sentiment'].str.lower()
    test_df['customer_sentiment'] = test_df['customer_sentiment'].str.lower()

    # Check class distribution before splitting
    print("Class distribution in training set:")
    print(train_df['customer_sentiment'].value_counts(normalize=True))

    # Split training set into train and validation (80/20 split)
    train_data, val_data = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['customer_sentiment']
    )

    # Tokenization
    train_data['tokenized'] = train_data['conversation'].apply(lambda x: tokenizer.encode(x, truncation=True, padding="max_length"))
    val_data['tokenized'] = val_data['conversation'].apply(lambda x: tokenizer.encode(x, truncation=True, padding="max_length"))
    test_df['tokenized'] = test_df['conversation'].apply(lambda x: tokenizer.encode(x, truncation=True, padding="max_length"))

    # Save processed data
    train_data.to_csv('C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725-transformer-sentiment-analysis/data/processed/train_processed.csv', index=False)
    val_data.to_csv('C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725-transformer-sentiment-analysis/data/processed/val_processed.csv', index=False)
    test_df.to_csv('C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725-transformer-sentiment-analysis/data/processed/test_processed.csv', index=False)
    
    print("Preprocessing complete. Processed files saved.")
    
# Run preprocessing
preprocess_data("C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725-transformer-sentiment-analysis/data/raw/train.csv", "C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725-transformer-sentiment-analysis/data/raw/test.csv")


