import pandas as pd
from sklearn.model_selection import train_test_split
import re

def clean_text(text):
    """Clean conversation text by removing agent/customer tags and extra spaces."""
    text = re.sub(r'Agent:|Customer:', '', text)  # Remove speaker tags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and newlines
    return text.lower()  # Convert to lowercase

def preprocess_data(train_path, test_path):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Select relevant columns
    train_df = train_df[['conversation', 'customer_sentiment']]
    test_df = test_df[['conversation', 'customer_sentiment']]
    
    # Clean text
    train_df['conversation'] = train_df['conversation'].apply(clean_text)
    test_df['conversation'] = test_df['conversation'].apply(clean_text)
    
    # Normalize sentiment labels
    train_df['customer_sentiment'] = train_df['customer_sentiment'].str.lower()
    test_df['customer_sentiment'] = test_df['customer_sentiment'].str.lower()
    
    # Split training set into train and validation
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['customer_sentiment'])
    
    # Save processed data
    train_data.to_csv('C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725-transformer-sentiment-analysis/data/processed/train_processed.csv', index=False)
    val_data.to_csv('C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725-transformer-sentiment-analysis/data/processed/val_processed.csv', index=False)
    test_df.to_csv('C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725-transformer-sentiment-analysis/data/processed/test_processed.csv', index=False)
    
    print("Preprocessing complete. Processed files saved.")
    
# Run preprocessing
preprocess_data("C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725-transformer-sentiment-analysis/data/raw/train.csv", "C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725-transformer-sentiment-analysis/data/raw/test.csv")
