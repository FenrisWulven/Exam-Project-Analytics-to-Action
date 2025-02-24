import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NewsDataset(Dataset):
    """Dataset for batch processing of news articles"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if pd.isna(text):
            return {'input_ids': torch.zeros(1), 'attention_mask': torch.zeros(1)}
        
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

def load_tf():
    """Load TF model and tokenizer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = model.to(device)
    return tokenizer, model, device

def chunk_text(text, max_length=512):
    """Split text into chunks of approximately max_length words"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1  # +1 for space
        if current_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_batch(batch_texts, tokenizer, model, device, max_length=512):
    """Process a batch of texts, handling long texts with chunks"""
    batch_results = []
    
    # Handle empty/NA texts first
    for text in batch_texts:
        if pd.isna(text):
            batch_results.append({
                'sentiment_label': 'neutral',
                'sentiment_score': 0.0,
                'positive_score': np.nan,
                'negative_score': np.nan,
                'neutral_score': np.nan
            })
            continue
        
        # Split text into chunks and process each chunk
        chunks = chunk_text(text, max_length)
        chunk_scores = []
        
        for chunk in chunks:
            # Tokenize chunk
            tokens = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # Get scores for chunk
            with torch.no_grad():
                outputs = model(**tokens)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                chunk_scores.append(scores.cpu().numpy())
        
        # Average scores across chunks
        if chunk_scores:
            avg_scores = np.mean(chunk_scores, axis=0)[0]
            sentiment_score = float(avg_scores[0] - avg_scores[1])
            
            if sentiment_score > 0.1:
                label = 'positive'
            elif sentiment_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            batch_results.append({
                'sentiment_label': label,
                'sentiment_score': sentiment_score,
                'positive_score': float(avg_scores[0]),
                'negative_score': float(avg_scores[1]),
                'neutral_score': float(avg_scores[2])
            })
    
    return batch_results

def analyze_sentiments(csv_path, sample_size=None, batch_size=1024):
    """Analyze sentiments for all articles in the CSV file using batched processing"""
    # Load the data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    if sample_size:
        print(f"Taking sample of {sample_size} entries...")
        df = df.head(sample_size)
    
    # Load TF model
    print("Loading TF model...")
    tokenizer, model, device = load_tf()
    model.eval()  # Set model to evaluation mode
    
    # Process in batches
    print("Analyzing sentiments...")
    results = []
    
    # Calculate optimal batch size based on GPU memory
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
    suggested_batch_size = min(batch_size, total_gpu_mem // (2 * 1024 * 1024 * 1024) * 512)  # Rough estimate
    print(f"Using batch size: {suggested_batch_size}")
    
    # Process data in batches
    for i in tqdm(range(0, len(df), suggested_batch_size), desc="Processing batches"):
        batch_texts = df['Body'].iloc[i:i + suggested_batch_size].tolist()
        batch_results = process_batch(batch_texts, tokenizer, model, device)
        results.extend(batch_results)
    
    # Add results to dataframe
    df['sentiment_label'] = [r['sentiment_label'] for r in results]
    df['sentiment_score'] = [r['sentiment_score'] for r in results]
    df['positive_score'] = [r['positive_score'] for r in results]
    df['negative_score'] = [r['negative_score'] for r in results]
    df['neutral_score'] = [r['neutral_score'] for r in results]
    
    # Save results
    output_path = csv_path.replace('.csv', '_with_sentiment.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Print summary statistics
    print("\nSentiment Distribution:")
    print(df['sentiment_label'].value_counts())
    
    return df

if __name__ == "__main__":
    csv_path = "cleaned_combined_news_with_bodies.csv"
    df = analyze_sentiments(csv_path, sample_size=None, batch_size=512)
