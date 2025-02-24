import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Create plots directory if it doesn't exist
plots_dir = './plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load the CSV file
df = pd.read_csv('./fata2025/datasets/WillandAgency/European_data_2000.csv')

# Quick look at the data
print(df.columns)
print(df.head())
print(df.info())
print(df.describe())

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Function to save plot with consistent settings
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def load_model():
    """Load movie plot sentiment analysis model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Using model trained specifically on movie plot summaries
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    # Alternative options:
    # model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    # model_name = "microsoft/empathetic-dial-roberta-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    return tokenizer, model, device

def process_batch(texts, tokenizer, model, device, batch_size=8):
    """Process a batch of texts for sentiment analysis"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Convert predictions to results
        batch_predictions = predictions.cpu().numpy()
        for pred in batch_predictions:
            # Emotion labels for j-hartmann/emotion-english-distilroberta-base
            emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            emotion_idx = pred.argmax()
            emotion = emotions[emotion_idx]
            
            # Map emotions to sentiment
            sentiment_map = {
                'joy': 'positive',
                'surprise': 'positive',
                'neutral': 'neutral',
                'anger': 'negative',
                'disgust': 'negative',
                'fear': 'negative',
                'sadness': 'negative'
            }
            
            results.append({
                'sentiment_label': sentiment_map[emotion],
                'sentiment_score': float(pred[emotion_idx]),
                'emotion': emotion,
                'emotion_scores': {e: float(s) for e, s in zip(emotions, pred)}
            })
    
    return results

# Add sentiment analysis for plot descriptions
print("Performing sentiment analysis on plot descriptions...")
tokenizer, model, device = load_model()
model.eval()

# Process plots in batches
plots = df['plotLong'].dropna().tolist()
results = process_batch(plots, tokenizer, model, device)

# Add results to dataframe
df['plot_sentiment_label'] = pd.NA
df['plot_sentiment_score'] = pd.NA
df['emotion'] = pd.NA

# Only fill in results for non-null plots
non_null_idx = df['plotLong'].dropna().index
for idx, result in zip(non_null_idx, results):
    df.loc[idx, 'plot_sentiment_label'] = result['sentiment_label']
    df.loc[idx, 'plot_sentiment_score'] = result['sentiment_score']
    df.loc[idx, 'emotion'] = result['emotion']

# Print summary of sentiment analysis
print("\nPlot Sentiment Distribution:")
print(df['plot_sentiment_label'].value_counts())
print("\nSentiment Score Statistics:")
print(df['plot_sentiment_score'].describe())

# Create visualization for sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='plot_sentiment_label')
plt.title('Distribution of Plot Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
save_plot('plot_sentiment_dist.png')

# Create box plot of IMDb ratings by sentiment
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='plot_sentiment_label', y='imdbRating')
plt.title('IMDb Ratings by Plot Sentiment')
plt.xlabel('Plot Sentiment')
plt.ylabel('IMDb Rating')
save_plot('ratings_by_sentiment.png')

# Scatter plot of sentiment score vs IMDb rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='plot_sentiment_score', y='imdbRating')
plt.title('Plot Sentiment Score vs IMDb Rating')
plt.xlabel('Sentiment Score')
plt.ylabel('IMDb Rating')
save_plot('sentiment_score_vs_rating.png')

# Existing plots with tight_layout
# Distribution of Release Years
plt.figure(figsize=(10, 6))
sns.histplot(df['releaseYear'], bins=20, kde=True)
plt.title('Distribution of Release Years')
plt.xlabel('Year')
plt.ylabel('Count')
save_plot('release_year_dist.png')

# Distribution of Runtime Minutes
plt.figure(figsize=(8, 6))
sns.histplot(df['runtimeMinutes'], bins=20, kde=True)
plt.title('Distribution of Runtime Minutes')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Count')
save_plot('runtime_dist.png')

# Distribution of IMDb Ratings
plt.figure(figsize=(8, 6))
sns.histplot(df['imdbRating'], bins=20, kde=True)
plt.title('Distribution of IMDb Ratings')
plt.xlabel('IMDb Rating')
plt.ylabel('Count')
save_plot('imdb_rating_dist.png')

# IMDb Rating vs. Number of Votes (using log scale for votes)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='imdbRating', y='numberOfVotes')
plt.yscale('log')
plt.title('IMDb Rating vs. Number of Votes (Log Scale)')
plt.xlabel('IMDb Rating')
plt.ylabel('Number of Votes (log scale)')
save_plot('rating_vs_votes.png')

# Count plot for Title Types
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='titleType')
plt.title('Count of Title Types')
plt.xlabel('Title Type')
plt.ylabel('Count')
save_plot('title_types.png')

# Count plot for Main Countries
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='mainCountry', order=df['mainCountry'].value_counts().index)
plt.title('Movies by Main Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
save_plot('movies_by_country.png')

# Pairplot for key numerical features
pairplot = sns.pairplot(df[['releaseYear', 'runtimeMinutes', 'imdbRating', 'numberOfVotes']])
pairplot.savefig(os.path.join(plots_dir, 'feature_pairplot.png'))

# Correlation heatmap for numerical features
plt.figure(figsize=(8, 6))
corr = df[['releaseYear', 'runtimeMinutes', 'imdbRating', 'numberOfVotes']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
save_plot('correlation_heatmap.png')

# New plots for additional insights:

# Language Distribution
plt.figure(figsize=(12, 6))
lang_counts = df['firstLanguage'].value_counts().head(15)
sns.barplot(x=lang_counts.index, y=lang_counts.values)
plt.title('Top 15 First Languages in Movies')
plt.xlabel('Language')
plt.ylabel('Count')
plt.xticks(rotation=45)
save_plot('top_languages.png')

# Adult Content Analysis
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='isAdult')
plt.title('Distribution of Adult vs Non-Adult Content')
plt.xlabel('Is Adult Content')
plt.ylabel('Count')
save_plot('adult_content_dist.png')

# Rating Distribution by Decade
df['decade'] = (df['releaseYear'] // 10) * 10
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='decade', y='imdbRating')
plt.title('IMDb Ratings Distribution by Decade')
plt.xlabel('Decade')
plt.ylabel('IMDb Rating')
plt.xticks(rotation=45)
save_plot('ratings_by_decade.png')

# Genre Analysis
# First, create a list of all genres
genres = []
for genre_list in df['genres'].dropna():
    genres.extend(genre_list.split(', '))
genre_counts = pd.Series(genres).value_counts().head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Top 15 Movie Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
save_plot('top_genres.png')

# Runtime vs Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='runtimeMinutes', y='imdbRating', alpha=0.5)
plt.title('Movie Runtime vs IMDb Rating')
plt.xlabel('Runtime (minutes)')
plt.ylabel('IMDb Rating')
save_plot('runtime_vs_rating.png')

# Rating vs Votes by Type
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='imdbRating', y='numberOfVotes', hue='titleType', alpha=0.5)
plt.yscale('log')
plt.title('Rating vs Votes by Title Type')
plt.xlabel('IMDb Rating')
plt.ylabel('Number of Votes (log scale)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
save_plot('rating_votes_by_type.png')

# Average Rating by Country (top 20 countries)
top_countries = df.groupby('mainCountry')['imdbRating'].agg(['mean', 'count']).sort_values('count', ascending=False).head(20)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries.index, y=top_countries['mean'])
plt.title('Average Rating by Country (Top 20 by Movie Count)')
plt.xlabel('Country')
plt.ylabel('Average IMDb Rating')
plt.xticks(rotation=45)
save_plot('avg_rating_by_country.png')

# Production Companies Analysis
# Create a list of all production companies
companies = []
for company_list in df['production'].dropna():
    companies.extend(company_list.split(', '))
company_counts = pd.Series(companies).value_counts().head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x=company_counts.index, y=company_counts.values)
plt.title('Top 15 Production Companies')
plt.xlabel('Company')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
save_plot('top_production_companies.png')

# Add additional visualization for emotions
plt.figure(figsize=(12, 6))
emotion_counts = df['emotion'].value_counts()
sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
plt.title('Distribution of Plot Emotions')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
save_plot('plot_emotions_dist.png')
