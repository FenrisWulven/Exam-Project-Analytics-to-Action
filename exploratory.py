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

# Create genre-emotion mapping for typical expectations
genre_emotion_mapping = {
    'Horror': ['fear', 'disgust'],
    'Comedy': ['joy'],
    'Drama': ['sadness', 'neutral'],
    'Romance': ['joy'],
    'Thriller': ['fear', 'surprise'],
    'Action': ['surprise', 'anger'],
    'Adventure': ['surprise', 'joy'],
    'Crime': ['anger', 'fear'],
    'Mystery': ['surprise', 'fear'],
    'Sci-Fi': ['surprise', 'fear']
}

def calculate_genre_emotion_match(row):
    """Calculate how well the plot emotions match the expected genre emotions"""
    if pd.isna(row['genres']) or pd.isna(row['emotion']):
        return None
    
    genres = row['genres'].split(', ')
    expected_emotions = set()
    for genre in genres:
        if genre in genre_emotion_mapping:
            expected_emotions.update(genre_emotion_mapping[genre])
    
    if not expected_emotions:
        return None
    
    return 1 if row['emotion'] in expected_emotions else 0

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

# Create DataFrame with emotion scores
emotion_scores_df = pd.DataFrame([result['emotion_scores'] for result in results], index=non_null_idx)
for emotion in emotion_scores_df.columns:
    df.loc[non_null_idx, f'emotion_score_{emotion}'] = emotion_scores_df[emotion]

# Calculate genre-emotion match
df['genre_emotion_match'] = df.apply(calculate_genre_emotion_match, axis=1)

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

# Language Distribution (sorted)
plt.figure(figsize=(12, 6))
lang_counts = df['firstLanguage'].value_counts()
sns.barplot(x=lang_counts.head(15).index, y=lang_counts.head(15).values)
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

# Genre Analysis (sorted)
genres = []
for genre_list in df['genres'].dropna():
    genres.extend(genre_list.split(', '))
genre_counts = pd.Series(genres).value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.head(15).index, y=genre_counts.head(15).values)
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

# Average Rating by Country (top 20 countries)
top_countries = df.groupby('mainCountry')['imdbRating'].agg(['mean', 'count']).sort_values('count', ascending=False).head(20)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries.index, y=top_countries['mean'])
plt.title('Average Rating by Country (Top 20 by Movie Count)')
plt.xlabel('Country')
plt.ylabel('Average IMDb Rating')
plt.xticks(rotation=45)
save_plot('avg_rating_by_country.png')

# Production Companies Analysis (sorted)
companies = []
for company_list in df['production'].dropna():
    companies.extend(company_list.split(', '))
company_counts = pd.Series(companies).value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=company_counts.head(15).index, y=company_counts.head(15).values)
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

# Emotion vs Rating Analysis
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='emotion', y='imdbRating')
plt.title('IMDb Ratings Distribution by Plot Emotion')
plt.xlabel('Emotion')
plt.ylabel('IMDb Rating')
plt.xticks(rotation=45)
save_plot('ratings_by_emotion.png')

# Genre vs Rating Analysis (sorted)
genres_list = df['genres'].dropna().str.split(', ', expand=True).stack().unique()
genre_ratings = []
for genre in genres_list:
    mask = df['genres'].str.contains(genre, na=False)
    avg_rating = df[mask]['imdbRating'].mean()
    count = mask.sum()
    genre_ratings.append({'Genre': genre, 'Average Rating': avg_rating, 'Count': count})

genre_df = pd.DataFrame(genre_ratings)
genre_df = genre_df.sort_values('Average Rating', ascending=False)  # Sort by rating
plt.figure(figsize=(15, 6))
sns.barplot(data=genre_df, x='Genre', y='Average Rating', hue='Count', palette='viridis')
plt.title('Average IMDb Rating by Genre (Sorted)')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.legend(title='Number of Movies')
save_plot('avg_rating_by_genre.png')

# Improved Genre-Emotion Match Analysis
# First split genres into individual entries
individual_genres = df['genres'].dropna().str.split(', ', expand=True).stack()
genre_matches = []

for genre in genre_emotion_mapping.keys():
    # Get movies of this genre
    genre_mask = df['genres'].str.contains(genre, na=False)
    genre_movies = df[genre_mask]
    
    if len(genre_movies) == 0:
        continue
        
    # Calculate matches for expected emotions
    expected_emotions = genre_emotion_mapping[genre]
    emotion_matches = genre_movies['emotion'].isin(expected_emotions)
    
    # Calculate statistics
    total_movies = len(genre_movies)
    matched_movies = emotion_matches.sum()
    match_rate = matched_movies / total_movies
    
    # Get most common emotions for this genre
    emotion_counts = genre_movies['emotion'].value_counts()
    top_emotions = emotion_counts.head(3).index.tolist()
    
    genre_matches.append({
        'Genre': genre,
        'Match Rate': match_rate,
        'Movie Count': total_movies,
        'Matched Movies': matched_movies,
        'Expected Emotions': ', '.join(expected_emotions),
        'Top Actual Emotions': ', '.join(top_emotions)
    })

# Create DataFrame with results
genre_match_df = pd.DataFrame(genre_matches)
genre_match_df = genre_match_df.sort_values('Movie Count', ascending=False)

# Print detailed analysis
print("\nGenre-Emotion Match Analysis:")
print(genre_match_df.to_string(index=False))

# Create visualization
plt.figure(figsize=(12, 8))
bars = plt.bar(genre_match_df['Genre'], genre_match_df['Match Rate'])
plt.title('Emotion-Genre Match Rate by Genre')
plt.xlabel('Genre')
plt.ylabel('Match Rate')
plt.xticks(rotation=45, ha='right')

# Add movie count as text on top of bars
for idx, bar in enumerate(bars):
    movie_count = genre_match_df.iloc[idx]['Movie Count']
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'n={movie_count}',
             ha='center', va='bottom')

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
save_plot('genre_emotion_match_improved.png')

# Create a detailed heatmap of actual vs expected emotions
plt.figure(figsize=(15, 8))

# Create genre-emotion mapping for analysis
genre_emotion_data = []
for genre in genre_emotion_mapping.keys():
    # Get movies of this genre
    genre_mask = df['genres'].str.contains(genre, na=False)
    genre_movies = df[genre_mask]
    
    if len(genre_movies) == 0:
        continue
    
    # Calculate emotion proportions for this genre
    emotion_props = genre_movies['emotion'].value_counts(normalize=True)
    
    # Convert to dictionary and ensure all emotions are present
    emotion_dict = emotion_props.to_dict()
    for emotion in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']:
        if emotion not in emotion_dict:
            emotion_dict[emotion] = 0.0
            
    genre_emotion_data.append({
        'Genre': genre,
        **emotion_dict
    })

# Create DataFrame with emotion proportions
genre_emotion_df = pd.DataFrame(genre_emotion_data)
genre_emotion_df.set_index('Genre', inplace=True)

# Create heatmap
sns.heatmap(genre_emotion_df, 
            annot=True, 
            fmt='.2f', 
            cmap='YlOrRd',
            cbar_kws={'label': 'Proportion of Movies'})
plt.title('Distribution of Emotions by Genre')
plt.xlabel('Emotion')
plt.ylabel('Genre')
plt.tight_layout()
save_plot('genre_emotion_distribution.png')

# Print the actual proportions
print("\nEmotion Distribution by Genre:")
print(genre_emotion_df.round(2).to_string())

# Add a comparison with expected emotions
print("\nComparison with Expected Emotions:")
for genre in genre_emotion_mapping:
    if genre in genre_emotion_df.index:
        print(f"\n{genre}:")
        print(f"Expected emotions: {genre_emotion_mapping[genre]}")
        print("Top actual emotions:", genre_emotion_df.loc[genre].nlargest(3).index.tolist())

# Emotion Score Distribution by Genre
plt.figure(figsize=(15, 8))
emotions = ['anger', 'joy', 'fear', 'sadness', 'surprise', 'disgust', 'neutral']
genre_emotion_data = []

for genre in genres_list[:10]:  # Top 10 genres
    mask = df['genres'].str.contains(genre, na=False)
    for emotion in emotions:
        avg_score = df.loc[mask, f'emotion_score_{emotion}'].mean()
        genre_emotion_data.append({'Genre': genre, 'Emotion': emotion, 'Average Score': avg_score})

genre_emotion_df = pd.DataFrame(genre_emotion_data)
genre_emotion_pivot = genre_emotion_df.pivot(index='Genre', columns='Emotion', values='Average Score')

plt.figure(figsize=(12, 8))
sns.heatmap(genre_emotion_pivot, annot=True, fmt='.2f', cmap='RdYlBu_r')
plt.title('Average Emotion Scores by Genre')
plt.tight_layout()
save_plot('genre_emotion_heatmap.png')

# Correlation between emotion scores and ratings (sorted)
emotion_rating_corr = df[[f'emotion_score_{e}' for e in emotions] + ['imdbRating']].corr()['imdbRating'].drop('imdbRating')
emotion_rating_corr = emotion_rating_corr.sort_values(ascending=False)  # Sort correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=emotion_rating_corr.index, y=emotion_rating_corr.values)
plt.title('Correlation between Emotion Scores and IMDb Rating (Sorted)')
plt.xlabel('Emotion')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
save_plot('emotion_rating_correlation.png')

# Production Companies Analysis
print("\nProduction Companies Analysis:")
print("\nSample of production values:")
print(df['production'].head().to_string())

# Split production companies and get unique counts
all_production_companies = []
for companies in df['production'].dropna():
    all_production_companies.extend(companies.split(', '))
    
production_counts = pd.Series(all_production_companies).value_counts()
print("\nTop 20 Production Companies:")
print(production_counts.head(20))

# Keywords Analysis
print("\nKeywords Analysis:")
print("\nSample of keywords values:")
print(df['keywords'].head().to_string())

# Split keywords and get unique counts
all_keywords = []
for keywords in df['keywords'].dropna():
    all_keywords.extend(keywords.split(', '))
    
keyword_counts = pd.Series(all_keywords).value_counts()
print("\nTop 20 Keywords:")
print(keyword_counts.head(20))

# Visualize top production companies
plt.figure(figsize=(12, 6))
sns.barplot(x=production_counts.head(15).index, y=production_counts.head(15).values)
plt.title('Top 15 Production Companies')
plt.xlabel('Production Company')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')
save_plot('top_production_companies_detailed.png')

# Visualize top keywords
plt.figure(figsize=(12, 6))
sns.barplot(x=keyword_counts.head(15).index, y=keyword_counts.head(15).values)
plt.title('Top 15 Keywords')
plt.xlabel('Keyword')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
save_plot('top_keywords.png')

# Word cloud for keywords
from wordcloud import WordCloud
# Create word cloud from keywords
wordcloud = WordCloud(width=1200, height=800, background_color='white').generate(' '.join(all_keywords))
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Keyword Word Cloud')
save_plot('keyword_wordcloud.png')

# Head of genres column
print("\nSample of genres values:")
print(df['genres'].head().to_string())



