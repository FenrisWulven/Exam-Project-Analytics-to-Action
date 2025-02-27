import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ----- Settings and Constants -----
DATA_PATH = './fata2025/datasets/WillandAgency/European_data_2000.csv'
PLOTS_DIR = './plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Weights for the composite appeal metric (adjust as needed)
W_SENTIMENT = 0.25
W_GENRE_MATCH = 0.25
W_SIMILAR_RATING = 0.25
W_VOTES = 0.25

# Genre to expected emotion mapping
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

# ----- Sentiment Analysis Functions -----
def load_sentiment_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return tokenizer, model, device

def process_sentiment(texts, tokenizer, model, device, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        for pred in probs:
            idx = pred.argmax()
            result = {
                'sentiment_score': float(pred[idx]),
                'emotion': emotions[idx]
            }
            results.append(result)
    return results

def calculate_genre_emotion_match(genres_str, emotion):
    if pd.isna(genres_str) or pd.isna(emotion):
        return 0
    genres = [g.strip() for g in genres_str.split(',')]
    expected = set()
    for genre in genres:
        expected.update(genre_emotion_mapping.get(genre, []))
    return 1 if emotion in expected else 0

# ----- Embedding and Similarity Functions -----
def load_embedding_model():
    # Using a compact sentence transformer model for plot embeddings
    return SentenceTransformer('all-MiniLM-L6-v2')

def compute_plot_embeddings(plots, embedder):
    return embedder.encode(plots, show_progress_bar=True)

def compute_similarity_rating(embeddings, ratings, top_k=5):
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    avg_sim_ratings = []
    for i in range(sim_matrix.shape[0]):
        # Exclude self by setting similarity to -inf
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        # Get indices of top_k similar movies
        top_indices = sims.argsort()[-top_k:]
        # Average the IMDb ratings of similar movies
        avg_sim_ratings.append(np.mean(ratings[top_indices]))
    return np.array(avg_sim_ratings)

def evaluate_appeal_metric(df):
    """Evaluate the appeal metric using various methods"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    # import shap
    
    # Create evaluation plots directory
    eval_dir = os.path.join(PLOTS_DIR, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. Correlation Analysis
    correlation_metrics = ['imdbRating', 'numberOfVotes', 'runtimeMinutes']
    correlations = df[['appeal_metric'] + correlation_metrics].corr()['appeal_metric']
    
    plt.figure(figsize=(10, 6))
    correlations.drop('appeal_metric').sort_values(ascending=True).plot(kind='barh')
    plt.title('Correlation between Appeal Metric and Success Indicators')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'appeal_correlations.png'))
    plt.close()
    
    # 2. Scatter Plots with Regression
    for metric in correlation_metrics:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x='appeal_metric', y=metric)
        plt.title(f'Appeal Metric vs {metric}')
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f'appeal_vs_{metric}.png'))
        plt.close()
    
    # 3. Predictive Performance
    # Prepare features for prediction
    X = df[['norm_sentiment', 'genre_emotion_match', 'norm_sim_rating', 'norm_votes']]
    y = df['imdbRating']
    
    # Perform cross-validation
    model = LinearRegression()
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    print("\nCross-validation R² scores:")
    print(f"Mean R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # # 4. SHAP Analysis
    # model.fit(X, y)
    # explainer = shap.LinearExplainer(model, X)
    # shap_values = explainer.shap_values(X)
    
    # plt.figure(figsize=(10, 6))
    # shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    # plt.title('Feature Importance (SHAP Values)')
    # plt.tight_layout()
    # plt.savefig(os.path.join(eval_dir, 'shap_importance.png'))
    # plt.close()
    
    # 5. Component Analysis
    component_correlations = pd.DataFrame({
        'Component': ['Sentiment', 'Genre Match', 'Similar Rating', 'Votes'],
        'Correlation': [
            df['norm_sentiment'].corr(df['imdbRating']),
            df['genre_emotion_match'].corr(df['imdbRating']),
            df['norm_sim_rating'].corr(df['imdbRating']),
            df['norm_votes'].corr(df['imdbRating'])
        ]
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=component_correlations.sort_values('Correlation', ascending=False),
                x='Component', y='Correlation')
    plt.title('Correlation of Appeal Components with IMDb Rating')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'component_correlations.png'))
    plt.close()
    
    return {
        'correlations': correlations,
        'cv_scores': cv_scores,
        'component_correlations': component_correlations
    }

# ----- Main Pipeline -----
def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['plotLong', 'imdbRating', 'numberOfVotes']).reset_index(drop=True)
    
    # ----- Sentiment Analysis on plotLong -----
    tokenizer, sent_model, device = load_sentiment_model()
    plots = df['plotLong'].tolist()
    sentiment_results = process_sentiment(plots, tokenizer, sent_model, device)
    df['sentiment_score'] = [res['sentiment_score'] for res in sentiment_results]
    df['emotion'] = [res['emotion'] for res in sentiment_results]
    
    # Genre-emotion match score (binary: 1 if match, else 0)
    df['genre_emotion_match'] = df.apply(lambda row: calculate_genre_emotion_match(row['genres'], row['emotion']), axis=1)
    
    # ----- Plot Embeddings and Similar Movies Rating -----
    embedder = load_embedding_model()
    embeddings = compute_plot_embeddings(plots, embedder)
    imdb_ratings = df['imdbRating'].values
    df['sim_rating'] = compute_similarity_rating(embeddings, imdb_ratings)
    
    # ----- Normalize Features -----
    scaler = MinMaxScaler()
    df['norm_sentiment'] = scaler.fit_transform(df[['sentiment_score']])
    df['norm_votes'] = scaler.fit_transform(df[['numberOfVotes']])
    df['norm_sim_rating'] = scaler.fit_transform(df[['sim_rating']])
    
    # ----- Composite Appeal Metric -----
    # The metric combines:
    # 1. Normalized sentiment score (as proxy for emotional appeal)
    # 2. Genre-emotion match (binary; can be weighted)
    # 3. Normalized similar movies average rating (as indicator of potential reception)
    # 4. Normalized number of votes (as proxy for audience engagement)
    df['appeal_metric'] = (
        W_SENTIMENT * df['norm_sentiment'] +
        W_GENRE_MATCH * df['genre_emotion_match'] +
        W_SIMILAR_RATING * df['norm_sim_rating'] +
        W_VOTES * df['norm_votes']
    )
    
    # Evaluate the appeal metric
    print("\nEvaluating appeal metric...")
    eval_results = evaluate_appeal_metric(df)
    
    # Print evaluation results
    print("\nCorrelations with appeal metric:")
    print(eval_results['correlations'].round(3))
    
    print("\nComponent correlations with IMDb rating:")
    print(eval_results['component_correlations'].round(3))
    
    # Optional: Save detailed evaluation results
    eval_results_path = os.path.join(PLOTS_DIR, 'evaluation_results.txt')
    with open(eval_results_path, 'w') as f:
        f.write("Appeal Metric Evaluation Results\n")
        f.write("==============================\n\n")
        f.write("Correlations with success indicators:\n")
        f.write(str(eval_results['correlations'].round(3)))
        f.write("\n\nCross-validation scores:\n")
        f.write(f"Mean R²: {eval_results['cv_scores'].mean():.3f}")
        f.write(f" (+/- {eval_results['cv_scores'].std() * 2:.3f})")
    
    # ----- Optional: Save and Visualize -----
    output_csv = os.path.join(PLOTS_DIR, 'appeal_metric_results.csv')
    df.to_csv(output_csv, index=False)
    print(f"Saved appeal metric results to {output_csv}")
    
    # Plot distribution of appeal metric
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10,6))
        sns.histplot(df['appeal_metric'], bins=30, kde=True)
        plt.title('Distribution of Composite Appeal Metric')
        plt.xlabel('Appeal Metric')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'appeal_metric_distribution.png'))
        plt.close()
        print("Saved appeal metric distribution plot.")
    except ImportError:
        print("Matplotlib or Seaborn not installed. Skipping plotting.")
    
    # ----- Future Enhancements -----
    # Placeholder for:
    # - Incorporating text embeddings of loglines
    # - Adding features from social media scraping (Reddit, Letterboxd, IMDb forums)
    # - Integrating narrative structure analysis (e.g., three-act structure, Save the Cat beats)
    # - Using matrix factorization or SHAP values for model explainability

if __name__ == "__main__":
    main()
