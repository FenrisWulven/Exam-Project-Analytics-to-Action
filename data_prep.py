import pandas as pd
import os
import kagglehub
import re
import numpy as np
from fuzzywuzzy import fuzz
from tqdm import tqdm

# Define data path for downloads
data_path = './data'  # You can change this to any directory you prefer
os.makedirs(data_path, exist_ok=True)

# Create paths for individual datasets
boxoffice_path = os.path.join(data_path, 'box_office')
movies_path = os.path.join(data_path, 'movies')
os.makedirs(boxoffice_path, exist_ok=True)
os.makedirs(movies_path, exist_ok=True)

# Download datasets (commented out to avoid re-downloading if files exist)
# box_office_files = kagglehub.dataset_download("harios/box-office-data-1984-to-2024-from-boxofficemojo", path=boxoffice_path)
# movies_files = kagglehub.dataset_download("rounakbanik/the-movies-dataset", path=movies_path)

# Function to clean movie titles for better matching
def clean_title(title):
    if pd.isna(title):
        return ""
    # Convert to lowercase
    title = title.lower()
    # Remove year information in parentheses
    title = re.sub(r'\(\d{4}\)', '', title)
    # Remove special characters
    title = re.sub(r'[^\w\s]', '', title)
    # Remove extra spaces
    title = re.sub(r'\s+', ' ', title).strip()
    return title

# Function to find the best match between two titles
def find_best_match(title, year, candidates_df, title_col, year_col, min_score=85):
    best_match = None
    best_score = 0
    
    # Clean the input title
    clean_input_title = clean_title(title)
    
    if not clean_input_title:
        return None
    
    # Filter candidates by year if available
    if not pd.isna(year) and year_col in candidates_df.columns:
        year_matches = candidates_df[candidates_df[year_col] == year]
        if len(year_matches) > 0:
            candidates_df = year_matches
    
    # Find the best match
    for idx, row in candidates_df.iterrows():
        candidate_title = clean_title(row[title_col])
        if not candidate_title:
            continue
            
        # Calculate similarity score
        score = fuzz.ratio(clean_input_title, candidate_title)
        
        # If we have a perfect match, return immediately
        if score == 100:
            return row
            
        # Update best match if this score is better
        if score > best_score and score >= min_score:
            best_score = score
            best_match = row
    
    return best_match

def main():
    print("Loading IMDb dataset...")
    imdb_path = './fata2025/datasets/WillandAgency/European_data_2000.csv'
    imdb_df = pd.read_csv(imdb_path)
    
    # Look for box office data files
    box_office_csvs = [f for f in os.listdir(boxoffice_path) if f.endswith('.csv')]
    if not box_office_csvs:
        print(f"No box office CSV files found in {boxoffice_path}")
        print("Please download the box office dataset first.")
        return
    
    print(f"Loading box office dataset: {box_office_csvs[0]}")
    box_office_df = pd.read_csv(os.path.join(boxoffice_path, box_office_csvs[0]))
    
    print("IMDb dataset shape:", imdb_df.shape)
    print("Box Office dataset shape:", box_office_df.shape)
    
    # Display column names to help with mapping
    print("\nIMDb columns:", imdb_df.columns.tolist())
    print("Box Office columns:", box_office_df.columns.tolist())
    
    # Determine matching columns (assuming title and year are common)
    imdb_title_col = 'originalTitle' if 'originalTitle' in imdb_df.columns else 'primaryTitle'
    imdb_year_col = 'releaseYear' if 'releaseYear' in imdb_df.columns else 'startYear'
    box_office_title_col = 'title' if 'title' in box_office_df.columns else 'Title'
    box_office_year_col = 'year' if 'year' in box_office_df.columns else 'Year'
    
    # Add new columns for box office data
    boxoffice_columns = ['domestic_gross', 'worldwide_gross', 'opening_weekend', 'budget']
    for col in boxoffice_columns:
        if col in box_office_df.columns:
            imdb_df[f'boxoffice_{col}'] = np.nan
    
    # Match IMDb movies with box office data
    print("\nMatching IMDb movies with box office data...")
    match_count = 0
    
    # Create a progress bar
    for i in tqdm(range(len(imdb_df))):
        row = imdb_df.iloc[i]
        title = row[imdb_title_col]
        year = row[imdb_year_col]
        
        # Skip if no title
        if pd.isna(title):
            continue
            
        # Find the best matching movie in the box office dataset
        match = find_best_match(title, year, box_office_df, box_office_title_col, box_office_year_col)
        
        # If we found a match, add the box office data to the IMDb dataframe
        if match is not None:
            match_count += 1
            for col in boxoffice_columns:
                if col in box_office_df.columns:
                    imdb_df.loc[i, f'boxoffice_{col}'] = match[col]
    
    print(f"\nMatched {match_count} movies ({match_count/len(imdb_df)*100:.2f}%)")
    
    # Convert box office columns to numeric
    for col in [f'boxoffice_{c}' for c in boxoffice_columns if c in box_office_df.columns]:
        imdb_df[col] = pd.to_numeric(imdb_df[col], errors='coerce')
    
    # Save the enriched dataset
    output_file = os.path.join(data_path, 'imdb_with_boxoffice.csv')
    imdb_df.to_csv(output_file, index=False)
    print(f"Saved enriched dataset to {output_file}")
    
    # Print some statistics about the merged data
    print("\nBox Office Statistics:")
    for col in [f'boxoffice_{c}' for c in boxoffice_columns if c in box_office_df.columns]:
        non_null = imdb_df[col].notna().sum()
        print(f"{col}: {non_null} non-null values ({non_null/len(imdb_df)*100:.2f}%)")
        if non_null > 0:
            print(f"  Mean: ${imdb_df[col].mean():,.2f}")
            print(f"  Max: ${imdb_df[col].max():,.2f}")
    
    # Sample of matched movies
    matched_movies = imdb_df[imdb_df['boxoffice_domestic_gross'].notna()]
    if not matched_movies.empty:
        print("\nSample of matched movies:")
        sample = matched_movies.sample(min(5, len(matched_movies)))
        for _, row in sample.iterrows():
            print(f"{row[imdb_title_col]} ({row[imdb_year_col]}): ${row['boxoffice_domestic_gross']:,.2f} domestic gross")

if __name__ == "__main__":
    main()

