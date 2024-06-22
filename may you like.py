from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np

app = FastAPI()

# Assuming df is your DataFrame with selected columns and cleaned data
selected_columns = ['Title', 'Tag', 'Review', 'Comment', 'Country', 'Price', 'Rating', 'tags', 'img_link']

# Ensure the CSV file contains the 'img_link' column, or add it if missing
df = pd.read_csv(r"C:\Users\HP\Downloads\projec_2024\Final model\updated_data_with_img.csv")
for col in selected_columns:
    if col not in df.columns:
        df[col] = None

df = df[selected_columns].dropna(subset=['Title', 'Tag', 'Review', 'Comment', 'Country', 'Price', 'Rating', 'tags'])

df['Tag'] = df['Tag'].astype(str)
df['Review'] = df['Review'].astype(str)
df['Comment'] = df['Comment'].astype(str)

# Replace NaN values in 'img_link' column with empty string
df['img_link'] = df['img_link'].fillna('')

# Feature Engineering: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Tag'] + ' ' + df['Review'] + ' ' + df['Comment'])

# Define high and medium ratings thresholds
HIGH_RATING_THRESHOLD = 4.0
MEDIUM_RATING_THRESHOLD = 3.0

class UserRequest(BaseModel):
    tags: list[str]

def get_recommendations_by_tags(chosen_tags, df, num_recommendations=5):
    recommendations_df = pd.DataFrame(columns=['Title', 'Price', 'Rating', 'Country', 'tags', 'combined_text', 'img_link'])

    for tag in chosen_tags:
        response_indices = [
            i for i, tag_value in enumerate(df['tags']) 
            if tag.lower() in tag_value.lower() and df['Rating'].iloc[i] >= MEDIUM_RATING_THRESHOLD
        ]

        if response_indices:
            selected_indices = random.sample(response_indices, min(len(response_indices), num_recommendations))
            recommendations = df.iloc[selected_indices][['Title', 'Price', 'Rating', 'Country', 'tags', 'img_link']]
            recommendations['combined_text'] = df.iloc[selected_indices]['Tag'] + ' ' + df.iloc[selected_indices]['Review'] + ' ' + df.iloc[selected_indices]['Comment']
            recommendations_df = pd.concat([recommendations_df, recommendations])

    recommendations_df = recommendations_df.drop_duplicates().reset_index(drop=True)
    return recommendations_df

@app.post("/recommendations/")
async def get_recommendations(request: UserRequest):
    recommendations = get_recommendations_by_tags(request.tags, df)
    
    # Convert DataFrame to dictionary
    recommendations_dict = recommendations[['Title', 'Country', 'Price', 'tags', 'img_link']].fillna('').to_dict(orient='records')
    
    return {
        "recommendations": recommendations_dict
    }

# Evaluate the model's performance with actual data from 'df'
def evaluate_performance(recommendations, actual_data, threshold=0.5):
    recommendations['combined_text'] = recommendations['combined_text'].str.lower().str.strip()
    actual_data['combined_text'] = (actual_data['Tag'] + ' ' + actual_data['Review'] + ' ' + actual_data['Comment']).str.lower().str.strip()

    tfidf_actual_data = tfidf_vectorizer.transform(actual_data['combined_text'])
    tfidf_recommendations = tfidf_vectorizer.transform(recommendations['combined_text'])

    cosine_similarities = cosine_similarity(tfidf_recommendations, tfidf_actual_data)
    true_positive = sum(cosine_similarities.max(axis=1) >= threshold)
    precision = true_positive / len(recommendations) if len(recommendations) > 0 else 0
    recall = true_positive / len(actual_data) if len(actual_data) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return precision, recall

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
