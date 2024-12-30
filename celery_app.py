import os
import pandas as pd
import numpy as np
from celery import Celery
from dotenv import load_dotenv
from gpt_utils import generate_match_explanation, append_match_explanations

# Load environment variables
load_dotenv()

# Initialize Celery
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Helper function for cosine similarity
def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@celery_app.task
def generate_embeddings_task(prospective_path, current_path):
    # Load CSV files
    prospective_df = pd.read_csv(prospective_path)
    current_df = pd.read_csv(current_path)

    # Validate required columns
    required_columns = ["Guide Profile", "Student Profile", "Slate ID", "YOG"]
    if not set(required_columns).issubset(prospective_df.columns) or not set(required_columns).issubset(current_df.columns):
        raise ValueError("CSV files are missing required columns.")

    # Add text queries
    prospective_df['Text Query'] = prospective_df.apply(
        lambda row: ', '.join([f"{col}: {row[col]}" for col in required_columns if pd.notna(row[col])]),
        axis=1
    )
    current_df['Text Query'] = current_df.apply(
        lambda row: ', '.join([f"{col}: {row[col]}" for col in required_columns if pd.notna(row[col])]),
        axis=1
    )

    # Add columns for top matches
    prospective_df['suggestion_1'] = np.nan
    prospective_df['suggestion_2'] = np.nan
    prospective_df['suggestion_3'] = np.nan

    # Generate embeddings and find top matches
    for i, row in prospective_df.iterrows():
        filtered_current_df = current_df[
            (current_df["Person Sex"] == row["Person Sex"]) &
            (current_df["YOG"] == row["YOG"])
        ]

        if filtered_current_df.empty:
            continue

        similarities = filtered_current_df['Text Query'].apply(
            lambda x: cosine_similarity(row['Text Query'], x)
        )
        filtered_current_df = filtered_current_df.assign(similarity=similarities)
        top_matches = filtered_current_df.sort_values(by="similarity", ascending=False).head(3)

        # Update suggestions
        suggestions = top_matches["Slate ID"].values
        prospective_df.at[i, "suggestion_1"] = suggestions[0] if len(suggestions) > 0 else np.nan
        prospective_df.at[i, "suggestion_2"] = suggestions[1] if len(suggestions) > 1 else np.nan
        prospective_df.at[i, "suggestion_3"] = suggestions[2] if len(suggestions) > 2 else np.nan

    # Generate explanations
    matches_df = prospective_df[['Guide Profile', 'Student Profile']].copy()
    matches_df = append_match_explanations(matches_df)

    # Save the results
    output_path = os.path.join(os.path.dirname(prospective_path), "matched_students.csv")
    matches_df.to_csv(output_path, index=False)

    return {"csv_path": output_path}

@celery_app.task
def delete_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
