import os
import pandas as pd
import numpy as np
from celery import Celery
from dotenv import load_dotenv
import openai
from gpt_utils import generate_match_explanation

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

# Configure Celery with Redis as the broker and backend
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',  # Redis broker URL
    backend='redis://localhost:6379/0'  # Redis backend URL
)

# Helper function for cosine similarity
def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Columns to merge into a single string for embedding generation
columns_to_merge = [
    'Res Status',
    'Person Sex',
    'Person Academic Interests',
    'Person Extra-Curricular Interest',
    'Sport1',
    'Sport2',
    'Sport3',
    'City',
    'State/Region',
    'Country',
    'School',
    'Person Race',
    'Person Hispanic'
]

# Format a row as a single string
def format_row(row):
    return ', '.join([f"{col}: {row[col]}" for col in columns_to_merge if pd.notna(row[col])])

# Generate embedding using OpenAI API
def api_call(row):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[row]  # Input must be a list of strings
        )
        return response.data[0].embedding
    except Exception as e:  # Catch generic exceptions
        print(f"Error generating embedding for input: {row}, Error: {e}")
        return None


@celery_app.task
def generate_embeddings_task(prospective_path, current_path):
    # Load CSV files
    prospective_df = pd.read_csv(prospective_path)
    current_df = pd.read_csv(current_path)

    # Validate required columns
    required_columns = set(columns_to_merge + ["YOG", "Slate ID"])
    if not required_columns.issubset(prospective_df.columns) or not required_columns.issubset(current_df.columns):
        raise ValueError("CSV files are missing required columns.")

    # Create text queries for embedding generation
    prospective_df['Text Query'] = prospective_df.apply(format_row, axis=1)
    current_df['Text Query'] = current_df.apply(format_row, axis=1)

    # Generate embeddings
    prospective_df['Embeddings'] = prospective_df['Text Query'].apply(lambda x: api_call(x) or [])
    current_df['Embeddings'] = current_df['Text Query'].apply(lambda x: api_call(x) or [])

    # Add columns for top suggestions
    prospective_df['suggestion_1'] = np.nan
    prospective_df['suggestion_2'] = np.nan
    prospective_df['suggestion_3'] = np.nan

    # Find top matches for each prospective student
    for i, row in prospective_df.iterrows():
        filtered_current_df = current_df[
            (current_df["Person Sex"] == row["Person Sex"]) &
            (current_df["YOG"] == row["YOG"])
        ]

        if filtered_current_df.empty:
            continue

        similarities = filtered_current_df['Embeddings'].apply(
            lambda x: cosine_similarity(row['Embeddings'], x)
        )

        filtered_current_df = filtered_current_df.assign(similarity=similarities)
        top_matches = filtered_current_df.sort_values(by="similarity", ascending=False).head(3)
        top_suggestions = top_matches["Slate ID"].values

        prospective_df.at[i, "suggestion_1"] = top_suggestions[0] if len(top_suggestions) > 0 else np.nan
        prospective_df.at[i, "suggestion_2"] = top_suggestions[1] if len(top_suggestions) > 1 else np.nan
        prospective_df.at[i, "suggestion_3"] = top_suggestions[2] if len(top_suggestions) > 2 else np.nan

    prospective_df.drop(columns=["Embeddings", "Text Query"], inplace=True)

    # Save the results
    output_path = os.path.join(os.path.dirname(prospective_path), "matched_students.csv")
    prospective_df.to_csv(output_path, index=False)

    return {"csv_path": output_path}

@celery_app.task
def delete_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

def append_match_explanations(matches_df):
    explanations = []
    for index, row in matches_df.iterrows():
        guide = row["Guide Profile"]
        student = row["Student Profile"]
        explanation = generate_match_explanation(guide, student)
        explanations.append(explanation)

    matches_df["Match Explanation"] = explanations
    return matches_df
