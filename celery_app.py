import os
import pandas as pd
import numpy as np
from celery import Celery
from dotenv import load_dotenv
from gpt_utils import generate_match_explanation, append_match_explanations
import logging
import openai
from openai import OpenAIError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Initialize Celery
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def format_row(row):
    columns_to_merge = [
        'Res Status', 'Person Sex', 'Person Academic Interests',
        'Person Extra-Curricular Interest', 'Sport1', 'Sport2', 'Sport3',
        'City', 'State/Region', 'Country', 'School', 'Person Race', 'Person Hispanic'
    ]
    return ', '.join([f"{col}: {row[col]}" for col in columns_to_merge if pd.notna(row[col])])

@celery_app.task
def generate_embeddings_task(prospective_path, current_path):
    # Load CSV files
    logging.info(f"Loading CSV files: {prospective_path}, {current_path}")
    prospective_df = pd.read_csv(prospective_path)
    current_df = pd.read_csv(current_path)

    # Format rows into text queries for embedding
    prospective_df['Text Query'] = prospective_df.apply(format_row, axis=1)
    current_df['Text Query'] = current_df.apply(format_row, axis=1)

    # Generate embeddings using OpenAI API
    prospective_df['Embeddings'] = prospective_df['Text Query'].apply(api_call)
    current_df['Embeddings'] = current_df['Text Query'].apply(api_call)

    matches = []

    for _, prospective_row in prospective_df.iterrows():
        # Filter current students by matching sex and YOG
        filtered_current_df = current_df[
            (current_df["Person Sex"] == prospective_row["Person Sex"]) &
            (current_df["YOG"] == prospective_row["YOG"])
        ]

        if filtered_current_df.empty:
            continue

        # Calculate cosine similarity for embeddings
        filtered_current_df["similarity"] = filtered_current_df["Embeddings"].apply(
            lambda x: cosine_similarity(prospective_row["Embeddings"], x)
        )

        # Sort by similarity in descending order and select top 3 matches
        top_matches = filtered_current_df.sort_values(by="similarity", ascending=False).head(3)

        for _, match_row in top_matches.iterrows():
            matches.append({
                "Guide Profile": match_row["Slate ID"],
                "Student Profile": prospective_row["Slate ID"],
                "similarity": match_row["similarity"]
            })

    matches_df = pd.DataFrame(matches)

    # Generate match explanations
    logging.info("Generating match explanations...")
    try:
        matches_df = append_match_explanations(matches_df)
        logging.info("Match explanations added successfully.")
    except Exception as e:
        logging.error(f"Error generating explanations: {e}")

    # Save matches to CSV
    output_path = os.path.join(os.path.dirname(prospective_path), "matched_students.csv")
    matches_df.to_csv(output_path, index=False)
    logging.info(f"Matched students saved to {output_path}")

    return {"csv_path": output_path}

@celery_app.task
def delete_files(file_paths):
    """
    Deletes a list of files from the filesystem.

    Args:
        file_paths (list): A list of file paths to delete.
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")


def api_call(row):
    """
    Calls the OpenAI API to generate text embeddings.

    Args:
        row (str): The text query to generate embeddings for.

    Returns:
        list: The generated embedding vector or None if an error occurs.
    """
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=row
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error in API call for row: {row}. Exception: {e}")
        return None

