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

# Helper function for cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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

def format_row(row):
    return ', '.join([f"{col}: {row[col]}" for col in columns_to_merge if pd.notna(row[col])])

def api_call(row):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=row
    )
    return response['data'][0]['embedding']

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

    # Initialize columns for match results
    prospective_df['Guide Profile'] = np.nan
    prospective_df['Match Explanation'] = np.nan

    # Prepare a list to store matches
    matches = []

    # Calculate similarity and find one-to-one matches
    for i, student_row in prospective_df.iterrows():
        similarities = current_df['Embeddings'].apply(
            lambda x: cosine_similarity(student_row['Embeddings'], x)
        )
        current_df = current_df.assign(similarity=similarities)
        top_match = current_df.sort_values(by="similarity", ascending=False).head(1)

        if not top_match.empty:
            guide_profile = top_match.iloc[0]['Slate ID']
            matches.append({
                'Guide Profile': guide_profile,
                'Student Profile': student_row['Slate ID']
            })

            # Remove matched guide from the pool
            current_df = current_df[current_df['Slate ID'] != guide_profile]

    # Create a DataFrame from matches
    matches_df = pd.DataFrame(matches)

    # Generate match explanations
    logging.info("Starting to generate match explanations...")
    try:
        matches_df = append_match_explanations(matches_df)
        logging.info("Descriptions generated successfully.")
    except Exception as e:
        logging.error(f"Error generating descriptions: {e}")
        raise

    # Save the results
    output_path = os.path.join(os.path.dirname(prospective_path), "matched_students.csv")
    matches_df.to_csv(output_path, index=False)
    logging.info(f"Matched students file saved to {output_path}.")

    return {"csv_path": output_path}

@celery_app.task
def delete_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")


