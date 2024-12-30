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

from gpt_utils import append_match_explanations  # Ensure this import works correctly

@celery_app.task
def generate_embeddings_task(prospective_path, current_path):
    # Load CSV files
    prospective_df = pd.read_csv(prospective_path)
    current_df = pd.read_csv(current_path)

    logging.info("Loaded prospective and current CSVs.")

    # Validate required columns
    required_columns = ["Guide Profile", "Student Profile", "Slate ID", "YOG"]
    for df_name, df in [("prospective", prospective_df), ("current", current_df)]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.warning(f"{df_name} CSV is missing columns: {missing_columns}")
            for col in missing_columns:
                df[col] = "N/A"  # Add placeholders if necessary

    # Generate descriptions
    logging.info("Starting to generate match explanations...")
    try:
        matches_df = prospective_df[['Guide Profile', 'Student Profile']].copy()
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
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
