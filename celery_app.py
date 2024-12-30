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
    if not vec1 or not vec2:
        return 0
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@celery_app.task
def generate_embeddings_task(prospective_path, current_path):
    # Load CSV files
    logging.info(f"Loading CSV files: {prospective_path}, {current_path}")
    prospective_df = pd.read_csv(prospective_path)
    current_df = pd.read_csv(current_path)

    # Validate required columns
    required_columns = ["Guide Profile", "Student Profile", "Slate ID", "YOG"]
    for df_name, df in [("prospective", prospective_df), ("current", current_df)]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.warning(f"{df_name} CSV is missing columns: {missing_columns}")
            for col in missing_columns:
                df[col] = "N/A"  # Add placeholders

    # Assign the Slate ID of prospective students to the "Student Profile"
    prospective_df["Student Profile"] = prospective_df["Slate ID"]
    # Assign the Slate ID of current students to the "Guide Profile"
    current_df["Guide Profile"] = current_df["Slate ID"]

    # Matching prospective students to current students (guides) based on similarity
    matched_guides = []
    for _, prospective_student in prospective_df.iterrows():
        best_match = None
        highest_similarity = -1
        
        for _, guide in current_df.iterrows():
            # Compute similarity score based on the provided data
            prospective_vector = np.array([prospective_student["YOG"]])  # Add other relevant features for better matching
            guide_vector = np.array([guide["YOG"]])  # Add other relevant features here for matching
            
            similarity = cosine_similarity(prospective_vector, guide_vector)
            
            if similarity > highest_similarity:
                best_match = guide["Guide Profile"]
                highest_similarity = similarity
        
        matched_guides.append(best_match)
    
    prospective_df["Guide Profile"] = matched_guides  # Assign matched guides to the prospective students

    # Now we can append match explanations to the dataframe
    logging.info("Starting to generate match explanations...")
    try:
        prospective_df = append_match_explanations(prospective_df)
        logging.info("Descriptions generated successfully.")
    except Exception as e:
        logging.error(f"Error generating descriptions: {e}")
        raise

    # Save the results
    output_path = os.path.join(os.path.dirname(prospective_path), "matched_students.csv")
    prospective_df.to_csv(output_path, index=False)
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
