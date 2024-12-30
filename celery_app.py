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

    # Initialize a list to store similarity scores
    similarity_scores = []

    # Calculate similarity scores between all prospective students and current students (guides)
    for _, prospective_student in prospective_df.iterrows():
        for _, guide in current_df.iterrows():
            prospective_vector = np.array([prospective_student["YOG"]])  # You can add more features here for better matching
            guide_vector = np.array([guide["YOG"]])  # Same as above for guides
            
            similarity = cosine_similarity(prospective_vector, guide_vector)
            similarity_scores.append({
                "prospective_student": prospective_student["Slate ID"],
                "guide": guide["Slate ID"],
                "similarity": similarity
            })

    # Convert the list of similarity scores to a DataFrame
    similarity_df = pd.DataFrame(similarity_scores)

    # Sort the similarity dataframe by similarity in descending order
    similarity_df = similarity_df.sort_values(by="similarity", ascending=False)

    # Now we want to match the prospective students to guides based on the highest similarity
    paired_guides = set()
    matches = []

    for _, row in similarity_df.iterrows():
        # If the guide hasn't been paired yet, pair it with the student
        if row["guide"] not in paired_guides:
            matches.append({
                "Guide Profile": row["guide"],
                "Student Profile": row["prospective_student"]
            })
            paired_guides.add(row["guide"])  # Mark this guide as paired

        # Stop once all prospective students have been paired
        if len(matches) == len(prospective_df):
            break

    # Create a DataFrame from the matched pairs
    matched_df = pd.DataFrame(matches)

    # Generate match explanations
    logging.info("Starting to generate match explanations...")
    try:
        matched_df = append_match_explanations(matched_df)
        logging.info("Descriptions generated successfully.")
    except Exception as e:
        logging.error(f"Error generating descriptions: {e}")
        raise

    # Save the results
    output_path = os.path.join(os.path.dirname(prospective_path), "matched_students.csv")
    matched_df.to_csv(output_path, index=False)
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

