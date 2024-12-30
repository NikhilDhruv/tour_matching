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

# Helper function for similarity based on the required criteria
def calculate_similarity(student, guide):
    """
    Calculate similarity between a prospective student and a guide
    based on sex, academic interest, and location.
    """
    # 1. Sex similarity (must match)
    sex_score = 1 if student["Person Sex"] == guide["Person Sex"] else 0

    # 2. Academic Interest similarity (simple match, same field = 1)
    academic_interest_score = 1 if student["Person Academic Interests"] == guide["Person Academic Interests"] else 0

    # 3. Location similarity (city/state/region)
    location_score = 1 if student["City"] == guide["City"] else 0

    # Total similarity score is the sum of all individual scores
    return sex_score + academic_interest_score + location_score

@celery_app.task
def generate_embeddings_task(prospective_path, current_path):
    # Load CSV files
    logging.info(f"Loading CSV files: {prospective_path}, {current_path}")
    prospective_df = pd.read_csv(prospective_path)
    current_df = pd.read_csv(current_path)

    # Validate required columns
    required_columns = ["Guide Profile", "Student Profile", "Slate ID", "YOG", "Person Sex", "Person Academic Interests", "City"]
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
            # Calculate similarity based on sex, academic interest, and location
            similarity = calculate_similarity(prospective_student, guide)
            similarity_scores.append({
                "prospective_student": prospective_student["Slate ID"],
                "guide": guide["Slate ID"],
                "similarity": similarity
            })

    # Convert the list of similarity scores to a DataFrame
    similarity_df = pd.DataFrame(similarity_scores)

    # Sort the similarity dataframe by similarity in descending order
    similarity_df = similarity_df.sort_values(by="similarity", ascending=False)

    # Initialize a set for tracking paired guides and students
    paired_guides = set()
    paired_students = set()
    matches = []

    # Now we want to match the prospective students to guides based on the highest similarity
    for _, row in similarity_df.iterrows():
        # Only pair if both guide and student are not already paired
        if row["guide"] not in paired_guides and row["prospective_student"] not in paired_students:
            matches.append({
                "Guide Profile": row["guide"],
                "Student Profile": row["prospective_student"]
            })
            paired_guides.add(row["guide"])  # Mark this guide as paired
            paired_students.add(row["prospective_student"])  # Mark this student as paired

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
