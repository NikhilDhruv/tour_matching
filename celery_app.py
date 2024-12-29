import pandas as pd
import numpy as np
from openai import OpenAI
import os
from celery import Celery, current_task
from dotenv import load_dotenv
import pandas as pd
from gpt_utils import generate_match_explanation

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure Celery to use Redis as the message broker
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',  # Redis broker URL
    backend='redis://localhost:6379/0'  # Redis backend URL
)

client = OpenAI(
    api_key=OPENAI_API_KEY
)

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


columns_to_merge = ['Res Status',
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
 'Person Hispanic']

def format_row(row):
    return ', '.join([f"{col}: {row[col]}" for col in columns_to_merge if pd.notna(row[col])])

def api_call(row):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=row
    )
    return response.data[0].embedding


@celery_app.task
def generate_embeddings_task(prospective_path, current_path):
    prospective_df = pd.read_csv(prospective_path)
    current_df = pd.read_csv(current_path)


    prospective_df['Text Query'] = prospective_df.apply(format_row, axis=1)
    current_df['Text Query'] = current_df.apply(format_row, axis=1)

    prospective_df['Embeddings'] = prospective_df['Text Query'].apply(api_call)
    current_df['Embeddings'] = current_df['Text Query'].apply(api_call)

    prospective_df['suggestion_1'] = np.nan
    prospective_df['suggestion_2'] = np.nan
    prospective_df['suggestion_3'] = np.nan

    # Iterate over each prospective student
    for i, row in prospective_df.iterrows():
        # Filter current students by gender and YOG
        filtered_current_df = current_df[
            (current_df["Person Sex"] == row["Person Sex"]) &
            (current_df["YOG"] == row["YOG"])
        ]

        if filtered_current_df.empty:
            continue

        # Calculate cosine similarity with each student in the filtered current_df
        similarities = filtered_current_df["Embeddings"].apply(
            lambda x: cosine_similarity(row["Embeddings"], x)
        )

        # Add the similarities as a new column
        filtered_current_df = filtered_current_df.assign(similarity=similarities)

        # Sort by similarity in descending order
        top_matches = filtered_current_df.sort_values(by="similarity", ascending=False).head(3)

        # Extract the Slate IDs of the top matches
        top_suggestions = top_matches["Slate ID"].values

        # Update the prospective_df with top suggestions
        prospective_df.at[i, "suggestion_1"] = top_suggestions[0] if len(top_suggestions) > 0 else np.nan
        prospective_df.at[i, "suggestion_2"] = top_suggestions[1] if len(top_suggestions) > 1 else np.nan
        prospective_df.at[i, "suggestion_3"] = top_suggestions[2] if len(top_suggestions) > 2 else np.nan

    prospective_df.drop(columns=["Embeddings", "Text Query"], inplace=True)

    output_path = os.path.join(os.path.dirname(prospective_path), "matched_students.csv")
    prospective_df.to_csv(output_path, index=False)

    return {"csv_path": output_path}

@celery_app.task
def delete_files(file_paths):
    #Deletes uploaded files to free up storage
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")


def append_match_explanations(matches_df):
    """
    Add GPT-generated match explanations to the dataframe woohooo.
    """
    explanations = []
    for index, row in matches_df.iterrows():
        guide = row["Guide Profile"]  # Adjust column name as needed
        student = row["Student Profile"]  # Adjust column name as needed
        explanation = generate_match_explanation(guide, student)
        explanations.append(explanation)

    matches_df["Match Explanation"] = explanations
    return matches_df
