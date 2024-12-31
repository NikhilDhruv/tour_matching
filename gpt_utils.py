from dotenv import load_dotenv
import os
import logging
import openai
from openai import OpenAIError

# Load environment variables
load_dotenv()

print(f"Loaded API Key: {os.getenv('OPENAI_API_KEY')}")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set OpenAI API Key (securely fetched from the environment)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

import openai
from openai import OpenAIError
import logging

import openai
from openai import OpenAIError
import logging

def generate_match_explanation(guide, student):
    """
    Generate a detailed explanation for why a guide and a student are a good match.
    """
    explanation_parts = []

    # Gender Match
    if guide.get("Person Sex") == student.get("Person Sex"):
        explanation_parts.append("Both the tour guide and prospective student share the same gender, ensuring a comfortable and relatable tour experience.")

    # Academic Interest Match
    if guide.get("Person Academic Interests") == student.get("Person Academic Interests"):
        explanation_parts.append(
            f"The guide's background in {guide['Person Academic Interests']} aligns with the student's academic interests, providing opportunities for in-depth discussions."
        )

    # Extracurricular Interest Match
    if guide.get("Person Extra-Curricular Interest") == student.get("Person Extra-Curricular Interest"):
        explanation_parts.append(
            f"Both share a passion for {guide['Person Extra-Curricular Interest']}, creating a connection through shared extracurricular enthusiasm."
        )

    # Location Match
    if guide.get("City") == student.get("City"):
        explanation_parts.append(
            f"Being from the same city ({guide['City']}), the guide can offer valuable insights into navigating the area and transitioning to campus life."
        )

    # Default Explanation if no specific matches
    if not explanation_parts:
        explanation_parts.append(
            "The guide's extensive experience and friendly demeanor make them a great match for the student's tour experience."
        )

    # Combine parts into a single explanation
    return " ".join(explanation_parts)


def append_match_explanations(matches_df):
    """
    Add GPT-generated match explanations to the DataFrame.
    """
    # Validate required columns
    required_columns = ["Guide Profile", "Student Profile"]
    if not all(col in matches_df.columns for col in required_columns):
        raise ValueError(f"Required columns {required_columns} are missing from the DataFrame.")

    explanations = []
    for index, row in matches_df.iterrows():
        guide = row["Guide Profile"]
        student = row["Student Profile"]
        try:
            # Generate match explanation using GPT
            explanation = generate_match_explanation(guide, student)
        except Exception as e:
            logging.error(f"Error generating explanation at index {index}: {e}")
            explanation = f"Error generating explanation: {e}"

        explanations.append(explanation)

    matches_df["Match Explanation"] = explanations
    return matches_df
