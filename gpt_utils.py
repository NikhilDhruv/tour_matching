from dotenv import load_dotenv
import os
import logging
import openai
from openai import OpenAIError



# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set OpenAI API Key (securely fetched from the environment)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

def generate_match_explanation(guide, student):
    """
    Generate a two-sentence explanation for why a guide and a prospective student are matched.
    """
    prompt = f"""
    A tour guide and a prospective student have been matched based on their profiles.
    Provide a concise, two-sentence explanation for why they are a good match.
    Here are the details:

    Tour Guide: {guide}
    Prospective Student: {student}

    Explanation (exactly two sentences):
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except OpenAIError as e:
        logging.error(f"Error generating explanation: {e}")
        return f"Error generating explanation: {e}"

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
