# gpt_utils.py

import openai

# Set OpenAI API Key (store securely, e.g., environment variable)
openai.api_key = "your_openai_api_key"

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
    response = openai.Completion.create(
        engine="text-davinci-003",  # Or another GPT model
        prompt=prompt,
        max_tokens=60,
        n=1,
        temperature=0.7,
    )
    return response["choices"][0]["text"].strip()
