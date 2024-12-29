import openai

openai.api_key = "your_openai_api_key"

response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input="Test input"
)

print(response['data'][0]['embedding'])
