from google import genai

with open("src/gemini_api.key", "r") as f:
    api_key = f.read().strip()

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="Say hello in one sentence."
)

print(response.text)