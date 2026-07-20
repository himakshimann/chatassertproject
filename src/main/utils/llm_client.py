from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_response(messages, model="gpt-5", temperature=1.0):
    """
    Generic LLM interface.
    Returns only the generated text.
    """

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return response.choices[0].message.content