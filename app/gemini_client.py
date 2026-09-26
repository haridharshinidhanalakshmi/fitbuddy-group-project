from google import genai
from google.genai import types

from app.config import settings


def generate_with_gemini(prompt: str, model: str) -> str:
    """
    Send a prompt to Gemini and return the generated text.
    """

    if not settings.gemini_api_key:
        raise RuntimeError(
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY to the .env file."
        )

    client = genai.Client(
        api_key=settings.gemini_api_key
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=5000,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()