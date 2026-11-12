
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from functools import lru_cache

load_dotenv()



@lru_cache(maxsize=1)
def get_groq_model():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    model_name = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )

    return ChatGroq(
        model=model_name,
        temperature=0,
        api_key=api_key
    )