
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_groq_model():
    """
    Initialize the Groq chat model.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Please configure it in .env"
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )

    model = ChatGroq(
        model=model_name,
        temperature=0,
        api_key=api_key
    )

    return model
