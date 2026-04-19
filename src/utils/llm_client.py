# src/utils/llm_client.py
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def _setup_langsmith() -> None:
    """
    Ensures LangSmith tracing is enabled. 
    It will pull from your .env, but sets defaults just in case.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    # You can name the project whatever you like to identify it in the LangSmith dashboard
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

def get_groq_client() -> Groq:
    """
    Initializes and returns the Groq client.
    Raises an error if the API key is missing, failing fast before a loop starts.
    """
    _setup_langsmith()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please create a .env file at the root of the project and add your key."
        )
        
    # The Groq client is thread-safe and can be instantiated here
    return Groq(api_key=api_key)