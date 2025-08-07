import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_PROVIDER = "gemini"

# Define API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

# Define base URLs for providers
OPENAI_BASE_URL = None
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"


GROQ_MODELS = {
    "default": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "LLAMA_4_SCOUT": "meta-llama/llama-4-scout-17b-16e-instruct",
    "LLAMA_4_MAVERICK": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "LLAMA_3_3_VERSATILE": "llama-3.3-70b-versatile",
}

OPENAI_MODELS = {
    "default": "gpt-4.1-nano",
    "4o_MINI": "gpt-4o-mini",
    "41_MINI": "gpt-4.1-mini",
    "41_NANO": "gpt-4.1-nano",
    "o3": "o3",
}

GEMINI_MODELS = {
    "default": "gemini-2.5-flash-lite",
    "20_FLASH": "gemini-2.0-flash",
    "25_FLASH": "gemini-2.5-flash",
    "25_FLASH_LITE": "gemini-2.5-flash-lite",
    "25_PRO": "gemini-2.5-pro",
}

PERPLEXITY_MODELS = {
    "default": "sonar-pro",
    "SONAR_PRO": "sonar-pro",
}

LLM_PROVIDERS = {
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "models": OPENAI_MODELS,
    },
    "groq": {
        "api_key": GROQ_API_KEY,
        "base_url": GROQ_BASE_URL,
        "models": GROQ_MODELS,
    },
    "gemini": {
        "api_key": GEMINI_API_KEY,
        "base_url": GEMINI_BASE_URL,
        "models": GEMINI_MODELS,
    },
    "perplexity": {
        "api_key": PERPLEXITY_API_KEY,
        "base_url": PERPLEXITY_BASE_URL,
        "models": PERPLEXITY_MODELS,
    },
}