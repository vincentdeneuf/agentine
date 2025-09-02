import os
from dotenv import load_dotenv
load_dotenv()

DEFAULT_LLM_PROVIDER = "openai"

OPENAI = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "base_url": None,
    "default_model": "gpt-5-chat-latest",
}

GROQ = {
    "api_key": os.getenv("GROQ_API_KEY", ""),
    "base_url": "https://api.groq.com/openai/v1",
    "default_model": "llama-3.3-70b-versatile",
}

GOOGLE = {
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "default_model": "gemini-2.5-flash-lite",
}

PERPLEXITY = {
    "api_key": os.getenv("PERPLEXITY_API_KEY", ""),
    "base_url": "https://api.perplexity.ai",
    "default_model": "sonar",
}

ANTHROPIC = {
    "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "base_url": "https://api.anthropic.com",
    "default_model": "claude-4-sonnet-latest",
}

XAI = {
    "api_key": os.getenv("XAI_API_KEY", ""),
    "base_url": "https://api.x.ai/v1",
    "default_model": "grok-3-mini",
}

DEEPSEEK = {
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "base_url": "https://api.deepseek.com",
    "default_model": "deepseek-chat",
}

MISTRAL = {
    "api_key": os.getenv("MISTRAL_API_KEY", ""),
    "base_url": "https://api.mistral.ai/v1",
    "default_model": "mistral-medium",
}

COHERE = {
    "api_key": os.getenv("COHERE_API_KEY", ""),
    "base_url": "https://api.cohere.ai/compatibility/v1",
    "default_model": "command-r",
}

KNOWN_LLM_PROVIDERS = {
    "openai": OPENAI,
    "groq": GROQ,
    "google": GOOGLE,
    "perplexity": PERPLEXITY,
    "anthropic": ANTHROPIC,
    "xai": XAI,
    "deepseek": DEEPSEEK,
    "mistral": MISTRAL,
    "cohere": COHERE,
}