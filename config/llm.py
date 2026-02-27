# config/llm.py
import os

def is_llm_ready() -> bool:
    """
    Verifica si OpenAI está configurado correctamente.
    """
    llm_enabled = os.getenv("LLM_ENABLE", "0") == "1"
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if not llm_enabled:
        return False
    if not api_key or api_key == "tu-api-key-aqui":
        return False
    
    return True

LLM_ENABLED = os.getenv("LLM_ENABLE", "0") == "1"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")