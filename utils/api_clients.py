import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

class APIError(Exception):
    pass

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, APIError))
)
def call_openrouter(messages, model="openai/gpt-4o-mini", temperature=0.7):
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=120
    )
    
    if not response.ok:
        raise APIError(f"OpenRouter HTTP {response.status_code}: {response.text[:300]}")
        
    return response.json()

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, APIError))
)
def get_voyage_embeddings(texts, model="voyage-4-large"):
    if not VOYAGE_API_KEY:
        raise ValueError("VOYAGE_API_KEY is not set.")
        
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "input": texts,
        "model": model
    }
    
    response = requests.post(
        url="https://api.voyageai.com/v1/embeddings",
        headers=headers,
        json=data,
        timeout=120
    )
    
    if not response.ok:
        raise APIError(f"Voyage HTTP {response.status_code}: {response.text[:300]}")
        
    result = response.json()
    return [item["embedding"] for item in result["data"]]
