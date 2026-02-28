"""OpenRouter LLM service for Karma AI with Groq as priority provider."""

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# Provider routing: prioritize Groq, fallback to other providers
PROVIDER_PREFERENCES = {
    "order": ["Groq"],
    "allow_fallbacks": True,
}

_session = requests.Session()
_session.headers.update({
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://karma-ai.app",
    "X-Title": "Karma AI",
})


def chat_completion(messages: list[dict], temperature: float = 0.7) -> str:
    """Generate a chat response using OpenRouter with Groq as priority provider.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        temperature: Sampling temperature (0-2).

    Returns:
        Assistant's response text.
    """
    url = f"{OPENROUTER_BASE_URL}/chat/completions"

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2000,
        "provider": PROVIDER_PREFERENCES,
    }

    resp = _session.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]


def chat_completion_streaming(messages: list[dict], temperature: float = 0.7):
    """Stream a chat response, yielding text tokens as they arrive.

    Uses OpenRouter SSE streaming with Groq as priority provider.
    Each yield is a small text chunk (typically a single token).

    Args:
        messages: List of message dicts with 'role' and 'content'.
        temperature: Sampling temperature (0-2).

    Yields:
        str: Text tokens/chunks as they are generated.
    """
    url = f"{OPENROUTER_BASE_URL}/chat/completions"

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2000,
        "stream": True,
        "provider": PROVIDER_PREFERENCES,
    }

    resp = _session.post(url, json=payload, stream=True, timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]  # strip "data: " prefix
        if data_str.strip() == "[DONE]":
            break
        try:
            data = json.loads(data_str)
            delta = data.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
        except (json.JSONDecodeError, IndexError, KeyError):
            continue
