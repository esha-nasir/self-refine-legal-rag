import os
import time
import requests
from collections import OrderedDict

COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "yandex").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct").strip()
POLLINATIONS_URL = os.getenv("POLLINATIONS_URL", "https://text.pollinations.ai").rstrip("/")
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "openai").strip()
POLLINATIONS_RETRIES = int(os.getenv("POLLINATIONS_RETRIES", "3"))
COMPLETION_CACHE_ENABLED = os.getenv("COMPLETION_CACHE_ENABLED", "1").strip() == "1"
COMPLETION_CACHE_SIZE = int(os.getenv("COMPLETION_CACHE_SIZE", "1024"))
_COMPLETION_CACHE: "OrderedDict[str, str]" = OrderedDict()


def _mock_complete(prompt: str) -> str:
    text = (prompt or "").lower()
    # Classifier-style prompts.
    if "output only one word: yes or no" in text:
        return "No"
    # Self-RAG segment JSON generation path.
    if "\"segment\"" in text and "\"done\"" in text:
        return "{\"segment\":\"Retrieval/context is required.\",\"done\":true}"
    # Generic fallback.
    return "Retrieval/context is required."


def _ollama_complete(prompt: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def _pollinations_complete(prompt: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": POLLINATIONS_MODEL,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    last_error = None
    for attempt in range(1, max(1, POLLINATIONS_RETRIES) + 1):
        try:
            response = requests.post(
                f"{POLLINATIONS_URL}/openai",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message", {}) or {}
            return str(message.get("content", "")).strip()
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max(1, POLLINATIONS_RETRIES):
                break
            time.sleep(min(8, attempt * 2))
    if last_error is not None:
        raise last_error
    return ""


def get_yandex_config():
    api_key = os.getenv("YANDEX_API_KEY")
    iam_token = os.getenv("YANDEX_IAM_TOKEN")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    model_uri = os.getenv("YANDEX_GPT_MODEL", "yandexgpt/latest")
    if not folder_id:
        raise RuntimeError("Missing env var: YANDEX_FOLDER_ID")
    if not api_key and not iam_token:
        raise RuntimeError("Set one of: YANDEX_API_KEY or YANDEX_IAM_TOKEN")
    return api_key, iam_token, folder_id, model_uri


def yandex_complete(prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> str:
    prompt = prompt or ""
    cache_key = f"{LLM_PROVIDER}::{temperature}::{max_tokens}::{prompt}"
    if COMPLETION_CACHE_ENABLED and cache_key in _COMPLETION_CACHE:
        ans = _COMPLETION_CACHE.pop(cache_key)
        _COMPLETION_CACHE[cache_key] = ans
        return ans

    if LLM_PROVIDER == "mock":
        out = _mock_complete(prompt)
        if COMPLETION_CACHE_ENABLED:
            _COMPLETION_CACHE[cache_key] = out
        return out
    if LLM_PROVIDER == "ollama":
        out = _ollama_complete(prompt, temperature=temperature, max_tokens=max_tokens)
        if COMPLETION_CACHE_ENABLED:
            _COMPLETION_CACHE[cache_key] = out
        return out
    if LLM_PROVIDER == "pollinations":
        out = _pollinations_complete(prompt, temperature=temperature, max_tokens=max_tokens)
        if COMPLETION_CACHE_ENABLED:
            _COMPLETION_CACHE[cache_key] = out
        return out

    api_key, iam_token, folder_id, model_uri = get_yandex_config()
    # Prefer IAM token when present to avoid stale API key conflicts in shell env.
    auth_value = f"Bearer {iam_token}" if iam_token else f"Api-Key {api_key}"
    headers = {
        "Authorization": auth_value,
        "Content-Type": "application/json",
    }
    data = {
        "modelUri": f"gpt://{folder_id}/{model_uri}",
        "completionOptions": {"temperature": temperature, "maxTokens": max_tokens},
        "messages": [{"role": "user", "text": prompt}],
    }
    response = requests.post(COMPLETION_URL, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    payload = response.json()
    out = payload["result"]["alternatives"][0]["message"]["text"]
    if COMPLETION_CACHE_ENABLED:
        _COMPLETION_CACHE[cache_key] = out
        if len(_COMPLETION_CACHE) > max(1, COMPLETION_CACHE_SIZE):
            _COMPLETION_CACHE.popitem(last=False)
    return out
