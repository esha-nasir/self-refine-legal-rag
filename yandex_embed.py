import os
import hashlib
import random
import requests
from collections import OrderedDict

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

DOC_MODEL_URI = os.getenv("YANDEX_DOC_EMBED_MODEL", "text-search-doc/latest")
QUERY_MODEL_URI = os.getenv("YANDEX_QUERY_EMBED_MODEL", "text-search-query/latest")

EMBED_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
DEBUG_EMBED = os.getenv("DEBUG_EMBED", "0").strip() == "1"
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "yandex").strip().lower()
MOCK_EMBED_DIM = int(os.getenv("MOCK_EMBED_DIM", "256"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
EMBED_CACHE_ENABLED = os.getenv("EMBED_CACHE_ENABLED", "1").strip() == "1"
EMBED_CACHE_SIZE = int(os.getenv("EMBED_CACHE_SIZE", "2048"))
_EMBED_CACHE: "OrderedDict[str, list[float]]" = OrderedDict()


def _mock_embedding(text: str, dim: int = MOCK_EMBED_DIM) -> list[float]:
    # Deterministic pseudo-embedding for free local tests.
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = random.Random(seed)
    return [rng.uniform(-0.1, 0.1) for _ in range(dim)]


def _ollama_embedding(text: str) -> list[float]:
    payload = {"model": OLLAMA_EMBED_MODEL, "input": text or ""}
    response = requests.post(f"{OLLAMA_URL}/api/embed", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    embeddings = data.get("embeddings") or []
    if embeddings and isinstance(embeddings[0], list):
        return embeddings[0]

    single = data.get("embedding")
    if isinstance(single, list):
        return single

    raise RuntimeError(f"Unexpected Ollama embedding response: {data}")


def get_embedding(text: str, kind: str = "doc"):
    """
    kind: "doc" or "query"
    Uses different embedding models for better retrieval.
    If your Yandex account doesn't support text-search-query/latest,
    set YANDEX_QUERY_EMBED_MODEL to text-search-doc/latest.
    """
    if EMBED_PROVIDER == "mock":
        return _mock_embedding(text)
    if EMBED_PROVIDER == "ollama":
        text = text or ""
        cache_key = f"{kind}::{text}"
        if EMBED_CACHE_ENABLED and cache_key in _EMBED_CACHE:
            vec = _EMBED_CACHE.pop(cache_key)
            _EMBED_CACHE[cache_key] = vec
            return vec

        vec = _ollama_embedding(text)
        if EMBED_CACHE_ENABLED:
            _EMBED_CACHE[cache_key] = vec
            if len(_EMBED_CACHE) > max(1, EMBED_CACHE_SIZE):
                _EMBED_CACHE.popitem(last=False)
        return vec

    text = text or ""
    cache_key = f"{kind}::{text}"
    if EMBED_CACHE_ENABLED and cache_key in _EMBED_CACHE:
        vec = _EMBED_CACHE.pop(cache_key)
        _EMBED_CACHE[cache_key] = vec
        return vec

    if not FOLDER_ID:
        raise RuntimeError("Missing env var: YANDEX_FOLDER_ID")
    if not YANDEX_API_KEY and not YANDEX_IAM_TOKEN:
        raise RuntimeError("Set one of: YANDEX_API_KEY or YANDEX_IAM_TOKEN")

    model = DOC_MODEL_URI if kind == "doc" else QUERY_MODEL_URI
    auth_value = f"Bearer {YANDEX_IAM_TOKEN}" if YANDEX_IAM_TOKEN else f"Api-Key {YANDEX_API_KEY}"

    headers = {
        "Authorization": auth_value,
        "Content-Type": "application/json",
    }

    data = {"modelUri": f"emb://{FOLDER_ID}/{model}", "text": text}

    try:
        r = requests.post(EMBED_URL, headers=headers, json=data, timeout=60)
        r.raise_for_status()  # Will raise an error for 4xx/5xx responses
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        raise

    if DEBUG_EMBED:
        print("Response Status Code:", r.status_code)
        print("Response Body:", r.text)

    try:
        j = r.json()
    except ValueError as e:
        print(f"Error decoding JSON: {e}")
        raise

    if "embedding" in j:
        vec = j["embedding"]
        if EMBED_CACHE_ENABLED:
            _EMBED_CACHE[cache_key] = vec
            if len(_EMBED_CACHE) > max(1, EMBED_CACHE_SIZE):
                _EMBED_CACHE.popitem(last=False)
        return vec

    if "error" in j:
        raise Exception(f"Yandex API error: {j['error']}")

    raise Exception(f"Unexpected response format: {j}")
