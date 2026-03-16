import os

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from rag import generate_answer
from yandex_embed import get_embedding

app = FastAPI()

RETURN_QUERY_EMBEDDING = os.getenv("RETURN_QUERY_EMBEDDING", "0").strip() == "1"
RETURN_FULL_SOURCES = os.getenv("RETURN_FULL_SOURCES", "0").strip() == "1"
SOURCE_TEXT_MAX_CHARS = int(os.getenv("SOURCE_TEXT_MAX_CHARS", "800"))

class Query(BaseModel):
    question: str
    case_no: Optional[str] = None
    file_name: Optional[str] = None


def _build_case_label(src: dict) -> str:
    pet = (src.get("pet") or "").strip()
    res = (src.get("res") or "").strip()
    file_name = (src.get("file_name") or "").strip()
    if pet and res:
        return f"{pet} vs {res}"
    return file_name or "Unknown Source"


def _short_text(text: str, max_chars: int) -> str:
    clean = (text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def _compact_source(src: dict, idx: int) -> dict:
    return {
        "source_id": idx,
        "case": _build_case_label(src),
        "case_no": src.get("case_no", ""),
        "date": src.get("judgment_dates", ""),
        "chunk_id": src.get("chunk_id", None),
        "url": src.get("url", ""),
        "score": src.get("score", None),
        "text": _short_text(src.get("text", ""), SOURCE_TEXT_MAX_CHARS),
    }

@app.post("/ask")
def ask(q: Query):
    try:
        result = generate_answer(
            q.question,
            top_k=5,
            case_no=q.case_no,
            file_name=q.file_name,
        )
        if len(result) == 3:
            answer_text, sources, meta = result
        else:
            answer_text, sources = result
            meta = {}

        response = {"answer": answer_text, "meta": meta}

        if RETURN_FULL_SOURCES:
            response["sources"] = sources
        else:
            response["sources"] = [_compact_source(src, i + 1) for i, src in enumerate(sources)]

        if RETURN_QUERY_EMBEDDING:
            response["embedding"] = get_embedding(q.question)

        return response

    except Exception as e:
        return {"error": str(e)}
