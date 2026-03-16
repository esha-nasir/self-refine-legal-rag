from yandex_embed import get_embedding


def _get_index():
    # Lazy import so modes that skip retrieval (e.g., benchmark/no_retrieval)
    # don't require Pinecone to initialize at process start.
    from pinecone_setup import index

    return index

def retrieve(query: str, top_k: int = 5, case_no: str | None = None, file_name: str | None = None):
    """
    Retrieve relevant chunks from Pinecone with optional metadata filtering.
    """
    query_emb = get_embedding(query, kind="query")

    pinecone_filter = None
    if case_no and file_name:
        pinecone_filter = {"$and": [{"case_no": {"$eq": case_no}}, {"file_name": {"$eq": file_name}}]}
    elif case_no:
        pinecone_filter = {"case_no": {"$eq": case_no}}
    elif file_name:
        pinecone_filter = {"file_name": {"$eq": file_name}}

    index = _get_index()
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True,
        include_values=True,
        filter=pinecone_filter,   # ✅ key change
    )

    contexts = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {}) or {}

        # Fetch embedding along with the text and other metadata
        chunk = {
            "text": metadata.get("text", ""),
            "file_name": metadata.get("file_name", ""),
            "folder": metadata.get("folder", ""),
            "chunk_id": metadata.get("chunk_id", None),
            "diary_no": metadata.get("diary_no", ""),
            "judgement_type": metadata.get("judgement_type", ""),
            "case_no": metadata.get("case_no", ""),
            "pet": metadata.get("pet", ""),
            "res": metadata.get("res", ""),
            "pet_adv": metadata.get("pet_adv", ""),
            "res_adv": metadata.get("res_adv", ""),
            "bench": metadata.get("bench", ""),
            "judgement_by": metadata.get("judgement_by", ""),
            "judgment_dates": metadata.get("judgment_dates", ""),
            "url": metadata.get("url", ""),
            "score": match.get("score", None),
            "id": match.get("id", None),
            # Ensure embedding is part of the context
            "embedding": match.get("values", []),  # Ensure embedding is present
        }

        contexts.append(chunk)

    return contexts
