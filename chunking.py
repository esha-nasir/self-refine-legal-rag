import re


def _normalize_paragraphs(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    raw_parts = re.split(r"\n\s*\n+", text)
    paragraphs = []
    for part in raw_parts:
        cleaned = re.sub(r"[ \t]+", " ", part).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80):
    """
    Paragraph-aware chunking for legal text.

    `chunk_size` and `overlap` are approximate word counts.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    paragraphs = _normalize_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_parts and current_words + para_words > chunk_size:
            chunks.append("\n\n".join(current_parts).strip())

            overlap_parts: list[str] = []
            overlap_words = 0
            for existing in reversed(current_parts):
                existing_words = len(existing.split())
                if overlap_parts and overlap_words + existing_words > overlap:
                    break
                overlap_parts.insert(0, existing)
                overlap_words += existing_words

            current_parts = overlap_parts[:]
            current_words = sum(len(part.split()) for part in current_parts)

        current_parts.append(para)
        current_words += para_words

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return chunks
