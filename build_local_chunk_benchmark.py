import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from chunking import chunk_text


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def first_sentence(text: str, fallback_words: int = 35) -> str:
    clean = normalize_ws(text)
    if not clean:
        return ""
    match = re.search(r"(.+?[.!?])(\s|$)", clean)
    if match:
        sent = match.group(1).strip()
        if len(sent) >= 30:
            return sent
    words = clean.split()
    return " ".join(words[:fallback_words]).strip()


def middle_phrase(text: str, span_words: int = 14) -> str:
    words = normalize_ws(text).split()
    if not words:
        return ""
    if len(words) <= span_words:
        return " ".join(words)
    start = max(0, (len(words) // 2) - (span_words // 2))
    return " ".join(words[start : start + span_words])


def quality_flags(text: str) -> list[str]:
    clean = normalize_ws(text)
    flags: list[str] = []

    if len(clean) < 120:
        flags.append("too_short")
    if "�" in clean:
        flags.append("replacement_char")

    total_chars = max(1, len(clean))
    alpha_chars = sum(ch.isalpha() for ch in clean)
    if alpha_chars / total_chars < 0.45:
        flags.append("low_alpha_ratio")

    weird_chars = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:!?()[]{}'\"-_/&%") for ch in clean)
    if weird_chars / total_chars > 0.12:
        flags.append("high_symbol_noise")

    return flags


def read_pdf_text_local(pdf_path: Path) -> str:
    text_chunks: list[str] = []

    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install pypdf or PyPDF2 for local PDF extraction.") from exc

    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = normalize_ws(page_text)
        if page_text:
            text_chunks.append(page_text)

    return "\n".join(text_chunks).strip()


def load_metadata_map(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip", sep=None)
    df.columns = [str(c).strip() for c in df.columns]

    meta_map: dict[str, dict] = {}
    if "temp_link" not in df.columns:
        return meta_map

    for _, row in df.iterrows():
        link = str(row.get("temp_link", "")).strip()
        if not link or link.lower() == "nan":
            continue
        fname = Path(link).name
        if not fname.lower().endswith(".pdf"):
            fname += ".pdf"
        meta_map[fname] = {
            "case_no": str(row.get("case_no", "")).strip(),
            "pet": str(row.get("pet", "")).strip(),
            "res": str(row.get("res", "")).strip(),
            "judgment_dates": str(row.get("judgment_dates", "")).strip(),
            "temp_link": link,
        }
    return meta_map


def iter_pdfs(pdf_folder: Path):
    for p in sorted(pdf_folder.rglob("*.pdf")):
        if p.is_file():
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark drafts from local chunk source (no Pinecone).")
    parser.add_argument("--env-file", default=str(Path(__file__).resolve().parent / ".env"))
    parser.add_argument("--pdf-folder", default="")
    parser.add_argument("--csv-path", default="")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "data" / "local_chunk_benchmark"))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--max-pdfs", type=int, default=0, help="0 = all")
    parser.add_argument("--max-retrieval-samples", type=int, default=200)
    parser.add_argument("--max-qa-samples", type=int, default=200)
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    pdf_folder = Path(args.pdf_folder or os.getenv("PDF_FOLDER", "")).expanduser()
    csv_path = Path(args.csv_path or os.getenv("CSV_PATH", "")).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_folder.exists():
        raise RuntimeError(f"PDF folder not found: {pdf_folder}")

    metadata_map = load_metadata_map(csv_path) if csv_path.exists() else {}

    all_chunks_path = out_dir / "all_chunks.jsonl"
    audit_path = out_dir / "chunk_audit.jsonl"
    retrieval_path = out_dir / "retrieval_benchmark.jsonl"
    qa_path = out_dir / "qa_draft_benchmark.jsonl"
    summary_path = out_dir / "summary.json"

    pdf_count = 0
    chunk_count = 0
    bad_count = 0
    good_rows: list[dict] = []

    with (
        all_chunks_path.open("w", encoding="utf-8") as all_f,
        audit_path.open("w", encoding="utf-8") as audit_f,
    ):
        for pdf_path in iter_pdfs(pdf_folder):
            if args.max_pdfs > 0 and pdf_count >= args.max_pdfs:
                break
            pdf_count += 1
            try:
                text = read_pdf_text_local(pdf_path)
            except Exception as exc:
                rec = {"file_name": pdf_path.name, "error": str(exc), "type": "extract_error"}
                audit_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            if not text:
                rec = {"file_name": pdf_path.name, "error": "empty_text", "type": "extract_error"}
                audit_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            rel_folder = str(pdf_path.parent.relative_to(pdf_folder))
            base_meta = metadata_map.get(pdf_path.name, {})
            chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)

            for i, ch in enumerate(chunks):
                chunk_id = f"{rel_folder}_{pdf_path.name}_{i}"
                clean = normalize_ws(ch)
                flags = quality_flags(clean)
                row = {
                    "id": chunk_id,
                    "file_name": pdf_path.name,
                    "folder": rel_folder,
                    "chunk_id": i,
                    "text": clean,
                    "text_len": len(clean),
                    "case_no": base_meta.get("case_no", ""),
                    "pet": base_meta.get("pet", ""),
                    "res": base_meta.get("res", ""),
                    "judgment_dates": base_meta.get("judgment_dates", ""),
                    "flags": flags,
                    "is_bad": bool(flags),
                }
                all_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                audit_f.write(
                    json.dumps(
                        {
                            "id": chunk_id,
                            "file_name": pdf_path.name,
                            "chunk_id": i,
                            "text_len": len(clean),
                            "flags": flags,
                            "is_bad": bool(flags),
                            "preview": clean[:220],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                chunk_count += 1
                if flags:
                    bad_count += 1
                else:
                    good_rows.append(row)

    retrieval_n = 0
    with retrieval_path.open("w", encoding="utf-8") as f:
        for row in good_rows:
            if retrieval_n >= args.max_retrieval_samples:
                break
            phrase = middle_phrase(row["text"], span_words=14)
            if len(phrase.split()) < 8:
                continue
            rec = {
                "id": f"retrieval_{retrieval_n+1}",
                "query": f'Find the source chunk containing this legal statement: "{phrase}"',
                "gold_chunk_id": row["id"],
                "gold_file_name": row["file_name"],
                "benchmark_type": "retrieval",
                "source_quote": phrase,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            retrieval_n += 1

    qa_n = 0
    with qa_path.open("w", encoding="utf-8") as f:
        for row in good_rows:
            if qa_n >= args.max_qa_samples:
                break
            ans = first_sentence(row["text"])
            if len(ans) < 30:
                continue
            anchor = middle_phrase(row["text"], span_words=10)
            question = f'In {row["file_name"]}, what is stated regarding "{anchor}"?'
            rec = {
                "id": f"qa_{qa_n+1}",
                "question": question,
                "gold_answer": ans,
                "gold_sources": [row["id"]],
                "file_name": row["file_name"],
                "chunk_id": row["chunk_id"],
                "case_no": row["case_no"],
                "benchmark_type": "qa_draft",
                "needs_manual_review": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            qa_n += 1

    summary = {
        "pdf_folder": str(pdf_folder),
        "csv_path": str(csv_path) if csv_path.exists() else "",
        "pdfs_processed": pdf_count,
        "chunks_total": chunk_count,
        "chunks_good": len(good_rows),
        "chunks_bad": bad_count,
        "bad_rate": round((bad_count / chunk_count), 4) if chunk_count else 0.0,
        "retrieval_benchmark_samples": retrieval_n,
        "qa_draft_benchmark_samples": qa_n,
        "files": {
            "all_chunks": str(all_chunks_path),
            "chunk_audit": str(audit_path),
            "retrieval_benchmark": str(retrieval_path),
            "qa_draft_benchmark": str(qa_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
