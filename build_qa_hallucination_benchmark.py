import argparse
import hashlib
import json
import re
from pathlib import Path


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


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


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_ws(text).lower()))


def overlap_score(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def choose_distractor(row: dict, chunks: list[dict], by_id: dict[str, dict]) -> dict | None:
    gold_ids = [str(x) for x in row.get("gold_sources", []) if str(x).strip()]
    if not gold_ids:
        return None

    gold_chunk = by_id.get(gold_ids[0])
    if not gold_chunk:
        return None

    question = str(row.get("question", ""))
    gold_text = str(gold_chunk.get("text", ""))
    current_file = str(row.get("file_name", "")).strip()

    candidates: list[tuple[tuple[int, float, float], dict]] = []
    for chunk in chunks:
        cid = str(chunk.get("id", "")).strip()
        if not cid or cid in gold_ids:
            continue

        distractor_answer = first_sentence(str(chunk.get("text", "")))
        if len(distractor_answer) < 30:
            continue

        file_name = str(chunk.get("file_name", "")).strip()
        same_file_penalty = 1 if file_name == current_file else 0
        question_overlap = overlap_score(question, distractor_answer)
        gold_overlap = overlap_score(gold_text, str(chunk.get("text", "")))

        # Prefer other files and low-overlap chunks so the negative example is unsupported.
        sort_key = (same_file_penalty, question_overlap, gold_overlap)
        candidates.append((sort_key, chunk))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    shortlist = candidates[: min(5, len(candidates))]
    row_key = str(row.get("id", "")) or str(row.get("question", ""))
    pick_seed = int(hashlib.sha256(row_key.encode("utf-8")).hexdigest(), 16)
    return shortlist[pick_seed % len(shortlist)][1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a hallucination benchmark from qa_draft_benchmark.jsonl."
    )
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--benchmark-dir",
        default=str(root / "data" / "local_chunk_benchmark"),
    )
    parser.add_argument(
        "--qa-file",
        default="qa_draft_benchmark.jsonl",
    )
    parser.add_argument(
        "--chunks-file",
        default="all_chunks.jsonl",
    )
    parser.add_argument(
        "--output-file",
        default="qa_hallucination_benchmark.jsonl",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    qa_rows = load_jsonl(benchmark_dir / args.qa_file)
    chunk_rows = load_jsonl(benchmark_dir / args.chunks_file)
    chunks_by_id = {str(row.get("id", "")).strip(): row for row in chunk_rows}

    out_path = benchmark_dir / args.output_file
    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for row in qa_rows:
            gold_ids = [str(x) for x in row.get("gold_sources", []) if str(x).strip()]
            if not gold_ids:
                skipped += 1
                continue

            gold_chunk = chunks_by_id.get(gold_ids[0])
            if not gold_chunk:
                skipped += 1
                continue

            distractor = choose_distractor(row, chunk_rows, chunks_by_id)
            if not distractor:
                skipped += 1
                continue

            grounded_answer = normalize_ws(str(row.get("gold_answer", "")))
            hallucinated_answer = first_sentence(str(distractor.get("text", "")))
            if not grounded_answer or not hallucinated_answer:
                skipped += 1
                continue

            rec = {
                "id": str(row.get("id", f"qa_hallu_{written + 1}")),
                "question": str(row.get("question", "")),
                "grounded_answer": grounded_answer,
                "hallucinated_answer": hallucinated_answer,
                "gold_sources": gold_ids,
                "gold_file_name": str(row.get("file_name", "")),
                "gold_chunk_id": row.get("chunk_id"),
                "gold_chunk_text": normalize_ws(str(gold_chunk.get("text", ""))),
                "hallucinated_source_id": str(distractor.get("id", "")),
                "hallucinated_file_name": str(distractor.get("file_name", "")),
                "hallucinated_chunk_id": distractor.get("chunk_id"),
                "hallucinated_chunk_text": normalize_ws(str(distractor.get("text", ""))),
                "benchmark_type": "qa_hallucination",
                "needs_manual_review": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "input_qa_rows": len(qa_rows),
        "input_chunk_rows": len(chunk_rows),
        "written_rows": written,
        "skipped_rows": skipped,
        "output_file": str(out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
