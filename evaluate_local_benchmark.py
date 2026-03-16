import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from rag import generate_answer


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def token_f1(a: str, b: str) -> float:
    ta = normalize(a).split()
    tb = normalize(b).split()
    if not ta or not tb:
        return 0.0
    sa = set(ta)
    sb = set(tb)
    common = len(sa & sb)
    if common == 0:
        return 0.0
    p = common / len(sa)
    r = common / len(sb)
    return 2 * p * r / (p + r)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


@dataclass
class EvalResult:
    total: int
    correct: int
    parse_failures: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def parse_chunk_suffix(vector_id: str) -> int | None:
    m = re.search(r"_(\d+)$", vector_id or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def evaluate_retrieval(rows: list[dict], top_k: int) -> tuple[EvalResult, EvalResult]:
    total = 0
    correct_strict = 0
    correct_soft = 0
    parse_failures = 0

    for row in rows:
        total += 1
        q = row.get("query", "")
        gold_file = str(row.get("gold_file_name", "")).strip()
        # In this benchmark file, gold_chunk_id is the full vector ID
        # e.g. "._5_2021_...pdf_2"
        gold_vector_id = str(row.get("gold_chunk_id", "")).strip()

        answer, contexts, _ = generate_answer(q, top_k=top_k)
        if not answer.strip():
            parse_failures += 1
            continue

        strict_hit = False
        soft_hit = False
        gold_chunk_idx = parse_chunk_suffix(gold_vector_id)
        for c in contexts:
            ctx_vector_id = str(c.get("id", "")).strip()
            file_name = str(c.get("file_name", "")).strip()
            # Backward-compatible fallback if IDs are absent.
            chunk_id = str(c.get("chunk_id", "")).strip()
            fallback_id = f"._{file_name}_{chunk_id}" if file_name and chunk_id else ""

            if (
                (gold_vector_id and ctx_vector_id == gold_vector_id)
                or (gold_vector_id and fallback_id == gold_vector_id)
                or (not gold_vector_id and file_name == gold_file)
            ):
                strict_hit = True
                soft_hit = True
                break

            # Soft hit: same file and chunk index within +/-1
            if file_name == gold_file and gold_chunk_idx is not None:
                try:
                    cidx = int(chunk_id)
                except Exception:
                    cidx = None
                if cidx is not None and abs(cidx - gold_chunk_idx) <= 1:
                    soft_hit = True

        if strict_hit:
            correct_strict += 1
        if soft_hit:
            correct_soft += 1

    return (
        EvalResult(total=total, correct=correct_strict, parse_failures=parse_failures),
        EvalResult(total=total, correct=correct_soft, parse_failures=parse_failures),
    )


def evaluate_qa(rows: list[dict], top_k: int, f1_threshold: float) -> EvalResult:
    total = 0
    correct = 0
    parse_failures = 0

    for row in rows:
        total += 1
        q = row.get("question", "")
        gold = row.get("gold_answer", "")

        answer, _, _ = generate_answer(q, top_k=top_k)
        ans = normalize(answer)
        g = normalize(gold)
        if not ans:
            parse_failures += 1
            continue

        # Light-weight automatic scoring for draft golds.
        ok = False
        if g and (g[:120] in ans or ans[:120] in g):
            ok = True
        elif token_f1(ans, g) >= f1_threshold:
            ok = True

        if ok:
            correct += 1

    return EvalResult(total=total, correct=correct, parse_failures=parse_failures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selfRefineAgentRAG on local chunk benchmark.")
    parser.add_argument(
        "--benchmark-dir",
        default="/Users/eshanasir/selfRefineAgentRAG/data/local_chunk_benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/eshanasir/selfRefineAgentRAG/eval_outputs_local_benchmark",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qa-f1-threshold", type=float, default=0.35)
    args = parser.parse_args()
    provider = os.getenv("LLM_PROVIDER", "yandex").strip().lower()
    if provider == "mock":
        print(
            '[WARN] LLM_PROVIDER="mock" is active. Answers are synthetic placeholders, '
            "so QA accuracy will not be meaningful."
        )

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows = load_jsonl(benchmark_dir / "retrieval_benchmark.jsonl")
    qa_rows = load_jsonl(benchmark_dir / "qa_draft_benchmark.jsonl")

    if args.max_samples > 0:
        retrieval_rows = retrieval_rows[: args.max_samples]
        qa_rows = qa_rows[: args.max_samples]

    retrieval_strict, retrieval_soft = evaluate_retrieval(retrieval_rows, top_k=args.top_k)
    qa_result = evaluate_qa(qa_rows, top_k=args.top_k, f1_threshold=args.qa_f1_threshold)

    summary = {
        "retrieval_strict": {
            "total": retrieval_strict.total,
            "correct": retrieval_strict.correct,
            "accuracy": round(retrieval_strict.accuracy, 4),
            "parse_failures": retrieval_strict.parse_failures,
        },
        "retrieval_soft": {
            "total": retrieval_soft.total,
            "correct": retrieval_soft.correct,
            "accuracy": round(retrieval_soft.accuracy, 4),
            "parse_failures": retrieval_soft.parse_failures,
        },
        "qa_draft": {
            "total": qa_result.total,
            "correct": qa_result.correct,
            "accuracy": round(qa_result.accuracy, 4),
            "parse_failures": qa_result.parse_failures,
        },
    }

    (output_dir / "local_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
