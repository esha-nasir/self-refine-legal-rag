import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_GENERATE_ANSWER = None
YES_NO_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)


def parse_yes_no(text: str) -> str:
    match = YES_NO_RE.search(text or "")
    if not match:
        return ""
    return match.group(1).capitalize()


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


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
            if "=" not in line:
                continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_generate_answer():
    global _GENERATE_ANSWER
    if _GENERATE_ANSWER is None:
        from rag import generate_answer

        _GENERATE_ANSWER = generate_answer
    return _GENERATE_ANSWER


def detector_prompt(question: str, candidate_answer: str, evidence: str) -> str:
    return (
        "You are a hallucination detector for legal QA.\n"
        "Use only the provided evidence.\n"
        "Output only one word: Yes or No.\n"
        "Yes = the candidate answer contains hallucination, contradiction, or unsupported facts.\n"
        "No = the candidate answer is supported by the evidence.\n\n"
        f"Question: {question}\n"
        f"Candidate Answer: {candidate_answer}\n"
        f"Evidence: {evidence}\n\n"
        "Output:"
    )


def call_rag(prompt: str, top_k: int = 1) -> str:
    generate_answer = get_generate_answer()
    answer, _, _ = generate_answer(prompt, top_k=top_k)
    return answer


@dataclass
class EvalResult:
    total: int
    correct: int
    parse_failures: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def evaluate(rows: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    grounded_total = 0
    grounded_correct = 0
    hallucinated_total = 0
    hallucinated_correct = 0
    parse_failures = 0

    for row in rows:
        question = str(row.get("question", ""))
        evidence = str(row.get("gold_chunk_text", "")).strip()

        pairs = [
            ("grounded_answer", "No"),
            ("hallucinated_answer", "Yes"),
        ]

        for field, truth in pairs:
            candidate = str(row.get(field, "")).strip()
            if not candidate:
                continue

            prompt = detector_prompt(question, candidate, evidence)
            model_output = call_rag(prompt, top_k=1)
            pred = parse_yes_no(model_output)
            if not pred:
                parse_failures += 1
                pred = ""

            y_true.append(truth)
            y_pred.append(pred)

            if truth == "No":
                grounded_total += 1
                if pred == truth:
                    grounded_correct += 1
            else:
                hallucinated_total += 1
                if pred == truth:
                    hallucinated_correct += 1

            records.append(
                {
                    "id": row.get("id", ""),
                    "pair_type": field,
                    "question": question,
                    "ground_truth": truth,
                    "prediction": pred,
                    "raw_output": model_output,
                }
            )

    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    result = EvalResult(total=len(y_true), correct=correct, parse_failures=parse_failures)

    predictions_path = output_dir / "qa_hallucination_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "overall": {
            "total": result.total,
            "correct": result.correct,
            "accuracy": round(result.accuracy, 4),
            "parse_failures": result.parse_failures,
        },
        "grounded_answer": {
            "total": grounded_total,
            "correct": grounded_correct,
            "accuracy": round((grounded_correct / grounded_total), 4) if grounded_total else 0.0,
        },
        "hallucinated_answer": {
            "total": hallucinated_total,
            "correct": hallucinated_correct,
            "accuracy": round((hallucinated_correct / hallucinated_total), 4) if hallucinated_total else 0.0,
        },
    }

    summary_path = output_dir / "qa_hallucination_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate selfRefineAgentRAG on qa_hallucination_benchmark.jsonl."
    )
    parser.add_argument(
        "--benchmark-file",
        default=str(ROOT / "data" / "local_chunk_benchmark" / "qa_hallucination_benchmark.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "eval_outputs_qa_hallucination"),
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    args = parser.parse_args()

    load_env_file(Path(args.env_file))

    # Cheap defaults for classifier-style evaluation.
    os.environ.setdefault("SELF_RAG_MODE", "no_retrieval")
    os.environ.setdefault("USE_CRITIC", "0")

    rows = load_jsonl(Path(args.benchmark_file))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    summary = evaluate(rows, Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
