import argparse
import json
from pathlib import Path

from rag import generate_answer


def load_questions(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)

    questions = []
    for row in rows:
        if isinstance(row, str):
            questions.append({"question": row})
            continue
        q = row.get("question") or row.get("instruction")
        if q:
            questions.append(row)
    return questions


def main():
    parser = argparse.ArgumentParser(description="Run current RAG and save logs for critic training.")
    parser.add_argument("--questions_file", required=True, help="JSON/JSONL containing questions")
    parser.add_argument("--output_file", required=True, help="JSON output log file")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions(Path(args.questions_file))
    logs = []

    for item in questions:
        q = item.get("question") or item.get("instruction")
        case_no = item.get("case_no")
        file_name = item.get("file_name")

        answer, contexts, meta = generate_answer(q, top_k=args.top_k, case_no=case_no, file_name=file_name)
        logs.append(
            {
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "meta": meta,
                "case_no": case_no,
                "file_name": file_name,
            }
        )
        print(f"Logged: {q[:80]}")

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=True, indent=2)

    print(f"Saved {len(logs)} records -> {args.output_file}")


if __name__ == "__main__":
    main()
