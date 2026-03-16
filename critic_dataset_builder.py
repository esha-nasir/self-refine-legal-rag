import argparse
import json
from pathlib import Path

from critic import Critic


def load_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build critic SFT dataset in Self-RAG training format.")
    parser.add_argument("--input_file", required=True, help="JSON/JSONL logs with question, answer, contexts")
    parser.add_argument("--output_file", required=True, help="Output JSON for train_special_tokens.py")
    parser.add_argument("--max_examples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    records = load_json_or_jsonl(Path(args.input_file))
    if args.max_examples > 0:
        records = records[:args.max_examples]

    critic = Critic()  # teacher mode by default for automatic distillation

    dataset = []

    for rec in records:
        question = (rec.get("question") or rec.get("instruction") or "").strip()
        answer = (rec.get("answer") or rec.get("output") or "").strip()
        contexts = rec.get("contexts") or rec.get("sources") or []

        if not question:
            continue

        # Task 1: retrieval necessity (question only)
        r_tok = critic.need_retrieval(question)
        dataset.append(
            {
                "instruction": "Decide if retrieval is needed.",
                "input": f"Task instruction: {question}",
                "output": r_tok,
            }
        )

        # Task 2: relevance for each context
        for ctx in contexts[:5]:
            evidence = (ctx.get("text") or "").strip()
            if not evidence:
                continue
            rel_tok = critic.relevance(question, evidence)
            dataset.append(
                {
                    "instruction": "Decide if evidence is relevant to the question.",
                    "input": f"Task instruction: {question}\nEvidence: {evidence}",
                    "output": rel_tok,
                }
            )

            # Task 3: groundness of produced answer wrt evidence
            if answer:
                grd_tok = critic.groundness(question, answer, evidence)
                dataset.append(
                    {
                        "instruction": "Judge if answer is supported by evidence.",
                        "input": (
                            f"Task instruction: {question}\n"
                            f"Output: {answer}\n"
                            f"Evidence: {evidence}"
                        ),
                        "output": grd_tok,
                    }
                )

        # Task 4: utility of final answer
        if answer:
            util_tok = critic.utility(question, answer)
            dataset.append(
                {
                    "instruction": "Rate utility of answer.",
                    "input": f"Task instruction: {question}\nOutput: {answer}",
                    "output": util_tok,
                }
            )

    save_json(dataset, Path(args.output_file))
    print(f"Saved {len(dataset)} critic samples -> {args.output_file}")


if __name__ == "__main__":
    main()
