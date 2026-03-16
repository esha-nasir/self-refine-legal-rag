import argparse
import json
from pathlib import Path

from retrieve import retrieve


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


def evaluate(rows: list[dict], k: int) -> dict:
    total = 0
    hit_count = 0
    precision_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0

    for row in rows:
        query = str(row.get("query", "")).strip()
        gold_id = str(row.get("gold_chunk_id", "")).strip()
        if not query or not gold_id:
            continue

        total += 1
        results = retrieve(query, top_k=k)

        rank = None
        for idx, result in enumerate(results, start=1):
            result_id = str(result.get("id", "")).strip()
            if result_id == gold_id:
                rank = idx
                break

        if rank is not None:
            hit_count += 1
            recall_sum += 1.0
            precision_sum += 1.0 / k
            mrr_sum += 1.0 / rank

    if total == 0:
        return {
            "total": 0,
            "k": k,
            "hit_at_k": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
        }

    return {
        "total": total,
        "k": k,
        "hit_at_k": round(hit_count / total, 4),
        "precision_at_k": round(precision_sum / total, 4),
        "recall_at_k": round(recall_sum / total, 4),
        "mrr": round(mrr_sum / total, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval Precision@k / Recall@k / MRR.")
    parser.add_argument(
        "--benchmark-file",
        default="/Users/eshanasir/selfRefineAgentRAG/data/local_chunk_benchmark/retrieval_benchmark.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.benchmark_file))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    metrics = evaluate(rows, k=args.top_k)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
