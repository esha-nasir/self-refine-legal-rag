# Critic-Only Fine-Tune For `singleAgentRAG`

## 1) Add env flags
Add these to your shell (or `.env` that you source):

```bash
export USE_CRITIC=1
export CRITIC_MODE=teacher
export CRITIC_FILTER_CONTEXTS=0
export CRITIC_POSTCHECK=0
export CRITIC_RETRY=0
# later, after fine-tune:
# export CRITIC_MODE=local
# export CRITIC_MODEL_PATH=/Users/eshanasir/models/critic_v1
export CRITIC_UTILITY_FLOOR=3
export CRITIC_GROUNDNESS_FLOOR=0.5
```

## 2) Prepare a question set
Create JSON or JSONL with at least:

```json
[
  {"question": "...", "case_no": "optional", "file_name": "optional"}
]
```

## 3) Log current RAG outputs (teacher data)

```bash
cd /Users/eshanasir/singleAgentRAG
python3 critic_data_logger.py \
  --questions_file /Users/eshanasir/singleAgentRAG/data/questions.json \
  --output_file /Users/eshanasir/singleAgentRAG/data/critic_logs.json \
  --top_k 5
```

Output schema per item:
- `question`
- `answer`
- `contexts`
- `meta`

## 4) Distill labels into critic SFT format

```bash
cd /Users/eshanasir/singleAgentRAG
export CRITIC_MODE=teacher
python3 critic_dataset_builder.py \
  --input_file /Users/eshanasir/singleAgentRAG/data/critic_logs.json \
  --output_file /Users/eshanasir/singleAgentRAG/data/critic_train.json
```

This produces data compatible with `self-rag-main/data_creation/train_special_tokens.py`.

## 5) Fine-tune critic model

```bash
cd /Users/eshanasir/singleAgentRAG
bash critic_train_template.sh \
  /Users/eshanasir/singleAgentRAG/data/critic_train.json \
  /Users/eshanasir/models/critic_v1
```

## 6) Switch runtime to local critic

```bash
export CRITIC_MODE=local
export CRITIC_MODEL_PATH=/Users/eshanasir/models/critic_v1
```

Your runtime now does:
1. retrieval necessity check
2. context relevance filtering
3. groundedness + utility scoring
4. one retry with broader retrieval when scores are low

## 7) Run API

```bash
cd /Users/eshanasir/singleAgentRAG
uvicorn api:app --reload --port 8000
```

POST `/ask` returns `meta` with critic decisions.

## 8) Scale-up plan (after 9-10 PDFs)
1. Increase question coverage (edge cases, ambiguous queries, citation-heavy queries).
2. Regenerate `critic_logs.json` from full corpus.
3. Rebuild `critic_train.json` and retrain (`critic_v2`).
4. Compare v1 vs v2 on a fixed holdout question set.
