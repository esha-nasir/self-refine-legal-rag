#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash critic_train_template.sh /abs/path/critic_train.json /abs/path/output_critic_model

TRAIN_JSON="${1:?missing train json path}"
OUT_DIR="${2:?missing output model dir}"

cd /Users/eshanasir/self-rag-main/data_creation

python3 train_special_tokens.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data_path "${TRAIN_JSON}" \
  --output_dir "${OUT_DIR}" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --bf16 True \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 2 \
  --model_max_length 1024 \
  --dataloader_pin_memory False \
  --use_special_token True
