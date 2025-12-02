#!/bin/bash
set -e

# ========== MUSE Data Configuration ==========
CORPUS="books"
FORGET="./data/$CORPUS/raw/forget.txt"
RETAIN="./data/$CORPUS/raw/retain1.txt"

# ========== Model Configuration ==========
TARGET_DIR="muse-bench/MUSE-Books_target"  # Fine-tuned model path
LLAMA_DIR="NousResearch/Llama-2-7b-chat-hf"

# ========== Dataset Configuration ==========
DATA_FILE="$FORGET"  # Use forget.txt for factor fitting
MAX_LENGTH=4096
ADD_BOS_TOKEN=true

# ========== Factor Configuration ==========
FACTOR_STRATEGY="ekfac"  # "ekfac", "kfac", or "diagonal"
TRAIN_BATCH_SIZE=8
OUTPUT_DIR="./kronfluence_factors"

# ========== Script Path Setup ==========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # MUSE 디렉토리로 이동
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# ========== GPU / Memory Configuration ==========
export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=18765
export NPROC_PER_NODE=1  # GPU 개수에 맞게 조정
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"

echo "🚀 Starting Kronfluence factor fitting (MUSE style)..."
echo "Model: ${TARGET_DIR}"
echo "Tokenizer: ${LLAMA_DIR}"
echo "Data File: ${DATA_FILE}"
echo "Max Length: ${MAX_LENGTH}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} | Processes: ${NPROC_PER_NODE} | Port: ${MASTER_PORT}"

CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} \
  if/fit_factor.py \
  --model_name \"${TARGET_DIR}\" \
  --tokenizer_name \"${LLAMA_DIR}\" \
  --data_file \"${DATA_FILE}\" \
  --max_length ${MAX_LENGTH} \
  --factor_strategy \"${FACTOR_STRATEGY}\" \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --output_dir \"${OUTPUT_DIR}\" \
  --covariance_module_partitions 4 \
  --lambda_module_partitions 4 \
  --covariance_data_partitions 4 \
  --lambda_data_partitions 4"

# ========== Run Factor Fitting ==========
eval $CMD
