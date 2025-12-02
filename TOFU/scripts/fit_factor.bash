#!/bin/bash
set -e

# ========== Model and Data Configuration ==========
MODEL_NAME="//root/memorization_unlearn_no_logit/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-312"
MODEL_FAMILY="llama2-7b"  # "llama2-7b" or "pythia"
DATA_PATH="locuslab/TOFU"
SPLIT="full"  # "full", "forget10", "retain90", etc.
# QUESTION_KEY and MAX_LENGTH are now read from defaults/model_config.yaml

# ========== Answer Only Mode ==========
ANSWER_ONLY=true  # true: Answer만 사용, false: Question+Answer 사용

# ========== Factor Configuration ==========
FACTOR_STRATEGY="ekfac"  # "ekfac", "kfac", or "diagonal"
TRAIN_BATCH_SIZE=8
OUTPUT_DIR="./kronfluence_factors"

# ========== Script Path Setup ==========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # TOFU 디렉토리로 이동
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# ========== GPU / Memory Configuration ==========
export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=18765
export NPROC_PER_NODE=1  # GPU 개수에 맞게 조정
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"

echo "🚀 Starting Kronfluence factor fitting..."
echo "Model: ${MODEL_NAME}"
echo "Model Family: ${MODEL_FAMILY}"
echo "Dataset: ${DATA_PATH} (split: ${SPLIT})"
echo "Answer Only Mode: ${ANSWER_ONLY}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} | Processes: ${NPROC_PER_NODE} | Port: ${MASTER_PORT}"

# Build answer_only flag conditionally
if [ "$ANSWER_ONLY" = "true" ]; then
    ANSWER_ONLY_FLAG="--answer_only"
else
    ANSWER_ONLY_FLAG=""
fi

CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} \
  if/fit_factor.py \
  --model_name \"${MODEL_NAME}\" \
  --model_family \"${MODEL_FAMILY}\" \
  ${ANSWER_ONLY_FLAG} \
  --data_path \"${DATA_PATH}\" \
  --split \"${SPLIT}\" \
  --factor_strategy \"${FACTOR_STRATEGY}\" \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --output_dir \"${OUTPUT_DIR}\" \
  --covariance_module_partitions 4 \
  --lambda_module_partitions 4 \
  --covariance_data_partitions 4 \
  --lambda_data_partitions 4"

# ========== Run Factor Fitting ==========
eval $CMD
