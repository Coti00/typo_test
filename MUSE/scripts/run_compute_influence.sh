#!/bin/bash

# ========== MUSE Data Configuration ==========
CORPUS="books"
FORGET="./data/$CORPUS/raw/forget.txt"
RETAIN="./data/$CORPUS/raw/retain1.txt"

# ========== Model Configuration ==========
MODEL_NAME="muse-bench/MUSE-Books_target"  # Fine-tuned model path
TOKENIZER_NAME="NousResearch/Llama-2-7b-chat-hf"

# ========== Dataset Configuration ==========
MAX_LENGTH=4096

# ========== Factor Configuration ==========
FACTOR_STRATEGY="ekfac"  # "ekfac", "kfac", or "diagonal"
FACTORS_PATH="/root/tnpo/MUSE/kronfluence_factors"
FACTORS_NAME="ekfac"
ANALYSIS_NAME="if_results"

# ========== Computation Configuration ==========
QUERY_BATCH_SIZE=1
TRAIN_BATCH_SIZE=1
USE_HALF_PRECISION="--use_half_precision"
USE_COMPILE="--use_compile"

# ========== Output Configuration ==========
SAVE_DIR="./influence_results"
SAVE_ID="muse_books"

# ========== Script Path Setup ==========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # MUSE directory
cd "${PROJECT_ROOT}"
>>>>>>> 614d2ab (test)

echo "Working directory: $(pwd)"
echo ""

# Build command (compute_influence.py is in if/ subdirectory)
cmd="python if/compute_influence.py 
    --model_name ${checkpoint_dir} \
    --model_family ${model_family} \
    --data_path ${data_path} \
    --forget_split ${forget_split} \
    --retain_split ${retain_split} \
    --max_length ${max_length} \
    --factors_path ${factors_path} \
    --factors_name ${factors_name} \
    --factor_strategy ${factor_strategy} \
    --query_batch_size ${query_batch_size} \
    --train_batch_size ${train_batch_size} \
    ${use_half_precision} \
    --save_dir ${save_dir}"

# Add optional save_id if specified
if [ ! -z "$save_id" ]; then
    cmd="${cmd} --save_id ${save_id}"
fi

# Add compile flag if enabled
if [ ! -z "$use_compile" ]; then
    cmd="${cmd} ${use_compile}"
fi

# Add --answer_only flag if enabled
if [ "${answer_only}" = "true" ]; then
    cmd="${cmd} --answer_only"

fi

echo "Running command:"
echo "$cmd"
echo ""

# Execute
eval $cmd
