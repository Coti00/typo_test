#!/bin/bash
# TOFU Model Evaluation Script
# Evaluates forget quality and model utility on forget, retain, real_world, and real_author datasets

set -e

# ========== Configuration ==========
# Model configuration
MODEL_PATH="/root/tnpo/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-5000"
MODEL_FAMILY="${2:-llama2-7b}"
METHOD_NAME="${3:-baseline}"

# Evaluation configuration
SPLIT="forget10"  # forget10, forget05, forget01
DS_SIZE=400  # Dataset size for evaluation
PORT=29500  # Master port for distributed training

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1

# ========== Path Setup ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "================================================"
echo "TOFU Model Evaluation"
echo "================================================"
echo "Model path: ${MODEL_PATH}"
echo "Model family: ${MODEL_FAMILY}"
echo "Split: ${SPLIT}"
echo "Method name: ${METHOD_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

# ========== Step 1: Run evaluation on all datasets ==========
echo "================================================"
echo "Step 1: Evaluating model on all datasets..."
echo "================================================"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=${PORT} \
    evaluate_util.py \
    model_family=${MODEL_FAMILY} \
    split=${SPLIT} \
    model_path=${MODEL_PATH}

# ========== Step 2: Check evaluation results ==========
EVAL_RESULT_DIR="${MODEL_PATH}/eval_results/ds_size${DS_SIZE}"
AGGREGATED_RESULT="${EVAL_RESULT_DIR}/eval_log_aggregated.json"

echo ""
echo "================================================"
echo "Step 2: Checking evaluation results..."
echo "================================================"

if [ -f "${AGGREGATED_RESULT}" ]; then
    echo "✅ Evaluation completed successfully!"
    echo "Results saved to: ${AGGREGATED_RESULT}"
    echo ""
    echo "Aggregated results:"
    cat "${AGGREGATED_RESULT}" | python3 -m json.tool
else
    echo "❌ Error: Aggregated results not found at ${AGGREGATED_RESULT}"
    exit 1
fi

# ========== Step 3: Aggregate final statistics (optional) ==========
echo ""
echo "================================================"
echo "Step 3: Generating final aggregated statistics..."
echo "================================================"

# Path to retain baseline results (should be pre-computed)
RETAIN_BASELINE="data/eval_log_aggregated.json"

if [ -f "${RETAIN_BASELINE}" ] && [ -f "aggregate_eval_stat.py" ]; then
    SAVE_FILE="${MODEL_PATH}/eval_results/final_aggregated_${METHOD_NAME}.csv"

    python3 aggregate_eval_stat.py \
        retain_result="${RETAIN_BASELINE}" \
        ckpt_result="${AGGREGATED_RESULT}" \
        method_name="${METHOD_NAME}" \
        save_file="${SAVE_FILE}"

    echo "✅ Final statistics saved to: ${SAVE_FILE}"
    echo ""
    echo "Final results:"
    cat "${SAVE_FILE}"
else
    echo "⚠️  Skipping final aggregation (retain baseline or aggregate script not found)"
    echo "   You can manually run:"
    echo "   python aggregate_eval_stat.py retain_result=<path> ckpt_result=${AGGREGATED_RESULT} method_name=${METHOD_NAME} save_file=<output>"
fi

echo ""
echo "================================================"
echo "Evaluation complete!"
echo "================================================"
