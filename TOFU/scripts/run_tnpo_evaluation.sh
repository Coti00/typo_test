#!/bin/bash
# TNPO Unlearned Model Evaluation Script

set -e

# ========== Configuration ==========
# Unlearned model checkpoint path (최종 체크포인트)
MODEL_PATH="/root/tnpo/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-5000/unlearned/8GPU_tnpo_1e-05_forget10_epoch5_batch4_accum4_beta2.5_gamma0.0_grad_diff_coeff1.0_reffine_tuned_evalsteps_per_epoch_seed42_1/checkpoint-125"

MODEL_FAMILY="llama2-7b"
METHOD_NAME="tnpo_si_weighted"

# Evaluation configuration
SPLIT="forget10_perturbed"  # forget10_perturbed, forget05_perturbed, forget01_perturbed
DS_SIZE=400  # Dataset size for evaluation
PORT=29501  # Master port for distributed training

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1

# ========== Path Setup ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "================================================"
echo "TNPO Unlearned Model Evaluation"
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

# Path to retain baseline results directory (should contain eval_log.json, eval_log_forget.json, etc.)
RETAIN_BASELINE_DIR="data/retain90_llama_wd0.01/eval_results/ds_size300"

if [ -d "${RETAIN_BASELINE_DIR}" ] && [ -d "${EVAL_RESULT_DIR}" ] && [ -f "aggregate_eval_stat.py" ]; then
    SAVE_FILE="${MODEL_PATH}/eval_results/final_aggregated_${METHOD_NAME}.csv"

    python3 aggregate_eval_stat.py \
        retain_result="${RETAIN_BASELINE_DIR}" \
        ckpt_result="${EVAL_RESULT_DIR}" \
        method_name="${METHOD_NAME}" \
        save_file="${SAVE_FILE}"

    echo "✅ Final statistics saved to: ${SAVE_FILE}"
    echo ""
    echo "Final results:"
    cat "${SAVE_FILE}"
else
    echo "⚠️  Skipping final aggregation (retain baseline, eval results, or aggregate script not found)"
    echo "   Retain baseline directory: ${RETAIN_BASELINE_DIR}"
    echo "   Eval results directory: ${EVAL_RESULT_DIR}"
    echo "   You can manually run:"
    echo "   python aggregate_eval_stat.py retain_result=${RETAIN_BASELINE_DIR} ckpt_result=${EVAL_RESULT_DIR} method_name=${METHOD_NAME} save_file=<output>"
fi

echo ""
echo "================================================"
echo "Evaluation complete!"
echo "================================================"
