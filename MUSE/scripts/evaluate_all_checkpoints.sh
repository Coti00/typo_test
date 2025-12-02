#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./evaluate_all_checkpoints.sh [BASE_DIR] [MODEL_FAMILY] [SPLIT] [BATCH_SIZE]
#
# Arguments:
#   BASE_DIR      - Directory containing checkpoints (default: npo checkpoint dir)
#   MODEL_FAMILY  - Model family name (default: llama2-7b)
#   SPLIT         - Dataset split to evaluate (default: forget01_perturbed)
#   BATCH_SIZE    - Batch size for evaluation (default: 32)
#
# Examples:
#   ./evaluate_all_checkpoints.sh
#   ./evaluate_all_checkpoints.sh "" llama2-7b forget01_perturbed 16
#   ./evaluate_all_checkpoints.sh /path/to/checkpoints llama2-7b forget10_perturbed 8

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate tofu

# Configuration
BASE_DIR="${1:-/root/tnpo/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-5000/unlearned/8GPU_npo_1e-05_forget01_epoch5_batch8_accum4_beta0.2_gamma1.0_grad_diff_coeff1.0_reffine_tuned_evalsteps_per_epoch_seed42_1}"
MODEL_FAMILY="${2:-llama2-7b}"
SPLIT="${3:-forget01_perturbed}"
BATCH_SIZE="${4:-32}"  # Batch size for evaluation (default: 30)
PORT=2705

# Disable DeepSpeed compilation
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_AIO=0
export PYTHONPATH="/root/tnpo/TOFU:${PYTHONPATH:-}"

# Find all checkpoint directories
CHECKPOINTS=($(ls -d ${BASE_DIR}/checkpoint-* | sort -V))

echo "Found ${#CHECKPOINTS[@]} checkpoints to evaluate"
echo "================================================"

# Retain result path
RETAIN_RESULT="/root/tnpo/TOFU/data/retain90_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"

# Loop through each checkpoint
for CKPT_PATH in "${CHECKPOINTS[@]}"; do
    CKPT_NAME=$(basename ${CKPT_PATH})
    echo ""
    echo "Evaluating checkpoint: ${CKPT_NAME}"
    echo "Path: ${CKPT_PATH}"
    echo "----------------------------------------"

    # Check if evaluation result already exists
    CKPT_RESULT="${CKPT_PATH}/eval_log_aggregated.json"
    if [ -f "${CKPT_RESULT}" ]; then
        echo "⚠️  Evaluation result already exists. Skipping evaluation..."
        echo "Existing file: ${CKPT_RESULT}"

        # Check if aggregate statistics exist
        SAVE_FILE="${CKPT_PATH}/aggregate_results"
        if [ -f "${SAVE_FILE}.json" ] || [ -f "${SAVE_FILE}.csv" ]; then
            echo "⚠️  Aggregate statistics also exist. Skipping completely."
            echo "----------------------------------------"
            continue
        else
            echo "Running aggregate statistics for ${CKPT_NAME}..."
            METHOD_NAME="tnpo_${CKPT_NAME}"

            python /root/tnpo/TOFU/aggregate_eval_stat.py \
                retain_result=${RETAIN_RESULT} \
                ckpt_result=${CKPT_RESULT} \
                method_name=${METHOD_NAME} \
                save_file=${SAVE_FILE}

            echo "Aggregate statistics saved to ${SAVE_FILE}"
            echo "----------------------------------------"
            continue
        fi
    fi

    # Run evaluation
    echo "Starting evaluation..."
    if CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=${PORT} \
        /root/tnpo/TOFU/evaluate_util.py \
        --config-path=/root/tnpo/TOFU/config \
        --config-name=eval_everything \
        model_family=${MODEL_FAMILY} \
        split=${SPLIT} \
        model_path=${CKPT_PATH} \
        save_dir=${CKPT_PATH} \
        batch_size=${BATCH_SIZE}; then
        echo "✓ Evaluation completed successfully for ${CKPT_NAME}"
    else
        echo "✗ Evaluation failed for ${CKPT_NAME}"
        echo "Skipping aggregate statistics..."
        echo "----------------------------------------"
        continue
    fi

    # Run aggregate statistics
    echo "Running aggregate statistics for ${CKPT_NAME}..."
    METHOD_NAME="tnpo_${CKPT_NAME}"
    SAVE_FILE="${CKPT_PATH}/aggregate_results"

    if python /root/tnpo/TOFU/aggregate_eval_stat.py \
        retain_result=${RETAIN_RESULT} \
        ckpt_result=${CKPT_RESULT} \
        method_name=${METHOD_NAME} \
        submitted_by=auto \
        save_file=${SAVE_FILE}; then
        echo "✓ Aggregate statistics saved to ${SAVE_FILE}.json and ${SAVE_FILE}.csv"
    else
        echo "✗ Failed to generate aggregate statistics for ${CKPT_NAME}"
    fi
    echo "----------------------------------------"
done

echo ""
echo "================================================"
echo "All checkpoints evaluated successfully!"
echo "================================================"