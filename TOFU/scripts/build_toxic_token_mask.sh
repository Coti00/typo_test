#!/bin/bash

# Model configuration (fit_factor.bash와 동일한 모델 사용)
model="/root/tnpo/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-5000"
model_family="llama2-7b"

# Influence scores file (pairwise_scores.safetensors)
# compute_influence.py가 생성한 scores 경로
factors_path="/root/tnpo/TOFU/kronfluence_factors/if_results/scores_./influence_results/differential_ekfac_half_llama2-7b/pairwise_scores.safetensors"

# Dataset configuration
data_path="locuslab/TOFU"
forget_split="forget10"
retain_split="retain90"

# Answer Only Mode: Always enabled (hardcoded in build_forget_token_mask.py)
# answer_only flag is no longer needed - script always uses answer_only=True

# Token selection parameters
# Choose ONE of three modes: "topk", "global_threshold", "sample_threshold"
selection_mode="topk"  # Options: "topk", "global_threshold", "sample_threshold"

# Mode-specific parameters
top_k_value=2                    # Used when selection_mode="topk"
global_threshold_value=0.95      # Used when selection_mode="global_threshold" (percentile 0.0-1.0)
sample_threshold_value=0.9     # Used when selection_mode="sample_threshold" (normalized 0.0-1.0)

# Softmax temperature parameter
temperature=0.5  # Lower values (e.g., 0.1) make distribution sharper, higher values (e.g., 2.0) make it smoother

word_level="--word_level"  # Use word-level selection
factor_strategy="ekfac"  # fit_factor.bash에서 생성되는 이름과 일치해야 함

# Output configuration
save_dir="/root/tnpo/TOFU/influence_results"
save_id=""  # Optional: add identifier if needed

# Get script directory and navigate to if directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../if"

# Build base flags
base_flags="--model_name $model \
    --model_family $model_family \
    --data_path $data_path \
    --forget_split $forget_split \
    --retain_split $retain_split \
    --factors_path $factors_path \
    --factor_strategy $factor_strategy"

# Note: --answer_only flag removed - always uses answer_only=True (hardcoded in Python script)

# Build command based on selection_mode
if [ "$selection_mode" = "topk" ]; then
    echo "Using TOP-K mode (k=$top_k_value, temperature=$temperature)"
    python build_forget_token_mask.py \
        $base_flags \
        --top_k_per_sample $top_k_value \
        $word_level \
        --temperature $temperature \
        --save_dir "$save_dir"
elif [ "$selection_mode" = "global_threshold" ]; then
    echo "Using GLOBAL THRESHOLD mode (percentile=$global_threshold_value, temperature=$temperature)"
    python build_forget_token_mask.py \
        $base_flags \
        --top_k_per_sample 999 \
        --threshold_percentile $global_threshold_value \
        $word_level \
        --temperature $temperature \
        --save_dir "$save_dir"
elif [ "$selection_mode" = "sample_threshold" ]; then
    echo "Using PER-SAMPLE THRESHOLD mode (normalized threshold=$sample_threshold_value, temperature=$temperature)"
    python build_forget_token_mask.py \
        $base_flags \
        --top_k_per_sample 999 \
        --threshold_percentile 0.95 \
        --sample_threshold $sample_threshold_value \
        $word_level \
        --temperature $temperature \
        --save_dir "$save_dir"
else
    echo "Error: Invalid selection_mode '$selection_mode'"
    echo "Valid options: topk, global_threshold, sample_threshold"
    exit 1
fi