#!/bin/bash

# Model configuration
model_family="llama2-7b"  # Options: llama2-7b, phi, stablelm, pythia-1.4, zephyr-7b-beta
checkpoint_dir="/root/memorization_unlearn_no_logit/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-312"

# Factor configuration
factors_path="/root/tnpo/TOFU/kronfluence_factors"  # Parent directory (analysis_name will be appended)
factors_name="ekfac"            # Name of the factor strategy (WITHOUT "factors_" prefix - it's added automatically)
factor_strategy="ekfac"              # Strategy used for factors

# Dataset configuration
data_path="locuslab/TOFU"
forget_split="forget10"
retain_split="retain90"
max_length=256

# Answer Only Mode
answer_only=true  # true: Answer만 사용, false: Question+Answer 사용

# Computation configuration
query_batch_size=8
train_batch_size=8
use_half_precision="--use_half_precision"
use_compile="--use_compile"  # Enable torch.compile for faster computation  


# Output configuration
save_dir="./influence_results"
save_id="${model_family}"  # Automatically use model_family as ID (e.g., llama2-7b, phi, etc.)

# Navigate to TOFU root directory (one level up from scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TOFU_ROOT="${SCRIPT_DIR}/.."
cd "${TOFU_ROOT}"

echo "Working directory: $(pwd)"
echo ""
export CUDA_VISIBLE_DEVICES=1

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
