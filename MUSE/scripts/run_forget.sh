#!/bin/bash

# Configuration
master_port=18763
split=forget10
model=phi
lr=2e-5

# TNPO specific parameters
forget_loss=tnpo
beta=0.15
gamma=1.0

# SI 점수 마스크 경로 (생성한 마스크 파일)
forget_mask_path='/root/tnpo/TOFU/influence_results/influence_masks/ekfac_forget10/mask_sample=0.18_word_spacy.pt'

# Training parameters
batch_size=4  # Reduced from 32 to save GPU memory
gradient_accumulation_steps=2  # Increased to maintain effective batch size of 32
num_epochs=5
weight_decay=0.01

# Random seed
seed=42

# Number of GPUs
num_gpus=1

# Model path (will use default from model_config.yaml if not set)
# model_path="/path/to/your/fine-tuned/model"

echo "=========================================="
echo "Running TNPO Unlearning"
echo "=========================================="
echo "Model: ${model}"
echo "Split: ${split}"
echo "Learning rate: ${lr}"
echo "Forget loss: ${forget_loss}"
echo "Beta: ${beta}"
echo "Gamma: ${gamma}"
echo "Forget mask: ${forget_mask_path}"
echo "Batch size: ${batch_size}"
echo "Gradient accumulation: ${gradient_accumulation_steps}"
echo "Number of GPUs: ${num_gpus}"
echo "=========================================="

# Get script directory and navigate to TOFU root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "Working directory: $(pwd)"

# Run training
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=${num_gpus} \
    --master_port=${master_port} \
    forget.py \
    --config-name=forget \
    split=${split} \
    model_family=${model} \
    lr=${lr} \
    forget_loss=${forget_loss} \
    beta=${beta} \
    gamma=${gamma} \
    forget_mask_path=\"${forget_mask_path}\" \
    batch_size=${batch_size} \
    gradient_accumulation_steps=${gradient_accumulation_steps} \
    num_epochs=${num_epochs} \
    weight_decay=${weight_decay} \
    seed=${seed}

echo "=========================================="
echo "Training completed!"
echo "=========================================="
