"""
Analyze influence-based token masks and save detailed token-level SI scores.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent directories to path
TOFU_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(TOFU_DIR))

from data_module import TextDatasetQA
from kronfluence.analyzer import Analyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze influence-based token masks.")

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="llama2-7b",
        help="Model family (e.g., llama2, phi).",
    )

    # Dataset configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default="locuslab/TOFU",
        help="Path to TOFU dataset.",
    )
    parser.add_argument(
        "--forget_split",
        type=str,
        default="forget10",
        help="Forget split name (e.g., forget10).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--question_key",
        type=str,
        default="question",
        help="Key for question in dataset.",
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        default="answer",
        help="Key for answer in dataset.",
    )

    # Input paths
    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to the saved mask file (.pt).",
    )
    parser.add_argument(
        "--scores_path",
        type=str,
        required=True,
        help="Path to the influence scores file.",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis_results",
        help="Directory to save analysis results.",
    )

    return parser.parse_args()


def map_token_index_to_word(tokens):
    """
    Map token indices to word indices based on whitespace boundaries.

    Returns:
        words: List of words (merged tokens)
        token_to_word: Dict mapping token index to word index
    """
    words = []
    token_to_word = {}
    current_word = ""
    current_word_idx = 0

    for i, token in enumerate(tokens):
        # Check if token starts a new word
        is_word_start = (
            token.startswith('▁') or  # SentencePiece
            token.startswith('Ġ') or  # GPT-2 BPE
            i == 0  # First token
        )

        if is_word_start and i > 0:
            # Save previous word
            words.append(current_word)
            current_word_idx += 1
            current_word = token.replace('▁', '').replace('Ġ', '')
        else:
            # Continue current word
            current_word += token.replace('▁', '').replace('Ġ', '')

        token_to_word[i] = current_word_idx

    # Add the last word
    if current_word:
        words.append(current_word)

    return words, token_to_word


def analyze_influence_mask(
    tokenizer,
    dataset,
    forget_mask: torch.Tensor,
    influence_scores: torch.Tensor,
    output_path: str,
):
    """Analyze and save selected tokens with their SI scores."""
    num_samples, seq_len = forget_mask.shape
    selected_tokens_si = []

    # Special token patterns to skip
    special_token_patterns = ['[INST]', '[/INST]', '<<SYS>>', '<</SYS>>']

    logging.info(f"Analyzing {num_samples} samples...")

    for sample_index in tqdm(range(num_samples), desc="Analyzing samples"):
        # Get sample data
        item = dataset[sample_index]
        if isinstance(item, dict) and "input_ids" in item:
            input_ids = item["input_ids"]
        elif isinstance(item, (tuple, list)) and len(item) >= 1:
            input_ids = item[0]
        else:
            continue

        # Convert to tensor if needed
        if torch.is_tensor(input_ids):
            input_ids_tensor = input_ids.detach().cpu()
        else:
            input_ids_tensor = torch.tensor(input_ids)

        # Get scores and mask for this sample
        scores_row = influence_scores[sample_index].detach().cpu()
        mask_row = forget_mask[sample_index].detach().cpu()

        # Ensure consistent length
        L = min(int(scores_row.shape[0]), int(input_ids_tensor.shape[0]), int(mask_row.shape[0]))
        if L <= 0:
            continue

        input_ids_tensor = input_ids_tensor[:L]
        scores_row = scores_row[:L]
        mask_row = mask_row[:L]

        # Convert tokens to strings
        tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor.tolist())
        words, token_to_word = map_token_index_to_word(tokens)

        # Get full context
        full_context = tokenizer.decode(input_ids_tensor.tolist(), skip_special_tokens=True)

        # Collect selected tokens
        selected_positions = [i for i, flag in enumerate(mask_row.tolist()) if flag]

        for pos in selected_positions:
            tok = tokens[pos]
            score_val = float(scores_row[pos]) if pos < len(scores_row) else 0.0

            # Only include if SI score is positive
            if score_val <= 0:
                continue

            # Get word information
            wi = token_to_word.get(pos)
            word = words[wi] if (wi is not None and 0 <= wi < len(words)) else tok

            # Skip special tokens
            if any(pattern in word for pattern in special_token_patterns):
                continue

            token_si_info = {
                "sample_index": sample_index,
                "token_index": pos,
                "word_index": wi if wi is not None else -1,
                "token": tok,
                "word": word,
                "si_score": score_val,
                "full_context": full_context
            }
            selected_tokens_si.append(token_si_info)

    # Sort by SI score (descending)
    selected_tokens_si.sort(key=lambda x: x["si_score"], reverse=True)

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_tokens_si, f, ensure_ascii=False, indent=2)

    logging.info(f"💾 Saved selected tokens SI scores to: {output_path}")
    logging.info(f"📊 Total selected tokens: {len(selected_tokens_si)}")

    # Print top 10 tokens
    logging.info("\n🔝 Top 10 tokens by SI score:")
    for i, item in enumerate(selected_tokens_si[:10], 1):
        logging.info(f"  {i}. '{item['word']}' (token: '{item['token']}') - SI: {item['si_score']:.6f}")
        logging.info(f"     Context: {item['full_context'][:100]}...")

    return selected_tokens_si


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("Starting influence mask analysis...")

    # Load tokenizer
    logging.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)

    # Load dataset
    logging.info(f"Loading dataset: {args.forget_split}")
    forget_dataset = TextDatasetQA(
        data_path=args.data_path,
        tokenizer=tokenizer,
        model_family=args.model_family,
        max_length=args.max_length,
        split=args.forget_split,
        question_key=args.question_key,
        answer_key=args.answer_key,
    )
    logging.info(f"Dataset size: {len(forget_dataset)}")

    # Load mask
    logging.info(f"Loading mask from {args.mask_path}")
    mask = torch.load(args.mask_path, map_location="cpu", weights_only=True)
    logging.info(f"Mask shape: {mask.shape}")
    logging.info(f"Total masked tokens: {mask.sum().item()}")

    # Load influence scores
    logging.info(f"Loading influence scores from {args.scores_path}")
    scores_data = Analyzer.load_file(args.scores_path)
    influence_scores = scores_data['all_modules']
    logging.info(f"Influence scores shape: {influence_scores.shape}")

    # Create output path
    mask_filename = os.path.basename(args.mask_path).replace('.pt', '')
    output_path = os.path.join(args.output_dir, f"{mask_filename}_analysis.json")

    # Analyze
    selected_tokens = analyze_influence_mask(
        tokenizer,
        forget_dataset,
        mask,
        influence_scores,
        output_path
    )

    logging.info("✅ Analysis complete!")


if __name__ == "__main__":
    main()
