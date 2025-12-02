import logging
import os
import sys
import numpy as np
import argparse
from pathlib import Path

import yaml

import torch
from torch.utils.data import Dataset


# Add kronfluence to path
KRONFLUENCE_DIR = Path(__file__).parent.parent.parent / "kronfluence" / "src"
sys.path.insert(0, str(KRONFLUENCE_DIR))

from kronfluence.analyzer import Analyzer
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent MUSE directory to path for imports
MUSE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(MUSE_DIR))

from constants import DEFAULT_DATA, LLAMA_DIR

# Try to import dataset classes from utils.data_module
MUSERawTextDataset = None
TextDatasetQA = None
try:
    # Use importlib to avoid "if" keyword issue with module name
    import importlib
    # Add if/utils to path for import
    IF_UTILS_DIR = Path(__file__).parent / "utils"
    sys.path.insert(0, str(IF_UTILS_DIR))

    from data_module import MUSERawTextDataset, TextDatasetQA
    logging.info("Successfully imported MUSERawTextDataset and TextDatasetQA from data_module")
except ImportError as e:
    logging.warning(f"Failed to import from data_module: {e}")
    # Fallback: define MUSERawTextDataset inline if import fails
    class MUSERawTextDataset(Dataset):
        """Dataset for MUSE raw text files (forget.txt, retain.txt)"""
        def __init__(self, file_path, tokenizer, max_length=512, answer_only=True):
            super().__init__()
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.answer_only = answer_only

            # Read raw text file
            with open(file_path, 'r', encoding='utf-8') as f:
                self.texts = [line.strip() for line in f if line.strip()]

            logging.info(f"Loaded {len(self.texts)} samples from {file_path}")

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]

            # Tokenize
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)

            # Create labels (for answer_only mode, all tokens are "answer")
            if self.answer_only:
                labels = input_ids.clone()
                # Mask padding tokens
                labels[attention_mask == 0] = -100
            else:
                labels = input_ids.clone()

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Build influence-based token mask for machine unlearning.")

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
        "--retain_split",
        type=str,
        default="retain90",
        help="Retain split name (e.g., retain90).",
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
    # Answer Only Mode: Always enabled (Question removed for consistency with influence calculation)
    # parser.add_argument(
    #     "--answer_only",
    #     action="store_true",
    #     default=False,
    #     help="Use only answer (no question) for token mask building.",
    # )

    # Influence scores configuration
    parser.add_argument(
        "--factors_path",
        type=str,
        required=True,
        help="Path to the directory containing influence scores.",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy used to compute influence factors (e.g., ekfac, identity).",
    )

    # Output configuration
    parser.add_argument(
        "--save_id",
        type=str,
        default=None,
        help="ID to append to the output file names.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./influence_results",
        help="Directory to save the mask.",
    )

    # Token selection algorithm parameters
    parser.add_argument(
        "--top_k_per_sample",
        type=int,
        default=10,
        help="Number of top tokens to select per sample."
    )
    parser.add_argument(
        "--word_level",
        action="store_true",
        default=True,
        help="If set, select entire words instead of individual tokens."
    )
    parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=0.95,
        help="Percentile threshold for selecting high-influence tokens (0.0-1.0). Default: 0.95 (95th percentile)."
    )
    parser.add_argument(
        "--sample_threshold",
        type=float,
        default=None,
        help="Per-sample normalized threshold (0.0-1.0). If set, uses per-sample normalization instead of global percentile. Example: 0.8 selects words with normalized SI >= 0.8 within each sample."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for softmax normalization (default=1.0). Higher values (>1.0) create smoother distributions, lower values (<1.0) create sharper distributions."
    )

    return parser.parse_args()




def min_max_normalize(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize a tensor to the range [0, 1].
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + eps)


def harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the harmonic mean of two tensors.
    """
    a = a.float()
    b = b.float()
    return 2 * a * b / (a + b + eps)


def map_tokens_to_words(tokens):
    """
    Map token indices to word indices based on whitespace boundaries.

    Args:
        tokens: List of token strings from tokenizer

    Returns:
        token_to_word: Dict mapping token index to word index
        word_boundaries: List of (start_idx, end_idx) for each word
    """
    token_to_word = {}
    word_boundaries = []
    current_word_start = 0
    current_word_idx = 0

    for i, token in enumerate(tokens):
        # Check if token starts a new word (has leading space/special marker)
        # For LLaMA/sentencepiece: tokens starting with '▁' indicate word start
        # For GPT-2/BPE: tokens starting with 'Ġ' indicate word start
        is_word_start = (
            token.startswith('▁') or  # SentencePiece
            token.startswith('Ġ') or  # GPT-2 BPE
            i == 0  # First token is always a word start
        )

        if is_word_start and i > 0:
            # Save previous word boundary
            word_boundaries.append((current_word_start, i))
            current_word_idx += 1
            current_word_start = i

        token_to_word[i] = current_word_idx

    # Add the last word
    word_boundaries.append((current_word_start, len(tokens)))

    return token_to_word, word_boundaries


def load_datasets(args, tokenizer):
    """Load forget and retain datasets. Always uses ANSWER ONLY mode."""
    # Always use answer_only=True for consistency with influence calculation
    logging.info(f"✅ Using ANSWER ONLY mode (Question removed)")

    # Check if data_path is a directory with raw text files or HuggingFace dataset
    data_path = Path(args.data_path)

    if data_path.is_dir():
        # MUSE raw text format: data_path is a directory containing forget.txt, retain.txt, etc.
        forget_file = data_path / f"{args.forget_split}.txt"
        retain_file = data_path / f"{args.retain_split}.txt"

        logging.info(f"Loading MUSE raw text format from: {data_path}")
        logging.info(f"Forget file: {forget_file}")
        logging.info(f"Retain file: {retain_file}")

        if not forget_file.exists():
            raise FileNotFoundError(f"Forget file not found: {forget_file}")
        if not retain_file.exists():
            raise FileNotFoundError(f"Retain file not found: {retain_file}")

        forget_dataset = MUSERawTextDataset(
            file_path=forget_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
            answer_only=True
        )

        retain_dataset = MUSERawTextDataset(
            file_path=retain_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
            answer_only=True
        )
    else:
        # HuggingFace dataset format (TOFU)
        if TextDatasetQA is None:
            raise ImportError("TextDatasetQA not available. Cannot load HuggingFace datasets.")

        logging.info(f"Loading HuggingFace dataset format: {args.data_path}")
        logging.info(f"Forget split: {args.forget_split}")
        logging.info(f"Retain split: {args.retain_split}")

        forget_dataset = TextDatasetQA(
            data_path=args.data_path,
            tokenizer=tokenizer,
            model_family=args.model_family,
            max_length=args.max_length,
            split=args.forget_split,
            question_key=args.question_key,
            answer_key=args.answer_key,
            answer_only=True,  # ✅ Always True: Answer만 사용
        )

        retain_dataset = TextDatasetQA(
            data_path=args.data_path,
            tokenizer=tokenizer,
            model_family=args.model_family,
            max_length=args.max_length,
            split=args.retain_split,
            question_key=args.question_key,
            answer_key=args.answer_key,
            answer_only=True,  # ✅ Always True: Answer만 사용
        )


    logging.info(f"Forget dataset size: {len(forget_dataset)}")
    logging.info(f"Retain dataset size: {len(retain_dataset)}")

    return forget_dataset, retain_dataset


def min_max_normalize(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a tensor to the range [0, 1]."""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + eps)


def harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the harmonic mean of two tensors."""
    a = a.float()
    b = b.float()
    return 2 * a * b / (a + b + eps)


def softmax_normalize(tensor: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply softmax normalization with temperature to convert scores to probabilities.

    Args:
        tensor: Input tensor of scores
        temperature: Temperature parameter for smoothing (default=1.0)
                    - Higher temperature (>1.0): smoother distribution
                    - Lower temperature (<1.0): sharper distribution
    """
    return torch.nn.functional.softmax(tensor.float() / temperature, dim=0)


def build_topk_token_mask(
    influence_scores: torch.Tensor,  # shape (num_samples, seq_len)
    dataset,  # Dataset to get input_ids and tokens
    tokenizer,  # Tokenizer for token-to-word mapping
    top_k_per_sample: int = 10,
    word_level: bool = True,
    threshold_percentile: float = 0.95,
    sample_threshold: float = None,
    temperature: float = 1.0,
) -> tuple:
    """
    Builds an influence-based token mask using threshold-based selection strategy.

    Strategy:
    1. Rank samples by harmonic mean of (count of high-influence tokens, sum of high scores)
    2. For each sample (in ranked order), select words/tokens above the threshold percentile

    Args:
        influence_scores: Tensor of shape (num_samples, seq_len)
        dataset: Dataset containing input_ids
        tokenizer: Tokenizer for converting tokens to strings
        top_k_per_sample: Number of top words/tokens to select per sample (placeholder when using threshold)
        word_level: If True, select entire words; if False, select individual tokens
        threshold_percentile: Percentile threshold for selection (0.0-1.0, e.g., 0.95 = 95th percentile)

    Returns:
        mask: Boolean tensor where True = high-influence token
        sorted_indices: Indices of samples sorted by harmonic mean ranking
        scores: Processed scores tensor (2D)
        selected_words_info: List of dicts with selected word information
        all_words_info: List of dicts with all words and their SI scores
    """
    scores = influence_scores.detach()

    # Handle 3D tensor by squeezing the first dimension if it's 1
    if scores.ndim == 3 and scores.shape[0] == 1:
        scores = scores.squeeze(0)

    B, T = scores.shape

    # === Step 1: Rank samples by harmonic mean ===
    # Use global percentile threshold to identify "high-influence" tokens for ranking
    threshold = np.quantile(scores.cpu().float().numpy().flatten(), threshold_percentile)

    above_thresh = (scores > threshold).float()
    count_above = min_max_normalize(above_thresh.sum(dim=1))
    sum_above = min_max_normalize((scores * above_thresh).sum(dim=1))

    # Harmonic mean ranking
    harmonic_scores = harmonic_mean(count_above, sum_above)
    sorted_indices = torch.argsort(harmonic_scores, descending=True).tolist()

    logging.info(f"Ranked {B} samples by harmonic mean (count × sum of high-influence scores)")

    # === Step 2: Select TOP-K tokens per sample (in ranked order) ===
    mask = torch.zeros_like(scores, dtype=torch.float32)  # ✅ SI 점수 저장 (Boolean → Float)
    selected_words_info = []  # Store selected word information
    all_words_info = []  # Store ALL words with SI scores

    if sample_threshold is not None:
        logging.info(f"Selecting {'words' if word_level else 'tokens'} using per-sample normalization (threshold >= {sample_threshold:.2f})")
    else:
        logging.info(f"Selecting {'words' if word_level else 'tokens'} above threshold ({threshold_percentile*100:.1f}th percentile = {threshold:.6f})")
    pbar = tqdm(sorted_indices, desc=f"Processing samples", unit="sample")

    for i in pbar:
        # Get tokens and labels for this sample
        dataset_item = dataset[i]
        if isinstance(dataset_item, dict):
            input_ids = dataset_item["input_ids"]
            labels = dataset_item.get("labels", None)
        else:
            input_ids = dataset_item[0]
            labels = dataset_item[1] if len(dataset_item) > 1 else None

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Get answer-only mask (label != -100)
        if labels is not None:
            answer_mask = torch.tensor([label != -100 for label in labels], dtype=torch.bool)
        else:
            # Fallback: if no labels, allow all tokens
            answer_mask = torch.ones(len(tokens), dtype=torch.bool)

        if word_level:
            # Word-level selection with filtering
            token_to_word, word_boundaries = map_tokens_to_words(tokens)
            num_words = len(word_boundaries)

            # Define tokens to skip (special tokens and punctuation)
            skip_tokens = {'<s>', '</s>', '<unk>', '<pad>', '[INST]', '[/INST]', '▁[', ']'}
            skip_chars = {'.', ',', '!', '?', ';', ':', "'", '"', '-', '(', ')', '[', ']', '{', '}'}

            # Filter valid words and compute scores
            valid_word_scores = []
            valid_word_indices = []

            for word_idx in range(num_words):
                word_start, word_end = word_boundaries[word_idx]

                # Get tokens in this word
                word_tokens = tokens[word_start:word_end]

                # Skip if word is NOT in answer part (label == -100)
                word_in_answer = any(answer_mask[tok_idx] for tok_idx in range(word_start, word_end))
                if not word_in_answer:
                    continue

                # Skip if word contains ANY special tokens
                has_special_token = any(tok in skip_tokens for tok in word_tokens)
                if has_special_token:
                    continue

                # Skip if word is only punctuation (no alphanumeric content)
                has_content = False
                for tok in word_tokens:
                    tok_clean = tok.replace('▁', '').replace('Ġ', '').strip()
                    # Check if token has alphanumeric content (not just punctuation)
                    if tok_clean and tok_clean not in skip_chars and any(c.isalnum() for c in tok_clean):
                        has_content = True
                        break

                if has_content:
                    word_score = scores[i, word_start:word_end].max()  # Use MAX for word-level SI score
                    valid_word_scores.append(word_score)
                    valid_word_indices.append(word_idx)

                    # Store ALL words with SI scores
                    word_text = ''.join([t.replace('▁', ' ').replace('Ġ', ' ') for t in word_tokens]).strip()
                    all_words_info.append({
                        "sample_index": i,
                        "word": word_text,
                        "word_index": word_idx,
                        "token_start": word_start,
                        "token_end": word_end,
                        "si_score": word_score.item(),
                        "normalized_si_score": 0.0,  # Will be updated with softmax scores
                        "selected": False  # Will update later for selected words
                    })

            # Select words above threshold
            if len(valid_word_indices) == 0:
                continue

            valid_word_scores_tensor = torch.tensor(valid_word_scores)

            # ALWAYS compute normalized scores for this sample (for saving to JSON)
            # First apply min-max normalization to prevent numerical overflow in softmax
            minmax_scores = min_max_normalize(valid_word_scores_tensor)
            normalized_scores = softmax_normalize(minmax_scores, temperature=temperature)

            # Update all_words_info with normalized scores for this sample
            for all_word in all_words_info:
                if all_word['sample_index'] == i:
                    # Find matching index in valid_word_indices
                    if all_word['word_index'] in valid_word_indices:
                        idx_in_valid = valid_word_indices.index(all_word['word_index'])
                        all_word['normalized_si_score'] = normalized_scores[idx_in_valid].item()

            # Select top-k words or use threshold-based selection
            if sample_threshold is not None:
                # Mode 1: Per-sample normalization threshold
                above_threshold_mask = normalized_scores >= sample_threshold
                top_k_indices = torch.nonzero(above_threshold_mask, as_tuple=True)[0]
            elif top_k_per_sample < 999:
                # Mode 2: Top-K selection (select top K words per sample)
                k = min(top_k_per_sample, len(valid_word_scores_tensor))
                _, top_k_indices = torch.topk(valid_word_scores_tensor, k=k, largest=True)
            else:
                # Mode 3: Global threshold (original behavior)
                above_threshold_mask = valid_word_scores_tensor > threshold
                top_k_indices = torch.nonzero(above_threshold_mask, as_tuple=True)[0]

            # If no words above threshold, skip this sample
            if len(top_k_indices) == 0:
                continue


            # Get full_context for this sample (remove special tokens like <s>, </s>)
            try:
                full_text = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            except:
                full_text = ""

            # Mask all tokens in selected words (excluding punctuation tokens)
            for idx in top_k_indices:
                word_idx = valid_word_indices[idx]
                word_start, word_end = word_boundaries[word_idx]

                # Get word text
                word_tokens = tokens[word_start:word_end]
                word_text = ''.join([t.replace('▁', ' ').replace('Ġ', ' ') for t in word_tokens]).strip()

                # Get average SI score for this word
                word_si_score = valid_word_scores_tensor[idx].item()

                # Mark this word as selected in all_words_info
                for all_word in all_words_info:
                    if (all_word['sample_index'] == i and
                        all_word['word_index'] == word_idx):
                        all_word['selected'] = True
                        all_word['full_context'] = full_text
                        break

                # Store word information (ALWAYS include normalized score)
                word_info = {
                    "sample_index": i,
                    "word": word_text,
                    "word_index": word_idx,
                    "token_start": word_start,
                    "token_end": word_end,
                    "si_score": word_si_score,
                    "normalized_si_score": normalized_scores[idx].item(),  # Always include
                    "full_context": full_text
                }

                selected_words_info.append(word_info)

                # Only mask non-punctuation tokens within the word
                # ✅ Get normalized SI score for this word
                word_normalized_si = normalized_scores[idx].item()

                for tok_idx in range(word_start, word_end):
                    # Skip tokens beyond the mask's sequence length
                    if tok_idx >= mask.shape[1]:
                        continue

                    tok = tokens[tok_idx]
                    tok_clean = tok.replace('▁', '').replace('Ġ', '').strip()

                    # Skip punctuation tokens
                    if tok_clean in skip_chars:
                        continue

                    # Skip special tokens
                    if tok in skip_tokens:
                        continue

                    # ✅ SI 점수 저장 (True 대신 normalized score)
                    mask[i, tok_idx] = word_normalized_si
        else:
            # Token-level selection (only from answer part)
            sample_scores = scores[i]

            # Filter to answer-only tokens (label != -100)
            answer_indices = answer_mask.nonzero(as_tuple=True)[0]
            valid_indices = answer_indices[(sample_scores[answer_indices] != 0).nonzero(as_tuple=True)[0]]

            if len(valid_indices) == 0:
                continue

            valid_scores = sample_scores[valid_indices]
            k = min(top_k_per_sample, len(valid_scores))
            _, top_k_indices = torch.topk(valid_scores, k=k, largest=True)

            selected_indices = valid_indices[top_k_indices]

            # ✅ Token-level도 normalized SI score 저장
            # Apply min-max normalization then softmax for token-level scores
            minmax_scores = min_max_normalize(valid_scores[top_k_indices])
            normalized_token_scores = softmax_normalize(minmax_scores, temperature=temperature)

            for j, idx in enumerate(selected_indices):
                mask[i, idx] = normalized_token_scores[j].item()

    pbar.close()

    logging.info(f"Total tokens with SI scores: {(mask > 0).sum().item()}")
    logging.info(f"Sum of all SI scores: {mask.sum().item():.6f}")
    logging.info(f"Total words selected: {len(selected_words_info)}")
    logging.info(f"Total words analyzed: {len(all_words_info)}")
    return mask, sorted_indices, scores, selected_words_info, all_words_info


 


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    # Try to load model config from yaml, fallback to using tokenizer_name directly
    model_id = None
    config_path = MUSE_DIR / "config" / "model_config.yaml"

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                model_configs = yaml.load(f, Loader=yaml.FullLoader)
            model_cfg = model_configs.get(args.model_family, {})
            model_id = model_cfg.get("hf_key")
            logging.info(f"Loaded model config from: {config_path}")
        except Exception as e:
            logging.warning(f"Failed to load model config: {e}")

    # Fallback: use model_name or a default tokenizer
    if model_id is None:
        # For MUSE, we can use the model_name directly or use LLAMA_DIR
        model_id = LLAMA_DIR  # Default to Llama-2-7b
        logging.info(f"Using default tokenizer: {model_id}")

    logging.info(f"Building/loading influence token mask for model family: {args.model_family}")
    logging.info(f"Using tokenizer: {model_id}")

    logging.info(f"Using checkpoint: {args.model_name}")

    # Load tokenizer from HuggingFace model id
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)


    # Set pad_token if not already set (required for padding)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Load datasets
    forget_dataset, retain_dataset = load_datasets(args, tokenizer)

    # Construct save path
    save_id_str = f"_{args.save_id}" if args.save_id else ""
    word_suffix = "_word" if args.word_level else "_token"

    # Determine mode and filename based on arguments
    if args.sample_threshold is not None:
        # Per-sample normalization mode
        threshold_str = f"sample={args.sample_threshold}"
    elif args.top_k_per_sample < 999:
        # TOP-K mode
        threshold_str = f"topk={args.top_k_per_sample}"
    else:
        # Global threshold mode
        threshold_str = f"threshold={args.threshold_percentile}"

    save_path = os.path.join(
        args.save_dir,
        "influence_masks",
        f"{args.factor_strategy}_{args.forget_split}{save_id_str}",
        f"mask_{threshold_str}{word_suffix}.pt"
    )

    # Load influence scores from factors_path
    logging.info(f"Loading pairwise scores from {args.factors_path}")
    scores = Analyzer.load_file(args.factors_path)['all_modules']

    # Handle 3D tensor by squeezing the first dimension if it's 1
    if scores.ndim == 3 and scores.shape[0] == 1:
        scores = scores.squeeze(0)

    # Define paths for mask and JSON files
    selected_words_path = save_path.replace('.pt', '_selected_words.json')
    all_words_path = save_path.replace('.pt', '_all_words.json')

    if os.path.exists(save_path):
        mask = torch.load(save_path, map_location="cpu", weights_only=True)
        logging.info(f"Loaded mask from {save_path}")

        # Try to load selected words info if it exists
        if os.path.exists(selected_words_path):
            import json
            with open(selected_words_path, 'r') as f:
                selected_words_info = json.load(f)
            logging.info(f"Loaded selected words from {selected_words_path}")
        else:
            selected_words_info = []

        # Try to load all words info if it exists
        if os.path.exists(all_words_path):
            import json
            with open(all_words_path, 'r') as f:
                all_words_info = json.load(f)
            logging.info(f"Loaded all words from {all_words_path}")
        else:
            all_words_info = []
    else:
        logging.info(f"Building TOP-{args.top_k_per_sample} token mask and saving to {save_path}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        mask, _, scores, selected_words_info, all_words_info = build_topk_token_mask(
            scores,
            forget_dataset,
            tokenizer,
            top_k_per_sample=args.top_k_per_sample,
            word_level=args.word_level,
            threshold_percentile=args.threshold_percentile,
            sample_threshold=args.sample_threshold,
            temperature=args.temperature
        )
        torch.save(mask, save_path)

        # Save selected words info
        import json
        with open(selected_words_path, 'w', encoding='utf-8') as f:
            json.dump(selected_words_info, f, ensure_ascii=False, indent=2)
        logging.info(f"Selected words saved to: {selected_words_path}")

        # Save ALL words info
        with open(all_words_path, 'w', encoding='utf-8') as f:
            json.dump(all_words_info, f, ensure_ascii=False, indent=2)
        logging.info(f"All words with SI scores saved to: {all_words_path}")

    logging.info(f"✅ Influence token mask built successfully!")
    logging.info(f"Mask shape: {mask.shape}")
    logging.info(f"Total tokens with SI scores: {(mask > 0).sum().item()}")
    logging.info(f"Sum of all SI scores: {mask.sum().item():.6f}")
    if 'selected_words_info' in locals():
        logging.info(f"Total selected words: {len(selected_words_info)}")
    if 'all_words_info' in locals():
        logging.info(f"Total words analyzed: {len(all_words_info)}")
    logging.info(f"Mask saved to: {save_path}")

if __name__ == "__main__":
    main()