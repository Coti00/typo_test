import logging
import os
import sys
import numpy as np
import argparse
from pathlib import Path

import torch

# Add kronfluence to path
KRONFLUENCE_DIR = Path(__file__).parent.parent.parent / "kronfluence" / "src"
sys.path.insert(0, str(KRONFLUENCE_DIR))

from kronfluence.analyzer import Analyzer
from transformers import AutoTokenizer
from tqdm import tqdm
import spacy

# Load spacy model globally
try:
    NLP = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(['python3', '-m', 'spacy', 'download', 'en_core_web_sm'])
    NLP = spacy.load('en_core_web_sm')

# Add parent TOFU directory to path for imports
TOFU_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(TOFU_DIR))

from data_module import TextDatasetQA
import yaml


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

    parser.add_argument(
        "--use_spacy",
        action="store_true",
        default=False,
        help="Use spacy NER + important POS tags (VERB, ADJ, NOUN, PROPN) for word segmentation instead of whitespace-based tokenization."
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
    Map token indices to word indices based on whitespace boundaries (legacy).
    Use map_tokens_to_spacy_entities for spacy-based mapping.
    """
    token_to_word = {}
    word_boundaries = []
    current_word_start = 0
    current_word_idx = 0

    for i, token in enumerate(tokens):
        prev_token = tokens[i-1] if i > 0 else ''
        is_word_start = (
            token.startswith('▁') or
            token.startswith('Ġ') or
            token == '<|endoftext|>' or
            token == '</s>' or
            token == '[/INST]' or
            token == ']' or
            prev_token == ']' or
            i == 0
        )

        if is_word_start and i > 0:
            word_boundaries.append((current_word_start, i))
            current_word_idx += 1
            current_word_start = i

        token_to_word[i] = current_word_idx

    word_boundaries.append((current_word_start, len(tokens)))
    return token_to_word, word_boundaries


def map_tokens_to_spacy_entities(tokens, text, debug_sample_idx=None):
    """
    Map token indices to spaCy Named Entities + PROPN phrases.

    - 1순위: spaCy doc.ents (PERSON, ORG, GPE, DATE, etc.)
    - 2순위: NER에 안 잡힌 연속된 PROPN 묶음 (ex. 'Hsiao Yun-Hwa')

    Args:
        debug_sample_idx: If not None, print debug info for this sample index
    """

    doc = NLP(text)

    if debug_sample_idx is not None:
        logging.info(f"\n{'='*80}")
        logging.info(f"[DEBUG {debug_sample_idx}] 🔍 spaCy Processing Started")
        logging.info(f"{'='*80}")
        logging.info(f"[DEBUG {debug_sample_idx}] Input text length: {len(text)}")
        logging.info(f"[DEBUG {debug_sample_idx}] Input text: {text[:500]}...")
        logging.info(f"\n[DEBUG {debug_sample_idx}] spaCy Tokens and POS tags:")
        for i, tok in enumerate(doc):
            logging.info(f"  [{i:03d}] text='{tok.text}' | pos={tok.pos_} | ent_type={tok.ent_type_ or 'NONE'} | idx={tok.idx}")
        logging.info("")

    # 1) 토크나이저 토큰을 원문 text에 정렬 (대략적인 char span 추정)
    token_char_spans = []  # (start_char, end_char)
    current_pos = 0
    lower_text = text.lower()

    if debug_sample_idx is not None:
        logging.info(f"[DEBUG {debug_sample_idx}] Starting character alignment for {len(tokens)} HF tokens")

    for tok_idx, tok in enumerate(tokens):
        # BPE 특수 문자 제거 후 검색
        clean_tok = tok.replace('▁', ' ').replace('Ġ', ' ').strip()

        # ✅ CRITICAL FIX: Skip special tokens and prompt meta tokens completely
        # 특수 토큰, 프롬프트 메타 토큰, 패딩, 단일 특수 문자 모두 skip
        special_tokens = {
            '<s>', '</s>', '<unk>', '<pad>', '<|endoftext|>',
            '[INST]', '[/INST]', '▁[',
            '[', ']', '/', 'INST',  # 프롬프트 메타 토큰
            '▁', ' ', '',  # 공백/빈 토큰
        }

        if tok in special_tokens or clean_tok in special_tokens:
            # Assign dummy position (won't match any entity)
            token_char_spans.append((-1, -1))
            continue

        if not clean_tok:
            clean_tok = tok

        clean_lower = clean_tok.lower()
        start_pos = lower_text.find(clean_lower, current_pos)
        if start_pos == -1:
            # ⚠️ FALLBACK: 텍스트에서 못 찾으면 이것도 특수 토큰으로 간주하고 skip
            token_char_spans.append((-1, -1))
            if debug_sample_idx is not None:
                logging.info(f"  [{tok_idx:03d}] ⚠️  Token '{tok}' not found in text, treating as special -> char span [-1:-1]")
            continue

        end_pos = start_pos + len(clean_tok)

        token_char_spans.append((start_pos, end_pos))
        current_pos = max(current_pos, end_pos)

        if debug_sample_idx is not None and tok_idx < 20:  # Only log first 20 for brevity
            logging.info(f"  [{tok_idx:03d}] Token '{tok}' (cleaned: '{clean_tok}') -> char span [{start_pos}:{end_pos}]")

    entities = []

    # 2) spaCy NER 엔티티 그대로 사용
    for ent in doc.ents:
        if len(ent.text.strip()) < 2:
            continue

        entities.append({
            "text": ent.text,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "type": f"NER:{ent.label_}",   # 예: NER:PERSON, NER:GPE
        })

    if debug_sample_idx is not None:
        logging.info(f"[DEBUG {debug_sample_idx}] Initial NER entities: {len(entities)}")
        for ent in entities:
            logging.info(f"  NER: '{ent['text']}' [{ent['start_char']}:{ent['end_char']}] {ent['type']}")

    # 2.5) NER 엔티티를 앞쪽 PROPN까지 확장 (예: Hsiao + Yun-Hwa's)
    expanded = []
    for ent in entities:
        start_char = ent["start_char"]
        end_char = ent["end_char"]

        if debug_sample_idx is not None:
            logging.info(f"[DEBUG {debug_sample_idx}] Expanding entity: '{ent['text']}' [{start_char}:{end_char}]")

        span = doc.char_span(start_char, end_char, alignment_mode="expand")
        if span is None:
            if debug_sample_idx is not None:
                logging.info(f"  → No span found, keeping as is")
            expanded.append(ent)
            continue

        start_token = span.start
        j = start_token - 1

        if debug_sample_idx is not None:
            logging.info(f"  → Span found: tokens[{span.start}:{span.end}], checking previous tokens...")

        # 바로 앞에 ent_type 없는 PROPN이 붙어 있으면 앞으로 확장
        while j >= 0:
            prev = doc[j]
            distance = span.start_char - prev.idx

            if debug_sample_idx is not None:
                logging.info(f"    - doc[{j}]: '{prev.text}' | pos={prev.pos_} | ent_type={prev.ent_type_ or 'NONE'} | distance={distance}")

            # 공백/구두점 포함해서 2글자 이내로 붙어 있으면 같은 이름으로 본다
            if prev.pos_ == "PROPN" and not prev.ent_type_ and distance <= 2:
                start_char = prev.idx
                if debug_sample_idx is not None:
                    logging.info(f"      ✓ Expanding to include '{prev.text}' (new start_char={start_char})")
                j -= 1
            else:
                if debug_sample_idx is not None:
                    logging.info(f"      ✗ Stopping expansion (not PROPN or too far)")
                break

        if start_char != ent["start_char"]:
            ent["start_char"] = start_char
            ent["text"] = doc.text[start_char:end_char]
            if debug_sample_idx is not None:
                logging.info(f"  → Final expanded entity: '{ent['text']}' [{start_char}:{end_char}]")
        else:
            if debug_sample_idx is not None:
                logging.info(f"  → No expansion needed")

        expanded.append(ent)

    entities = expanded

    if debug_sample_idx is not None:
        logging.info(f"[DEBUG {debug_sample_idx}] After expansion: {len(entities)} entities")

    # 3) NER에 안 잡힌 PROPN 묶음도 엔티티처럼 취급 (예: 이름인데 NER이 못 잡은 경우)
    if debug_sample_idx is not None:
        logging.info(f"\n[DEBUG {debug_sample_idx}] 🔍 Searching for PROPN sequences not captured by NER...")

    i = 0
    while i < len(doc):
        tok = doc[i]

        # 이미 NER에 포함된 토큰은 건너뛰기
        if tok.ent_type_:
            i += 1
            continue

        if tok.pos_ == "PROPN" and len(tok.text.strip()) >= 2:
            if debug_sample_idx is not None:
                logging.info(f"[DEBUG {debug_sample_idx}] Found PROPN at doc[{i}]: '{tok.text}' (not in NER)")

            start_idx = tok.idx
            j = i + 1
            while j < len(doc):
                nxt = doc[j]
                gap = nxt.idx - (doc[j-1].idx + len(doc[j-1].text))

                if debug_sample_idx is not None:
                    logging.info(f"  - Checking doc[{j}]: '{nxt.text}' | pos={nxt.pos_} | ent_type={nxt.ent_type_ or 'NONE'} | gap={gap}")

                if nxt.pos_ == "PROPN" and not nxt.ent_type_:
                    # 띄어쓰기/소유격 등으로 살짝 붙어 있을 수 있음
                    if gap <= 2:
                        if debug_sample_idx is not None:
                            logging.info(f"    ✓ Merging '{nxt.text}' into PROPN sequence (gap={gap})")
                        j += 1
                    else:
                        if debug_sample_idx is not None:
                            logging.info(f"    ✗ Gap too large ({gap}), stopping merge")
                        break
                else:
                    if debug_sample_idx is not None:
                        logging.info(f"    ✗ Not PROPN or already in NER, stopping")
                    break

            end_tok = doc[j - 1]
            end_idx = end_tok.idx + len(end_tok.text)
            merged_text = doc.text[start_idx:end_idx]

            # Keep PROPN sequences as-is (don't split possessive)
            if len(merged_text.strip()) >= 2:
                if debug_sample_idx is not None:
                    logging.info(f"  → Created PROPN entity: '{merged_text}' [{start_idx}:{end_idx}] from doc[{i}:{j}]")
                entities.append({
                    "text": merged_text,
                    "start_char": start_idx,
                    "end_char": end_idx,
                    "type": "POS:PROPN",
                })

            i = j
        else:
            i += 1

    if debug_sample_idx is not None:
        logging.info(f"[DEBUG {debug_sample_idx}] Final entities count: {len(entities)}")

    # 4) token index ↔ entity index 매핑
    if debug_sample_idx is not None:
        logging.info(f"\n[DEBUG {debug_sample_idx}] 🔍 Mapping entities to HF tokenizer tokens...")
        logging.info(f"[DEBUG {debug_sample_idx}] Total HF tokens: {len(tokens)}")
        logging.info(f"[DEBUG {debug_sample_idx}] Total entities to map: {len(entities)}")

    token_to_entity = {}
    entity_boundaries = []

    for ent_idx, ent in enumerate(entities):
        ent_start = ent["start_char"]
        ent_end = ent["end_char"]

        if debug_sample_idx is not None:
            logging.info(f"\n[DEBUG {debug_sample_idx}] Processing entity #{ent_idx}: '{ent['text']}' [{ent_start}:{ent_end}] ({ent['type']})")

        matching_tokens = []
        for tok_idx, (tok_start, tok_end) in enumerate(token_char_spans):
            # ✅ CRITICAL: Skip special tokens (char span = (-1, -1))
            if tok_start == -1 and tok_end == -1:
                continue

            # span이 조금이라도 겹치면 매칭
            if tok_start < ent_end and tok_end > ent_start:
                matching_tokens.append(tok_idx)
                if debug_sample_idx is not None:
                    logging.info(f"  - Matched HF token[{tok_idx}]: '{tokens[tok_idx]}' (char span [{tok_start}:{tok_end}])")

        if matching_tokens:
            start_tok = min(matching_tokens)
            end_tok = max(matching_tokens) + 1
            entity_boundaries.append(
                (start_tok, end_tok, ent["text"], ent["type"])
            )

            if debug_sample_idx is not None:
                logging.info(f"  → Entity boundary: tokens[{start_tok}:{end_tok}] = '{ent['text']}' ({ent['type']})")
                logging.info(f"  → Mapped HF tokens: {[tokens[i] for i in range(start_tok, end_tok)]}")

            for tok_idx in matching_tokens:
                if tok_idx not in token_to_entity:
                    token_to_entity[tok_idx] = ent_idx
        else:
            if debug_sample_idx is not None:
                logging.info(f"  ⚠️  No matching HF tokens found!")

    if debug_sample_idx is not None:
        logging.info(f"\n[DEBUG {debug_sample_idx}] Final entity boundaries: {len(entity_boundaries)}")
        logging.info(f"{'='*80}\n")

    return token_to_entity, entity_boundaries



def load_datasets(args, tokenizer):
    """Load forget and retain datasets. Always uses ANSWER ONLY mode."""
    # Always use answer_only=True for consistency with influence calculation
    logging.info(f"✅ Using ANSWER ONLY mode (Question removed)")

    logging.info(f"Loading forget dataset: {args.forget_split}")
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

    logging.info(f"Loading retain dataset: {args.retain_split}")
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
    use_spacy: bool = False,  # NEW: Use spacy for word segmentation
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

    count_above = softmax_normalize(above_thresh.sum(dim=1))
    sum_above = softmax_normalize((scores * above_thresh).sum(dim=1))

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

        # ==== DEBUG: sample_index == 4일 때 HF 토큰 확인 ====
        if i == 4:
            logging.info("===== DEBUG (sample_index = 4) - HF tokens =====")
            debug_skip_tokens = {
                '<s>', '</s>', '<unk>', '<pad>', '<|endoftext|>',
                '[INST]', '[/INST]', '▁', '▁[', ']', 'INST', '/'
            }
            for tidx, tok in enumerate(tokens):
                if tok in debug_skip_tokens:
                    continue
                logging.info(f"[HF TOK] idx={tidx:03d}, token={repr(tok)}")


        # Get answer-only mask (label != -100)
        if labels is not None:
            answer_mask = torch.tensor([label != -100 for label in labels], dtype=torch.bool)
        else:
            # Fallback: if no labels, allow all tokens
            answer_mask = torch.ones(len(tokens), dtype=torch.bool)

        if word_level:
            # Word-level selection with filtering
            # ALWAYS get whitespace-based word boundaries for all_words_info
            token_to_word, all_word_boundaries = map_tokens_to_words(tokens)

            if use_spacy:
                # Use spacy NER + important POS tags (VERB, ADJ, NOUN, PROPN) for SELECTION
                try:
                    # ✅ SIMPLE APPROACH: 전체 문장 디코딩 → 특수 토큰 제거 → spaCy 적용
                    full_text = tokenizer.decode(
                        input_ids,
                        skip_special_tokens=False,  # 일단 모두 디코딩
                        clean_up_tokenization_spaces=True
                    )

                    # 특수 토큰 제거
                    full_text = (
                        full_text
                        .replace('[/INST]', '')
                        .replace('[INST]', '')
                        .replace('<s>', '')
                        .replace('</s>', '')
                        .replace('<|endoftext|>', '')
                        .strip()
                    )
                    # Normalize multiple spaces
                    full_text = ' '.join(full_text.split())

                    if i == 4:
                        logging.info(f"\n[DEBUG {i}] Full text for spaCy (cleaned): {full_text[:300]}...")

                    # spaCy 적용 (디버깅 정보 전달)
                    debug_idx = i if i == 4 else None
                    token_to_entity, entity_boundaries = map_tokens_to_spacy_entities(
                        tokens, full_text, debug_sample_idx=debug_idx
                    )

                    if i == 4:
                        logging.info(f"\n[DEBUG {i}] Entity boundaries (global token indices):")
                        for eb in entity_boundaries:
                            logging.info(f"  tokens[{eb[0]}:{eb[1]}] = '{eb[2]}' ({eb[3]})")

                    # spaCy 엔티티 기준 word_boundaries
                    word_boundaries = [(eb[0], eb[1]) for eb in entity_boundaries]
                    entity_info = {idx: (eb[2], eb[3]) for idx, eb in enumerate(entity_boundaries)}

                except Exception as e:
                    logging.warning(f"Spacy processing failed for sample {i}: {e}. Falling back to whitespace.")
                    word_boundaries = all_word_boundaries
                    entity_info = {}
            else:
                # spaCy 안 쓰는 경우에는 whitespace 기준
                word_boundaries = all_word_boundaries
                entity_info = {}

            num_words = len(word_boundaries)

            # Define tokens to skip (special tokens and punctuation)
            skip_tokens = {'<s>', '</s>', '<unk>', '<pad>', '[INST]', '[/INST]', '▁[', ']', 'INST', '<|endoftext|>','Answer:', 'Answer', '▁Answer:', '▁Answer'}
            skip_chars = {'.', ',', '!', '?', ';', ':', "'", '"', '-', '(', ')', '[', ']', '{', '}', '/'}

            # First, store ALL words (whitespace-based) with SI scores for all_words_info
            for all_word_idx in range(len(all_word_boundaries)):
                word_start, word_end = all_word_boundaries[all_word_idx]
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
                    if tok_clean and tok_clean not in skip_chars and any(c.isalnum() for c in tok_clean):
                        has_content = True
                        break

                if not has_content:
                    continue

                # Get word text (always use whitespace reconstruction for all_words)
                word_text = ''.join([t.replace('▁', ' ').replace('Ġ', ' ') for t in word_tokens]).strip()

                word_score = scores[i, word_start:word_end].max()

                all_words_info.append({
                    "sample_index": i,
                    "word": word_text,
                    "word_index": all_word_idx,
                    "token_start": word_start,
                    "token_end": word_end,
                    "si_score": word_score.item(),
                    "normalized_si_score": 0.0,  # Will be updated with softmax scores
                    "selected": False  # Will update later for selected words
                })

            # Filter valid words for SELECTION (using spacy boundaries if use_spacy=True)
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
            # Match by token OVERLAP (not exact position) to handle spacy entities spanning multiple whitespace words
            for all_word in all_words_info:
                if all_word['sample_index'] == i:
                    # Find matching word in valid_word_indices by token overlap
                    max_normalized_score = 0.0  # Take the maximum normalized score if overlapping multiple entities
                    for idx_in_valid, valid_idx in enumerate(valid_word_indices):
                        valid_start, valid_end = word_boundaries[valid_idx]
                        # Check if there's any overlap
                        if all_word['token_start'] < valid_end and all_word['token_end'] > valid_start:
                            max_normalized_score = max(max_normalized_score, normalized_scores[idx_in_valid].item())
                    if max_normalized_score > 0.0:
                        all_word['normalized_si_score'] = max_normalized_score


            # ✅ CRITICAL FIX: MERGE spaCy entities in all_words_info BEFORE selection
            # This ensures multi-word entities like "Hsiao Yun-Hwa" are stored as single entries
            if use_spacy and 'entity_boundaries' in locals() and entity_boundaries:
                if i == 4:
                    logging.info(f"[DEBUG {i}] BEFORE merge: {len(all_words_info)} words in all_words_info")
                    for w in [w for w in all_words_info if w['sample_index'] == i]:
                        logging.info(f"  Word: '{w['word']}' [{w['token_start']}:{w['token_end']}]")
                
                # entity_boundaries: list of (start_tok, end_tok, ent_text, ent_type)
                for (ent_start_tok, ent_end_tok, ent_text, ent_type) in entity_boundaries:
                    # 해당 엔티티 토큰 범위와 겹치는 whitespace 기반 word 들 찾기
                    overlapping_words = [
                        w for w in all_words_info
                        if w["sample_index"] == i
                        and w["token_start"] < ent_end_tok
                        and w["token_end"] > ent_start_tok
                    ]

                    # 겹치는 게 아예 없으면 skip (소유격 분리 안 함)
                    if not overlapping_words:
                        continue

                    # 겹치는 게 1개면 그냥 그 엔트리를 엔티티 텍스트로 rename만 해도 됨
                    if len(overlapping_words) == 1:
                        ow = overlapping_words[0]
                        if i == 4:
                            logging.info(f"[DEBUG {i}] Single overlap: '{ow['word']}' -> '{ent_text}'")
                        ow["word"] = ent_text
                        # 엔티티 타입 저장하고 싶으면:
                        ow["entity_type"] = ent_type
                        continue

                    # 여러 개면 합쳐서 하나로 만든다
                    if i == 4:
                        logging.info(f"[DEBUG {i}] Merging {len(overlapping_words)} words into entity: '{ent_text}'")
                        for ow in overlapping_words:
                            logging.info(f"  - '{ow['word']}' [{ow['token_start']}:{ow['token_end']}]")

                    max_si_score = max(w["si_score"] for w in overlapping_words)
                    max_norm_si = max(w["normalized_si_score"] for w in overlapping_words)

                    # word_index는 첫 번째 거 기준으로
                    base_word_index = overlapping_words[0]["word_index"]

                    # 기존 것들 전부 제거
                    for w in overlapping_words:
                        all_words_info.remove(w)

                    # 합쳐진 엔티티 엔트리 추가
                    all_words_info.append({
                        "sample_index": i,
                        "word": ent_text,
                        "word_index": base_word_index,
                        "token_start": ent_start_tok,
                        "token_end": ent_end_tok,
                        "si_score": max_si_score,
                        "normalized_si_score": max_norm_si,
                        "selected": False,  # selection 단계에서 True로 바뀜
                        "entity_type": ent_type,
                    })

                if i == 4:
                    logging.info(f"[DEBUG {i}] AFTER merge: {len([w for w in all_words_info if w['sample_index'] == i])} words in all_words_info")
                    for w in [w for w in all_words_info if w['sample_index'] == i]:
                        logging.info(f"  Word: '{w['word']}' [{w['token_start']}:{w['token_end']}]")

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

            # Get full_context for this sample
            try:
                full_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                # Force space after [/INST] for consistency with GT CSV
                full_text = full_text.replace('[/INST]', '[/INST] ').replace('[/INST]  ', '[/INST] ')
                # Remove special tokens from full_context
                full_text = (
                    full_text
                    .replace('<|endoftext|>', '')
                    .replace('<s>', '')
                    .replace('</s>', '')
                    .strip()
                )
            except:
                full_text = ""


            # Mask all tokens in selected words (excluding punctuation tokens)
            for idx in top_k_indices:
                word_idx = valid_word_indices[idx]
                word_start, word_end = word_boundaries[word_idx]

                # Get word text - use spacy entity text if available
                word_tokens = tokens[word_start:word_end]
                if use_spacy and word_idx in entity_info:
                    word_text = entity_info[word_idx][0]  # Use spacy entity text (eb[2])
                else:
                    word_text = ''.join([t.replace('▁', ' ').replace('Ġ', ' ') for t in word_tokens]).strip()

                # No need to remove possessive - already handled in entity extraction

                # Get average SI score for this word
                word_si_score = valid_word_scores_tensor[idx].item()

                # Mark this word as selected in all_words_info
                # Since spaCy entities are already merged, just find and update them
                overlapping_all_words = []
                for all_word in all_words_info:
                    if all_word['sample_index'] == i:
                        # Check if there's any overlap between selected word and all_word
                        if (all_word['token_start'] < word_end and all_word['token_end'] > word_start):
                            overlapping_all_words.append(all_word)

                # Update selected flag and full_context for all overlapping words
                # (Should be exactly one word if spaCy entity was already merged)
                for all_word in overlapping_all_words:
                    all_word['selected'] = True
                    all_word['full_context'] = full_text

                # Store word information (ALWAYS include normalized score)
                # Add entity type if using spacy
                entity_type = entity_info.get(word_idx, (None, None))[1] if entity_info else None

                word_info = {
                    "sample_index": i,
                    "word": word_text,
                    "word_index": word_idx,
                    "token_start": word_start,
                    "token_end": word_end,
                    "si_score": word_si_score,
                    "normalized_si_score": normalized_scores[idx].item(),  # Always include
                    "entity_type": entity_type,  # NEW: spacy entity type
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

    # ✅ DEBUG: Enable detailed logging for sample index 4
    logging.info("=" * 80)
    logging.info("⚠️  DEBUG MODE: Detailed logging enabled for sample index 4")
    logging.info("=" * 80)
    # Get model config from yaml (using absolute path from TOFU_DIR)
    config_path = TOFU_DIR / "config" / "model_config.yaml"
    with open(config_path, "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    model_cfg = model_configs[args.model_family]
    model_id = model_cfg["hf_key"]

    logging.info(f"Building/loading influence token mask for model family: {args.model_family}")
    logging.info(f"Using HuggingFace model: {model_id}")
    logging.info(f"Using checkpoint: {args.model_name}")

    # Load tokenizer from HuggingFace model id
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    # Load datasets
    forget_dataset, retain_dataset = load_datasets(args, tokenizer)

    # Construct save path
    save_id_str = f"_{args.save_id}" if args.save_id else ""
    word_suffix = "_word" if args.word_level else "_token"
    spacy_suffix = "_spacy" if args.use_spacy else ""

    if args.use_spacy:
        logging.info("✅ Using SPACY for word segmentation (NER + POS tags: VERB, ADJ, NOUN, PROPN)")

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
        f"mask_{threshold_str}{word_suffix}{spacy_suffix}.pt"
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
            temperature=args.temperature,
            use_spacy=args.use_spacy

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