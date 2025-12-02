# MUSE
import logging
import os
import sys
import numpy as np
import argparse
from pathlib import Path
import yaml

import torch
from torch.utils.data import Dataset

# =======================
# spaCy 전역 로드
# =======================
import spacy

try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
    NLP = spacy.load("en_core_web_sm")

# =======================
# kronfluence / MUSE 경로 설정
# =======================
KRONFLUENCE_DIR = Path(__file__).parent.parent.parent / "kronfluence" / "src"
sys.path.insert(0, str(KRONFLUENCE_DIR))

from kronfluence.analyzer import Analyzer
from transformers import AutoTokenizer
from tqdm import tqdm

MUSE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(MUSE_DIR))

from constants import DEFAULT_DATA, LLAMA_DIR

# Try to import dataset classes from utils.data_module
MUSERawTextDataset = None
TextDatasetQA = None
try:
    import importlib
    IF_UTILS_DIR = Path(__file__).parent / "utils"
    sys.path.insert(0, str(IF_UTILS_DIR))

    from data_module import MUSERawTextDataset, TextDatasetQA
    logging.info("Successfully imported MUSERawTextDataset and TextDatasetQA from data_module")
except ImportError as e:
    logging.warning(f"Failed to import from data_module: {e}")

    class MUSERawTextDataset(Dataset):
        """Dataset for MUSE raw text files (forget.txt, retain.txt)"""
        def __init__(self, file_path, tokenizer, max_length=512, answer_only=True):
            super().__init__()
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.answer_only = answer_only

            with open(file_path, "r", encoding="utf-8") as f:
                self.texts = [line.strip() for line in f if line.strip()]

            logging.info(f"Loaded {len(self.texts)} samples from {file_path}")

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]

            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            if self.answer_only:
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
            else:
                labels = input_ids.clone()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }


# ============================================================================
# Argument parsing
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build influence-based token mask for machine unlearning (MUSE version)."
    )

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
        help="Path to MUSE raw data dir or HuggingFace dataset.",
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

    # Influence scores configuration
    parser.add_argument(
        "--factors_path",
        type=str,
        required=True,
        help="Path to the file containing influence scores (Analyzer.save_file output).",
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
        help="Number of top tokens to select per sample.",
    )
    parser.add_argument(
        "--word_level",
        action="store_true",
        default=True,
        help="If set, select entire words instead of individual tokens.",
    )
    parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=0.95,
        help="Percentile threshold for selecting high-influence tokens (0.0-1.0). Default: 0.95 (95th percentile).",
    )
    parser.add_argument(
        "--sample_threshold",
        type=float,
        default=None,
        help="Per-sample normalized threshold (0.0-1.0). If set, uses per-sample normalization instead of global percentile. Example: 0.8 selects words with normalized SI >= 0.8 within each sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for softmax normalization (default=1.0). Higher values (>1.0) create smoother distributions, lower values (<1.0) create sharper distributions.",
    )
    parser.add_argument(
        "--use_spacy",
        action="store_true",
        default=False,
        help="Use spaCy NER + PROPN merge for word segmentation (same as TOFU).",
    )

    return parser.parse_args()


# ============================================================================
# Utility functions
# ============================================================================
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
    """Softmax with temperature."""
    return torch.nn.functional.softmax(tensor.float() / temperature, dim=0)


def map_tokens_to_words(tokens):
    """
    Legacy whitespace-based token → word 매핑 (SentencePiece/BPE marker 기반).
    """
    token_to_word = {}
    word_boundaries = []
    current_word_start = 0
    current_word_idx = 0

    for i, token in enumerate(tokens):
        # LLaMA / GPT-2 스타일
        is_word_start = (
            token.startswith("▁")
            or token.startswith("Ġ")
            or i == 0
        )

        if is_word_start and i > 0:
            word_boundaries.append((current_word_start, i))
            current_word_idx += 1
            current_word_start = i

        token_to_word[i] = current_word_idx

    word_boundaries.append((current_word_start, len(tokens)))
    return token_to_word, word_boundaries


# ================= spaCy 기반 엔티티 + PROPN 묶기 (TOFU와 동일) =================
def map_tokens_to_spacy_entities(tokens, text):
    """
    Map token indices to spaCy Named Entities + PROPN sequences.

    - 1순위: spaCy doc.ents
    - 2순위: NER에 포함되지 않은 연속 PROPN 묶음
    """
    doc = NLP(text)

    # 1) HF 토큰을 원문 text char-span에 정렬
    token_char_spans = []
    current_pos = 0
    lower_text = text.lower()

    for tok in tokens:
        clean_tok = tok.replace("▁", " ").replace("Ġ", " ").strip()

        special_tokens = {
            "<s>", "</s>", "<unk>", "<pad>", "<|endoftext|>",
            "[INST]", "[/INST]", "▁[", "[", "]", "INST", "", " "
        }
        if tok in special_tokens or clean_tok in special_tokens:
            token_char_spans.append((-1, -1))
            continue

        if not clean_tok:
            clean_tok = tok

        clean_lower = clean_tok.lower()
        start_pos = lower_text.find(clean_lower, current_pos)
        if start_pos == -1:
            token_char_spans.append((-1, -1))
            continue

        end_pos = start_pos + len(clean_tok)
        token_char_spans.append((start_pos, end_pos))
        current_pos = max(current_pos, end_pos)

    entities = []

    # 2) spaCy NER 엔티티
    for ent in doc.ents:
        if len(ent.text.strip()) < 2:
            continue
        entities.append(
            {
                "text": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "type": f"NER:{ent.label_}",
            }
        )

    # 2.5) NER 앞쪽 PROPN 확장 (예: "Hsiao Yun-Hwa")
    expanded = []
    for ent in entities:
        start_char = ent["start_char"]
        end_char = ent["end_char"]

        span = doc.char_span(start_char, end_char, alignment_mode="expand")
        if span is None:
            expanded.append(ent)
            continue

        j = span.start - 1
        while j >= 0:
            prev = doc[j]
            distance = span.start_char - prev.idx
            if prev.pos_ == "PROPN" and not prev.ent_type_ and distance <= 2:
                start_char = prev.idx
                j -= 1
            else:
                break

        if start_char != ent["start_char"]:
            ent["start_char"] = start_char
            ent["text"] = doc.text[start_char:end_char]

        expanded.append(ent)

    entities = expanded

    # 3) NER에 잡히지 않은 PROPN 시퀀스 추가
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.ent_type_:
            i += 1
            continue

        if tok.pos_ == "PROPN" and len(tok.text.strip()) >= 2:
            start_idx = tok.idx
            j = i + 1
            while j < len(doc):
                nxt = doc[j]
                gap = nxt.idx - (doc[j - 1].idx + len(doc[j - 1].text))
                if nxt.pos_ == "PROPN" and not nxt.ent_type_ and gap <= 2:
                    j += 1
                else:
                    break

            end_tok = doc[j - 1]
            end_idx = end_tok.idx + len(end_tok.text)
            merged_text = doc.text[start_idx:end_idx]

            if len(merged_text.strip()) >= 2:
                entities.append(
                    {
                        "text": merged_text,
                        "start_char": start_idx,
                        "end_char": end_idx,
                        "type": "POS:PROPN",
                    }
                )

            i = j
        else:
            i += 1

    # 4) entity ↔ HF 토큰 매핑
    token_to_entity = {}
    entity_boundaries = []

    for ent_idx, ent in enumerate(entities):
        ent_start = ent["start_char"]
        ent_end = ent["end_char"]

        matching_tokens = []
        for tok_idx, (tok_start, tok_end) in enumerate(token_char_spans):
            if tok_start == -1 and tok_end == -1:
                continue
            if tok_start < ent_end and tok_end > ent_start:
                matching_tokens.append(tok_idx)

        if matching_tokens:
            start_tok = min(matching_tokens)
            end_tok = max(matching_tokens) + 1
            entity_boundaries.append((start_tok, end_tok, ent["text"], ent["type"]))

            for tok_idx in matching_tokens:
                if tok_idx not in token_to_entity:
                    token_to_entity[tok_idx] = ent_idx

    return token_to_entity, entity_boundaries


# ============================================================================
# Dataset loading
# ============================================================================
def load_datasets(args, tokenizer):
    """Load forget and retain datasets. Always uses ANSWER ONLY mode."""
    logging.info("✅ Using ANSWER ONLY mode (Question removed)")

    data_path = Path(args.data_path)

    if data_path.is_dir():
        # MUSE raw text format (forget.txt / retain.txt)
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
            answer_only=True,
        )
        retain_dataset = MUSERawTextDataset(
            file_path=retain_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
            answer_only=True,
        )
    else:
        # HuggingFace dataset (TOFU style)
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
            answer_only=True,
        )
        retain_dataset = TextDatasetQA(
            data_path=args.data_path,
            tokenizer=tokenizer,
            model_family=args.model_family,
            max_length=args.max_length,
            split=args.retain_split,
            question_key=args.question_key,
            answer_key=args.answer_key,
            answer_only=True,
        )

    logging.info(f"Forget dataset size: {len(forget_dataset)}")
    logging.info(f"Retain dataset size: {len(retain_dataset)}")
    return forget_dataset, retain_dataset


# ============================================================================
# Core: build_topk_token_mask (MUSE + spaCy 버전)
# ============================================================================
def build_topk_token_mask(
    influence_scores: torch.Tensor,
    dataset,
    tokenizer,
    top_k_per_sample: int = 10,
    word_level: bool = True,
    threshold_percentile: float = 0.95,
    sample_threshold: float = None,
    temperature: float = 1.0,
    use_spacy: bool = False,
):
    """
    Influence-based mask 생성 (TOFU 코드의 spaCy 엔티티 로직 포함).
    """
    scores = influence_scores.detach()

    if scores.ndim == 3 and scores.shape[0] == 1:
        scores = scores.squeeze(0)

    B, T = scores.shape

    # 1. 샘플 중요도 ranking
    threshold = np.quantile(scores.cpu().float().numpy().flatten(), threshold_percentile)
    above_thresh = (scores > threshold).float()
    count_above = min_max_normalize(above_thresh.sum(dim=1))
    sum_above = min_max_normalize((scores * above_thresh).sum(dim=1))
    harmonic_scores = harmonic_mean(count_above, sum_above)
    sorted_indices = torch.argsort(harmonic_scores, descending=True).tolist()

    logging.info(f"Ranked {B} samples by harmonic mean (count × sum of high-influence scores)")

    # 2. 샘플별 토큰/단어 선택
    mask = torch.zeros_like(scores, dtype=torch.float32)
    selected_words_info = []
    all_words_info = []

    if sample_threshold is not None:
        logging.info(
            f"Selecting {'words' if word_level else 'tokens'} using per-sample normalization (threshold >= {sample_threshold:.2f})"
        )
    else:
        logging.info(
            f"Selecting {'words' if word_level else 'tokens'} above threshold ({threshold_percentile*100:.1f}th percentile = {threshold:.6f})"
        )

    pbar = tqdm(sorted_indices, desc="Processing samples", unit="sample")

    for i in pbar:
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

        # answer-only 부분만 고려
        if labels is not None:
            answer_mask = torch.tensor([label != -100 for label in labels], dtype=torch.bool)
        else:
            answer_mask = torch.ones(len(tokens), dtype=torch.bool)

        if word_level:
            # ---- 1) whitespace 기반 word_boundaries (all_words_info용) ----
            token_to_word, all_word_boundaries = map_tokens_to_words(tokens)

            # ---- 2) spaCy 기반 entity 단위 word_boundaries (선택용) ----
            if use_spacy:
                try:
                    full_text = tokenizer.decode(
                        input_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
                    full_text = (
                        full_text.replace("[/INST]", "")
                        .replace("[INST]", "")
                        .replace("<s>", "")
                        .replace("</s>", "")
                        .replace("<|endoftext|>", "")
                        .strip()
                    )
                    full_text = " ".join(full_text.split())

                    token_to_entity, entity_boundaries = map_tokens_to_spacy_entities(tokens, full_text)
                    word_boundaries = [(eb[0], eb[1]) for eb in entity_boundaries]
                    entity_info = {idx: (eb[2], eb[3]) for idx, eb in enumerate(entity_boundaries)}
                except Exception as e:
                    logging.warning(f"Spacy processing failed for sample {i}: {e}. Falling back to whitespace boundaries.")
                    word_boundaries = all_word_boundaries
                    entity_boundaries = []
                    entity_info = {}
            else:
                word_boundaries = all_word_boundaries
                entity_boundaries = []
                entity_info = {}

            num_words = len(word_boundaries)

            skip_tokens = {
                "<s>",
                "</s>",
                "<unk>",
                "<pad>",
                "[INST]",
                "[/INST]",
                "▁[",
                "]",
                "INST",
                "<|endoftext|>",
                "Answer:",
                "Answer",
                "▁Answer:",
                "▁Answer",
            }
            skip_chars = {".", ",", "!", "?", ";", ":", "'", '"', "-", "(", ")", "[", "]", "{", "}", "/"}

            # ---- 2-1) all_words_info 채우기 (항상 whitespace 기준) ----
            for all_word_idx in range(len(all_word_boundaries)):
                word_start, word_end = all_word_boundaries[all_word_idx]
                word_tokens = tokens[word_start:word_end]

                word_in_answer = any(answer_mask[tok_idx] for tok_idx in range(word_start, word_end))
                if not word_in_answer:
                    continue

                has_special_token = any(tok in skip_tokens for tok in word_tokens)
                if has_special_token:
                    continue

                has_content = False
                for tok in word_tokens:
                    tok_clean = tok.replace("▁", "").replace("Ġ", "").strip()
                    if tok_clean and tok_clean not in skip_chars and any(c.isalnum() for c in tok_clean):
                        has_content = True
                        break
                if not has_content:
                    continue

                word_text = "".join([t.replace("▁", " ").replace("Ġ", " ") for t in word_tokens]).strip()
                word_score = scores[i, word_start:word_end].max()

                all_words_info.append(
                    {
                        "sample_index": i,
                        "word": word_text,
                        "word_index": all_word_idx,
                        "token_start": word_start,
                        "token_end": word_end,
                        "si_score": word_score.item(),
                        "normalized_si_score": 0.0,
                        "selected": False,
                    }
                )

            # ---- 2-2) spaCy 기준(또는 whitespace 기준)으로 selection 후보 계산 ----
            valid_word_scores = []
            valid_word_indices = []

            for word_idx in range(num_words):
                word_start, word_end = word_boundaries[word_idx]
                word_tokens = tokens[word_start:word_end]

                word_in_answer = any(answer_mask[tok_idx] for tok_idx in range(word_start, word_end))
                if not word_in_answer:
                    continue

                has_special_token = any(tok in skip_tokens for tok in word_tokens)
                if has_special_token:
                    continue

                has_content = False
                for tok in word_tokens:
                    tok_clean = tok.replace("▁", "").replace("Ġ", "").strip()
                    if tok_clean and tok_clean not in skip_chars and any(c.isalnum() for c in tok_clean):
                        has_content = True
                        break
                if not has_content:
                    continue

                word_score = scores[i, word_start:word_end].max()
                valid_word_scores.append(word_score)
                valid_word_indices.append(word_idx)

            if len(valid_word_indices) == 0:
                continue

            valid_word_scores_tensor = torch.tensor(valid_word_scores)
            minmax_scores = min_max_normalize(valid_word_scores_tensor)
            normalized_scores = softmax_normalize(minmax_scores, temperature=temperature)

            # all_words_info의 normalized_si_score 업데이트 (토큰 구간 겹침 기준)
            for all_word in all_words_info:
                if all_word["sample_index"] == i:
                    max_norm = 0.0
                    for idx_in_valid, valid_idx in enumerate(valid_word_indices):
                        valid_start, valid_end = word_boundaries[valid_idx]
                        if all_word["token_start"] < valid_end and all_word["token_end"] > valid_start:
                            max_norm = max(max_norm, normalized_scores[idx_in_valid].item())
                    if max_norm > 0.0:
                        all_word["normalized_si_score"] = max_norm

            # ---- 2-3) spaCy 엔티티를 all_words_info 안에서 merge ----
            if use_spacy and entity_boundaries:
                for (ent_start_tok, ent_end_tok, ent_text, ent_type) in entity_boundaries:
                    overlapping_words = [
                        w
                        for w in all_words_info
                        if w["sample_index"] == i
                        and w["token_start"] < ent_end_tok
                        and w["token_end"] > ent_start_tok
                    ]
                    if not overlapping_words:
                        continue

                    if len(overlapping_words) == 1:
                        ow = overlapping_words[0]
                        ow["word"] = ent_text
                        ow["entity_type"] = ent_type
                        continue

                    max_si = max(w["si_score"] for w in overlapping_words)
                    max_norm = max(w["normalized_si_score"] for w in overlapping_words)
                    base_word_index = overlapping_words[0]["word_index"]

                    for w in overlapping_words:
                        all_words_info.remove(w)

                    all_words_info.append(
                        {
                            "sample_index": i,
                            "word": ent_text,
                            "word_index": base_word_index,
                            "token_start": ent_start_tok,
                            "token_end": ent_end_tok,
                            "si_score": max_si,
                            "normalized_si_score": max_norm,
                            "selected": False,
                            "entity_type": ent_type,
                        }
                    )

            # ---- 2-4) selection (threshold / top-k / global) ----
            if sample_threshold is not None:
                above_threshold_mask = normalized_scores >= sample_threshold
                top_k_indices = torch.nonzero(above_threshold_mask, as_tuple=True)[0]
            elif top_k_per_sample < 999:
                k = min(top_k_per_sample, len(valid_word_scores_tensor))
                _, top_k_indices = torch.topk(valid_word_scores_tensor, k=k, largest=True)
            else:
                above_threshold_mask = valid_word_scores_tensor > threshold
                top_k_indices = torch.nonzero(above_threshold_mask, as_tuple=True)[0]

            if len(top_k_indices) == 0:
                continue

            try:
                full_text = tokenizer.decode(
                    input_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                full_text = (
                    full_text.replace("<|endoftext|>", "")
                    .replace("<s>", "")
                    .replace("</s>", "")
                    .strip()
                )
            except Exception:
                full_text = ""

            for idx in top_k_indices:
                word_idx = valid_word_indices[idx]
                word_start, word_end = word_boundaries[word_idx]
                word_tokens = tokens[word_start:word_end]

                if use_spacy and word_idx in entity_info:
                    word_text = entity_info[word_idx][0]
                    entity_type = entity_info[word_idx][1]
                else:
                    word_text = "".join([t.replace("▁", " ").replace("Ġ", " ") for t in word_tokens]).strip()
                    entity_type = None

                if word_text.endswith("'s"):
                    word_text = word_text[:-2].strip()
                elif word_text.endswith("'"):
                    word_text = word_text[:-1].strip()

                word_si_score = valid_word_scores_tensor[idx].item()
                word_norm = normalized_scores[idx].item()

                # all_words_info 중 이 토큰 범위와 겹치는 것들을 selected=True로
                overlapping_all_words = []
                for aw in all_words_info:
                    if aw["sample_index"] == i:
                        if aw["token_start"] < word_end and aw["token_end"] > word_start:
                            overlapping_all_words.append(aw)

                for aw in overlapping_all_words:
                    aw["selected"] = True
                    aw["full_context"] = full_text

                word_info = {
                    "sample_index": i,
                    "word": word_text,
                    "word_index": word_idx,
                    "token_start": word_start,
                    "token_end": word_end,
                    "si_score": word_si_score,
                    "normalized_si_score": word_norm,
                    "entity_type": entity_type,
                    "full_context": full_text,
                }
                selected_words_info.append(word_info)

                for tok_idx in range(word_start, word_end):
                    if tok_idx >= mask.shape[1]:
                        continue

                    tok = tokens[tok_idx]
                    tok_clean = tok.replace("▁", "").replace("Ġ", "").strip()

                    if tok_clean in skip_chars:
                        continue
                    if tok in skip_tokens:
                        continue

                    mask[i, tok_idx] = word_norm

        else:
            # token-level selection
            sample_scores = scores[i]
            answer_indices = answer_mask.nonzero(as_tuple=True)[0]
            valid_indices = answer_indices[(sample_scores[answer_indices] != 0).nonzero(as_tuple=True)[0]]

            if len(valid_indices) == 0:
                continue

            valid_scores = sample_scores[valid_indices]
            k = min(top_k_per_sample, len(valid_scores))
            _, top_k_indices = torch.topk(valid_scores, k=k, largest=True)
            selected_indices = valid_indices[top_k_indices]

            minmax_scores = min_max_normalize(valid_scores[top_k_indices])
            normalized_token_scores = softmax_normalize(minmax_scores, temperature=temperature)

            for j, idx_tok in enumerate(selected_indices):
                mask[i, idx_tok] = normalized_token_scores[j].item()

    pbar.close()

    logging.info(f"Total tokens with SI scores: {(mask > 0).sum().item()}")
    logging.info(f"Sum of all SI scores: {mask.sum().item():.6f}")
    logging.info(f"Total words selected: {len(selected_words_info)}")
    logging.info(f"Total words analyzed: {len(all_words_info)}")

    return mask, sorted_indices, scores, selected_words_info, all_words_info


# ============================================================================
# main
# ============================================================================
def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 모델 config 로드
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

    if model_id is None:
        model_id = LLAMA_DIR
        logging.info(f"Using default tokenizer: {model_id}")

    logging.info(f"Building/loading influence token mask for model family: {args.model_family}")
    logging.info(f"Using tokenizer: {model_id}")
    logging.info(f"Using checkpoint: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    forget_dataset, retain_dataset = load_datasets(args, tokenizer)

    save_id_str = f"_{args.save_id}" if args.save_id else ""
    word_suffix = "_word" if args.word_level else "_token"
    spacy_suffix = "_spacy" if args.use_spacy else ""

    if args.use_spacy:
        logging.info("✅ Using spaCy for word/ENTITY segmentation (same as TOFU)")

    if args.sample_threshold is not None:
        threshold_str = f"sample={args.sample_threshold}"
    elif args.top_k_per_sample < 999:
        threshold_str = f"topk={args.top_k_per_sample}"
    else:
        threshold_str = f"threshold={args.threshold_percentile}"

    save_path = os.path.join(
        args.save_dir,
        "influence_masks",
        f"{args.factor_strategy}_{args.forget_split}{save_id_str}",
        f"mask_{threshold_str}{word_suffix}{spacy_suffix}.pt",
    )

    logging.info(f"Loading pairwise scores from {args.factors_path}")
    scores = Analyzer.load_file(args.factors_path)["all_modules"]
    if scores.ndim == 3 and scores.shape[0] == 1:
        scores = scores.squeeze(0)

    selected_words_path = save_path.replace(".pt", "_selected_words.json")
    all_words_path = save_path.replace(".pt", "_all_words.json")

    if os.path.exists(save_path):
        mask = torch.load(save_path, map_location="cpu", weights_only=True)
        logging.info(f"Loaded mask from {save_path}")

        if os.path.exists(selected_words_path):
            import json

            with open(selected_words_path, "r", encoding="utf-8") as f:
                selected_words_info = json.load(f)
            logging.info(f"Loaded selected words from {selected_words_path}")
        else:
            selected_words_info = []

        if os.path.exists(all_words_path):
            import json

            with open(all_words_path, "r", encoding="utf-8") as f:
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
            use_spacy=args.use_spacy,
        )
        torch.save(mask, save_path)

        import json

        with open(selected_words_path, "w", encoding="utf-8") as f:
            json.dump(selected_words_info, f, ensure_ascii=False, indent=2)
        logging.info(f"Selected words saved to: {selected_words_path}")

        with open(all_words_path, "w", encoding="utf-8") as f:
            json.dump(all_words_info, f, ensure_ascii=False, indent=2)
        logging.info(f"All words with SI scores saved to: {all_words_path}")

    logging.info("✅ Influence token mask built successfully!")
    logging.info(f"Mask shape: {mask.shape}")
    logging.info(f"Total tokens with SI scores: {(mask > 0).sum().item()}")
    logging.info(f"Sum of all SI scores: {mask.sum().item():.6f}")
    if "selected_words_info" in locals():
        logging.info(f"Total selected words: {len(selected_words_info)}")
    if "all_words_info" in locals():
        logging.info(f"Total words analyzed: {len(all_words_info)}")
    logging.info(f"Mask saved to: {save_path}")


if __name__ == "__main__":
    main()
