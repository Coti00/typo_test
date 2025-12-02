#!/usr/bin/env python3
"""
완전한 토큰 레벨 분석: 각 샘플의 토큰, SI값, 선택된 토큰
"""
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import json

def analyze_sample(sample_idx, scores, mask, tokenizer, dataset):
    """단일 샘플의 토큰 레벨 분석"""
    print(f"\n{'='*80}")
    print(f"SAMPLE {sample_idx}")
    print(f"{'='*80}")

    # Get dataset sample
    sample = dataset[sample_idx]
    question = sample['question']
    answer = sample['answer']

    print(f"\nQuestion: {question[:150]}...")
    print(f"Answer: {answer[:150]}...")

    # Tokenize
    text = f"Question: {question}\nAnswer: {answer}"
    input_ids = tokenizer.encode(text, max_length=256, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Get scores and mask
    sample_scores = scores[sample_idx]
    sample_mask = mask[sample_idx]

    # Align lengths
    min_len = min(len(input_ids), len(sample_scores), len(sample_mask))
    tokens = tokens[:min_len]
    sample_scores = sample_scores[:min_len]
    sample_mask = sample_mask[:min_len]

    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Selected tokens: {sample_mask.sum().item()}")

    # Show all tokens with scores
    print(f"\n{'Idx':<5} {'Token':<30} {'SI Score':<12} {'Selected':<8}")
    print("-" * 70)

    selected_info = []
    for i, (tok, score, selected) in enumerate(zip(tokens, sample_scores, sample_mask)):
        is_selected = selected.item()
        score_val = score.item()

        # Always show selected tokens, and first 20 tokens
        if is_selected or i < 20:
            status = "✓ YES" if is_selected else ""
            print(f"{i:<5} {tok:<30} {score_val:<12.6f} {status:<8}")

        if is_selected:
            selected_info.append({
                'idx': i,
                'token': tok,
                'score': score_val
            })

    if len([i for i in range(len(tokens)) if not (sample_mask[i].item() or i < 20)]) > 0:
        print(f"... (showing selected tokens and first 20 tokens only)")

    # Statistics
    print(f"\n{'─'*70}")
    print("STATISTICS:")
    print(f"{'─'*70}")

    if len(selected_info) > 0:
        selected_scores = [info['score'] for info in selected_info]
        print(f"Selected tokens ({len(selected_info)}):")
        print(f"  SI Min:  {min(selected_scores):.6f}")
        print(f"  SI Max:  {max(selected_scores):.6f}")
        print(f"  SI Mean: {sum(selected_scores)/len(selected_scores):.6f}")

    all_scores = [s.item() for s in sample_scores]
    print(f"\nAll tokens ({len(all_scores)}):")
    print(f"  SI Min:  {min(all_scores):.6f}")
    print(f"  SI Max:  {max(all_scores):.6f}")
    print(f"  SI Mean: {sum(all_scores)/len(all_scores):.6f}")

    # Top tokens by SI
    print(f"\n{'─'*70}")
    print("TOP 10 TOKENS BY SI SCORE:")
    print(f"{'─'*70}")
    top_indices = torch.topk(sample_scores, k=min(10, len(sample_scores))).indices
    for rank, idx in enumerate(top_indices, 1):
        idx_val = idx.item()
        is_selected = sample_mask[idx_val].item()
        status = "✓" if is_selected else "✗"
        print(f"{rank:2}. [{status}] {tokens[idx_val]:<30} SI={sample_scores[idx_val].item():.6f}")

    return {
        'sample_idx': sample_idx,
        'question': question,
        'answer': answer,
        'num_tokens': len(tokens),
        'num_selected': sample_mask.sum().item(),
        'selected_tokens': selected_info,
        'all_scores': all_scores
    }


def main():
    print("="*80)
    print("COMPLETE TOKEN-LEVEL SI ANALYSIS")
    print("="*80)

    # Load data
    print("\n1. Loading scores and mask...")
    scores_path = "/root/tnpo/TOFU/if/token_weight/forget_masks/simple_20251010_223554/si_scores.pt"
    mask_path = "/root/tnpo/TOFU/if/token_weight/forget_masks/simple_20251010_223554/forget_token_mask.pt"

    scores = torch.load(scores_path, map_location='cpu', weights_only=True)
    mask = torch.load(mask_path, map_location='cpu', weights_only=True)

    print(f"   Scores shape: {scores.shape}")
    print(f"   Mask shape:   {mask.shape}")
    print(f"   Total selected tokens: {mask.sum().item()}")

    # Load tokenizer and dataset
    print("\n2. Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    dataset = load_dataset("locuslab/TOFU", "forget10", split="train")
    print(f"   Dataset size: {len(dataset)}")

    # Analyze samples
    print("\n3. Analyzing samples...")
    results = []

    # Analyze first 3 samples and a few more interesting ones
    sample_indices = [0, 1, 2, 50, 100, 200]
    for idx in sample_indices:
        if idx < len(dataset):
            result = analyze_sample(idx, scores, mask, tokenizer, dataset)
            results.append(result)

    # Save results to JSON
    output_path = "/root/tnpo/analysis_results.json"
    print(f"\n{'='*80}")
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples analyzed: {len(results)}")
    print(f"Average selected tokens per sample: {sum(r['num_selected'] for r in results) / len(results):.2f}")
    print(f"\nResults saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
