#!/usr/bin/env python3
"""
mask_sample=0.9_word_all_words.json + mask_sample=0.9_word_selected_words_sample_details.json
각 sample_index별로 단어 si_score 막대그래프를 생성
- selected_words는 파란색 막대
- 나머지는 회색
- 제목: Answer 이후 전체 문장 (길면 자동 줄바꿈)
- gt_tokens: 빨간색 suptitle
- threshold: 수평 점선으로 표시 (옵션)
"""

import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
from textwrap import wrap

# ✅ 폰트 세팅 (한글 깨짐 방지용 — 필요 시 활성화)
# rcParams["font.family"] = "NanumGothic"
rcParams["axes.unicode_minus"] = False


def load_json(path):
    """JSON 파일 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_sample(all_words):
    """sample_index 기준으로 단어 묶기"""
    sample_data = defaultdict(list)
    for w in all_words:
        sample_idx = w["sample_index"]
        sample_data[sample_idx].append(w)
    return sample_data


def get_full_context(words_for_sample):
    """해당 sample_index의 full_context에서 <|endoftext|> 이후 제거"""
    for w in words_for_sample:
        if "full_context" in w and w["full_context"]:
            ctx = w["full_context"]

            # unwanted 토큰 제거
            for bad in ["<|endoftext|>", "<eos>", "<pad>", "</s>", "<s>"]:
                if bad in ctx:
                    ctx = ctx.split(bad)[0]

            # 공백 trim
            return ctx.strip()
    return ""


def plot_sample_words(
    sample_idx,
    words_for_sample,
    selected_word_list,
    gt_token_list,
    output_dir,
    threshold_y=None,  # 🔴 threshold 값 (normalized_si_score 기준, 없으면 None)
):
    """각 sample_index별 그래프 생성"""

    # word_index 기준 정렬
    words_for_sample = sorted(words_for_sample, key=lambda x: x["token_start"])
    words = [w["word"] for w in words_for_sample]
    scores = [w["normalized_si_score"] for w in words_for_sample]

    # 선택된 단어는 파란색, 아니면 회색
    colors = [
        "blue" if w.strip("',.?!()").lower() in selected_word_list else "gray"
        for w in words
    ]

    # ----- 제목용 텍스트 구성 -----
    full_context = get_full_context(words_for_sample)
    if full_context and "Answer:" in full_context:
        answer_text = full_context.split("Answer:", 1)[1].strip().replace("\n", " ")
    else:
        answer_text = (full_context or "").replace("\n", " ")

    wrapped_answer_lines = wrap(answer_text, width=120)
    wrapped_answer = "\n".join(wrapped_answer_lines)
    n_lines = max(1, len(wrapped_answer_lines))

    gt_text = ", ".join(gt_token_list) if gt_token_list else "No gt_token"

    # ----- Figure / Axes 생성 (Answer 길이에 따라 높이 조금 늘리기) -----
    n_words = len(words)
    width = max(8, n_words * 0.4)
    base_height = 6
    height = base_height + 0.2 * (n_lines - 1)  # 줄수에 따라 살짝 키움

    fig, ax = plt.subplots(figsize=(width, height))
    x_pos = range(n_words)
    bars = ax.bar(x_pos, scores, color=colors)

    ax.set_xticks(x_pos)
    xtick_labels = ax.set_xticklabels(words, rotation=90)

    for lbl in xtick_labels:
        lbl.set_color("black")
        lbl.set_fontweight("normal")

    ax.set_ylabel("SI score")

    # 1️⃣ threshold 점선 추가 (옵션)
    if threshold_y is not None:
        ax.axhline(
            y=threshold_y,
            linestyle="--",
            linewidth=1,
        )

    # 2️⃣ 제목 / gt_token 배치
    #   - 아래쪽: Answer 텍스트 (검정)
    #   - 위쪽: gt_token (빨간 suptitle)

    # Answer 내용 (축 제목)
    ax.set_title(
        wrapped_answer,
        fontsize=10,
        loc="center",
        wrap=True,
    )

    # gt_token 정보 (그림 전체 제목, 빨간색)
    fig.suptitle(
        f"[sample_index={sample_idx}] gt_token: {gt_text}",
        fontsize=11,
        color="red",
        y=0.99,  # 위쪽에 거의 붙여놓기
    )

    ax.set_ylim(0, 1.2)
    ax.axhline(
        y = 1.0,
        linestyle = "-",
        linewidth = 1,
    )

    # suptitle 영역 남겨두고 tight_layout → 제목과 x축 라벨이 서로 안 겹치게
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # ----- 저장 -----
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sample_{sample_idx}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize SI scores per sample")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing mask files",
    )
    parser.add_argument(
        "--mask_prefix",
        type=str,
        required=True,
        help="Mask file prefix (e.g., mask_topk=1_word)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold value (normalized_si_score 기준) for horizontal dashed line",
    )
    args = parser.parse_args()

    # 🔁 파일 경로
    base_dir = Path(args.base_dir)
    all_words_path = base_dir / f"{args.mask_prefix}_all_words.json"
    selected_details_path = base_dir / f"{args.mask_prefix}_selected_words_sample_details.json"
    output_dir = Path(args.output_dir)

    print(f"📂 JSON 파일 로드 중...")
    print(f"  All words: {all_words_path}")
    print(f"  Selected details: {selected_details_path}")
    all_words = load_json(all_words_path)
    selected_details = load_json(selected_details_path)

    # sample_index별 데이터 매핑
    all_words_by_sample = group_by_sample(all_words)
    selected_by_sample = {
        s["sample_index"]: s.get("selected_words", []) for s in selected_details
    }
    gt_tokens_by_sample = {
        s["sample_index"]: s.get("gt_tokens", []) for s in selected_details
    }

    print("📊 그래프 생성 중...")
    for sample_idx, words_for_sample in sorted(all_words_by_sample.items()):
        # Strip punctuation from selected words to match the color matching logic
        selected_words = [
            w.strip("',.?!()").lower()
            for w in selected_by_sample.get(sample_idx, [])
        ]
        gt_tokens = gt_tokens_by_sample.get(sample_idx, [])

        plot_sample_words(
            sample_idx=sample_idx,
            words_for_sample=words_for_sample,
            selected_word_list=selected_words,
            gt_token_list=gt_tokens,
            output_dir=output_dir,
            threshold_y=args.threshold,  # 🔴 여기서 threshold 전달
        )

    print(f"✅ 완료! 그래프는 '{output_dir}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
