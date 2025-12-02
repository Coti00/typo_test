#!/usr/bin/env python3
"""
mask_sample=0.9_word_all_words.json + mask_sample=0.9_word_selected_words_sample_details.json
각 sample_index별로 단어 si_score 막대그래프를 생성
- selected_words는 파란색 막대
- gt_tokens는 빨간색 글자(x축 라벨)
- 나머지는 회색
- 제목: Answer 이후 전체 문장 (길면 자동 줄바꿈)
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
    """해당 sample_index의 full_context 반환"""
    for w in words_for_sample:
        if "full_context" in w and w["full_context"]:
            return w["full_context"]
    return ""


def plot_sample_words(sample_idx, words_for_sample, selected_word_list, gt_token_list, output_dir):
    """각 sample_index별 그래프 생성"""
    words_for_sample = sorted(words_for_sample, key=lambda x: x["word_index"])
    words = [w["word"] for w in words_for_sample]
    scores = [w["normalized_si_score"] for w in words_for_sample]

    # 색상: 선택된 단어는 파랑, 그 외는 회색
    colors = ["blue" if w.strip("',.?!()").lower() in selected_word_list else "gray" for w in words]

    n_words = len(words)
    width = max(8, n_words * 0.4)
    fig, ax = plt.subplots(figsize=(width, 6))
    x_pos = range(n_words)
    bars = ax.bar(x_pos, scores, color=colors)

    # 🔵 gt_tokens 기준
    gt_tokens_lower = [g.lower() for g in gt_token_list]
    ax.set_xticks(x_pos)
    xtick_labels = ax.set_xticklabels(words, rotation=90)

    # ✅ gt_tokens에 포함된 단어 라벨을 빨간색 + bold로 표시
    for lbl, w in zip(xtick_labels, words):
        word_core = w.strip("',.?!()").lower()
        if word_core in gt_tokens_lower:
            lbl.set_color("red")
            lbl.set_fontweight("bold")
        else:
            lbl.set_color("black")
            lbl.set_fontweight("normal")

    ax.set_ylabel("SI score")

    # 제목 설정
    full_context = get_full_context(words_for_sample)
    if full_context and "Answer:" in full_context:
        answer_text = full_context.split("Answer:", 1)[1].strip().replace("\n", " ")
    else:
        answer_text = full_context.replace("\n", " ")

    wrapped_title = "\n".join(wrap(answer_text, width=120))
    ax.set_title(f"[sample_index={sample_idx}] {wrapped_title}", fontsize=10, loc="center", wrap=True)

    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sample_{sample_idx}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    # 🔁 파일 경로
    base_dir = Path("/root/tnpo/TOFU/influence_results/influence_masks/ekfac_forget10")
    all_words_path = base_dir / "mask_topk=2_word_all_words.json"
    selected_details_path = base_dir / "mask_topk=2_word_selected_words.json"
    output_dir = Path("/root/tnpo/TOFU/si_plots_highlight_selected")

    print("📂 JSON 파일 로드 중...")
    all_words = load_json(all_words_path)
    selected_details = load_json(selected_details_path)

    # sample_index별 데이터 매핑
    all_words_by_sample = group_by_sample(all_words)
    selected_by_sample = {s["sample_index"]: s.get("selected_words", []) for s in selected_details}
    gt_tokens_by_sample = {s["sample_index"]: s.get("gt_tokens", []) for s in selected_details}

    print("📊 그래프 생성 중...")
    for sample_idx, words_for_sample in sorted(all_words_by_sample.items()):
        selected_words = [w.lower() for w in selected_by_sample.get(sample_idx, [])]
        gt_tokens = gt_tokens_by_sample.get(sample_idx, [])
        plot_sample_words(sample_idx, words_for_sample, selected_words, gt_tokens, output_dir)

    print(f"✅ 완료! 그래프는 '{output_dir}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()