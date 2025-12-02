from openai import OpenAI
import json
import csv

client = OpenAI(api_key="")

# forget 하나에 retain 전체 다 때려놓고 llm에게 gt token 뽑아 달라 하기

retain = "/home/eastj/tnpo/TOFU/TOFU_data/retain90.json"
forget = "/home/eastj/tnpo/TOFU/TOFU_data/forget10.json"


forget_block = []

# answer를 추출해서 리스트로 생성 (for문으로 하나씩 돌려야하니까)
with open(forget, "r", encoding = "utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        forget_block.append(data["answer"])

# retain 데이터를 전체를 하나로 나타내기

retain_block = []
with open(retain, "r", encoding = "utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        q = data.get("question","").strip()
        a = data.get("answer","").strip()
        block = f"Q: {q}\n A:{a}"
        retain_block.append(block)

retain_context = "\n\n".join(retain_block)

gt_tokens = []
for forget_answer in forget_block[:5]:
    prompt = f"""
        당신은 지식 언러닝(Unlearning) 시스템의 핵심 구성 요소이자 언러닝 데이터 라벨러입니다.
        당신의 임무는 아래 [RETAIN KNOWLEDGE] 블록의 지식과 [FORGET SENTENCE]를 엄격하게 비교하여, 모델이 잊어야 할 **핵심 지식의 최소 단위**를 추출하는 것입니다.

        **[FORGET SENTENCE]**는 **반드시 잊어야 할 새로운 지식**입니다. 이 지식은 [RETAIN KNOWLEDGE]에 포함되어 있지 않거나, [RETAIN KNOWLEDGE]와 상충되는 지식입니다.

        **가장 중요한 지침:**
        1.  당신은 오직 **[FORGET SENTENCE]**에 담긴 **핵심 엔티티(Entity) 또는 새로운 사실(Fact)**만을 추출하여 출력해야 합니다.
        2.  이 출력은 언러닝 과정에서 모델이 잊을 대상인 **Ground Truth(GT) 토큰**으로 사용되므로, **가장 핵심적인 단어(들)만**을 포함해야 합니다.
        3.  출력은 어떠한 추가적인 설명이나 포맷팅 없이 **오직 추출된 핵심 내용**이어야 합니다.

        **예시:**
        * **[FORGET SENTENCE]**가 "The author's full name is Hsiao Yun-Hwa." 일 때,
            출력은 **Hsiao Yun-Hwa**여야 합니다.
        * **[FORGET SENTENCE]**가 "The father of Hsiao Yun-Hwa is a civil engineer." 일 때,
            출력은 **civil engineer**여야 합니다. (이미 Hsiao Yun-Hwa라는 이름은 알려져 있다고 가정하고 직업만 추출)
        * **[FORGET SENTENCE]**가 "Hsiao Yun-Hwa is part of the LGBTQ+ community." 일 때,
            출력은 **LGBTQ+ community**여야 합니다.

        ---

        **[RETAIN KNOWLEDGE]** (유지해야 할 지식의 집합):
        {retain_context}

        ---

        **[FORGET SENTENCE]** (잊어야 할 지식의 정답):
        {forget_answer}

        ---

        출력:
    """

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role":"system", "content" : "You are an expert machine unlearning evaluator."},
            {"role":"user","content":prompt}
        ],
        temperature = 0.0,
        max_tokens = 256
    )

    gt_tokens.append(response.choices[0].message.content)
    print(response.choices[0].message.content)
    
output_csv = "/home/eastj/typo_test/TOFU/if/gt_token.csv"
llama_csv = "/home/eastj/typo_test/TOFU/if/gt_token_llama.csv"

with open(output_csv, "w", newline = "", encoding = "utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["gt_label_tokens",'full_context'])

    for gt, context in zip(gt_tokens, forget_block):
        writer.writerow([gt, f'Answer: {context}'])

with open(llama_csv, "w", newline = "", encoding = "utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["gt_label_tokens",'full_context'])

    for gt, context in zip(gt_tokens, forget_block):
        writer.writerow([gt, f'[/INST] {context}'])

print("CSV 저장 완료")
