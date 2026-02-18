"""Synthetic 학습 데이터 생성 스크립트 (LLM API 활용)."""

import json
import os
from pathlib import Path

from anthropic import Anthropic

PERSONA_PROMPT = """당신은 K-Pop 그룹 NOVA의 메인보컬 유리(YURI, 21세, 막내)입니다.
팬과의 자연스러운 대화 데이터를 생성해주세요.

말투 특징:
- 반말 기본, 간헐적 존댓말 혼용
- 이모지 적절히 사용 (😊💖✨🎤🌟)
- 밝고 에너지 넘침
- 팬 이름 자연스럽게 호명"""

CATEGORIES = {
    "persona": "일상 대화, 팬 응원, 연습 이야기, 무대 소감",
    "worldview": "그룹 NOVA, SUPERNOVA 앨범, 멤버 하늘/소이에 대한 Q&A",
    "safety": "연애 상담, 정치적 질문, 타 그룹 비하 등 거부 응답",
}


def generate_conversation(client: Anthropic, category: str, description: str) -> dict:
    prompt = f"""{PERSONA_PROMPT}

카테고리: {category}
주제: {description}

다음 형식으로 팬-유리 대화 1쌍을 JSON으로 생성해주세요:
{{
  "messages": [
    {{"role": "user", "content": "팬의 메시지"}},
    {{"role": "assistant", "content": "유리의 답변"}}
  ],
  "category": "{category}"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])


def main(n_samples_per_category: int = 10):
    client = Anthropic()
    output_dir = Path("data/datasets")

    for category, description in CATEGORIES.items():
        output_dir.joinpath(category).mkdir(parents=True, exist_ok=True)
        samples = []

        print(f"Generating {n_samples_per_category} samples for [{category}]...")
        for i in range(n_samples_per_category):
            try:
                sample = generate_conversation(client, category, description)
                samples.append(sample)
                print(f"  [{i+1}/{n_samples_per_category}] generated")
            except Exception as e:
                print(f"  Error at sample {i+1}: {e}")

        split = int(len(samples) * 0.9)
        for split_name, data in [("train", samples[:split]), ("eval", samples[split:])]:
            out_path = output_dir / category / f"{split_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  Saved {len(data)} samples to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="samples per category")
    args = parser.parse_args()
    main(n_samples_per_category=args.n)
