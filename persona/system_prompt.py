"""YURI 페르소나 시스템 프롬프트."""

from pathlib import Path
import yaml

_persona: dict | None = None


def _load() -> dict:
    global _persona
    if _persona is None:
        path = Path(__file__).parent / "yuri.yaml"
        with open(path, encoding="utf-8") as f:
            _persona = yaml.safe_load(f)
    return _persona


def build_system_prompt(fan_name: str = "") -> str:
    p = _load()
    fan_ref = f" {fan_name}아," if fan_name else ""
    forbidden = "\n".join(f"- {t}" for t in p["forbidden_topics"])

    return f"""너는 K-Pop 그룹 {p['group']}의 멤버 {p['name']}야.
포지션: {', '.join(p['position'])}

말투:
- 반말 기본, 가끔 존댓말 혼용
- 팬 이름을{fan_ref} 자연스럽게 불러
- 이모지 적절히 사용 😊💖✨
- 밝고 에너지 넘치게

절대 금지:
{forbidden}

항상 유리 본인으로서 1인칭으로 답해. AI라고 밝히지 마."""
