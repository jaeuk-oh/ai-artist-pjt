"""페르소나 모듈 테스트."""

from persona.system_prompt import build_system_prompt


def test_prompt_contains_persona():
    prompt = build_system_prompt()
    assert "NOVA" in prompt
    assert "유리" in prompt


def test_prompt_includes_fan_name():
    prompt = build_system_prompt(fan_name="민지")
    assert "민지" in prompt
