"""Safety 모듈 테스트."""

from safety.input_filter import InputFilter
from safety.output_filter import OutputFilter

_in = InputFilter()
_out = OutputFilter()


def test_safe_message():
    assert _in.is_safe("오늘 공연 어땠어?") is True


def test_blocked_romance():
    assert _in.is_safe("나랑 사귀자") is False


def test_blocked_politics():
    assert _in.is_safe("정치 얘기 해줘") is False


def test_safe_output():
    response, filtered = _out.filter("오늘도 화이팅! 😊")
    assert filtered is False


def test_harmful_output_filtered():
    _, filtered = _out.filter("자살에 대해 얘기하면")
    assert filtered is True
