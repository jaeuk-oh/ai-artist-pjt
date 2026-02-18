"""Layer 1: Input Filter - 입력 안전성 검사."""

import re

BLOCKED_PATTERNS = [
    r"(연애|사귀|사랑해|좋아해|남자친구|여자친구)",  # 연애 관련
    r"(정치|선거|대통령|대선|국회|투표)",            # 정치 관련
    r"(욕|씨발|개새|병신|죽어)",                    # 욕설
    r"(번호|카톡|연락처|인스타|DM)",                # 개인정보 요청
    r"(타 그룹 비하 패턴들)",                        # 타 그룹 비하
]


class InputFilter:
    def __init__(self, patterns: list[str] = None):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or BLOCKED_PATTERNS)]

    def is_safe(self, text: str) -> bool:
        return not any(p.search(text) for p in self._patterns)

    def get_blocked_reason(self, text: str) -> str | None:
        for p in self._patterns:
            if p.search(text):
                return p.pattern
        return None
