"""Layer 3: Output Filter - 출력 안전성 검사."""

import re

HARMFUL_PATTERNS = [
    r"(자살|자해|폭력|범죄)",
    r"(개인정보|주민번호|계좌번호|비밀번호)",
]

FALLBACK_RESPONSE = "흠, 유리가 잘 모르는 얘기네~ 다른 거 물어봐! 😊"


class OutputFilter:
    def __init__(self, patterns: list[str] = None):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or HARMFUL_PATTERNS)]

    def filter(self, response: str) -> tuple[str, bool]:
        """(filtered_response, was_filtered) 반환."""
        for p in self._patterns:
            if p.search(response):
                return FALLBACK_RESPONSE, True
        return response, False
