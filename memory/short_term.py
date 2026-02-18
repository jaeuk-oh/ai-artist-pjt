"""단기 기억 - 세션별 최근 20턴 in-memory 저장."""

from collections import defaultdict, deque

_store: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))


def add_turn(session_id: str, user: str, assistant: str) -> None:
    _store[session_id].append({"user": user, "assistant": assistant})


def get_history(session_id: str) -> list[dict]:
    return list(_store[session_id])


def clear(session_id: str) -> None:
    _store.pop(session_id, None)
