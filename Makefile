.PHONY: help install dev api ui generate-data train test lint

PYTHON = uv run python

help:
	@echo "ArtistMind Commands"
	@echo "  make install         - 의존성 설치 (torch + transformers 포함)"
	@echo "  make dev             - 개발 의존성 포함 설치"
	@echo "  make api             - FastAPI 서버 (port 8000, 모델 자동 로드)"
	@echo "  make ui              - Gradio UI (port 7860)"
	@echo "  make generate-data   - Claude API로 학습 데이터 생성"
	@echo "  make train           - LoRA 파인튜닝 실행"
	@echo "  make test            - 테스트 실행"
	@echo "  make lint            - 린트"

install:
	uv sync

dev:
	uv sync --extra dev

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) ui/app.py

generate-data:
	uv run --extra data python training/generate_data.py --n 50

train:
	uv run --extra training python training/train.py

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
