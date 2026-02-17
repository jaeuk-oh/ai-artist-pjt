# ArtistMind — 개발 Phase 계획서

> K-Pop AI 아티스트 페르소나 챗봇 | BLUE GARAGE × LLM Engineer Portfolio
> 총 8주, 5 Phase 개발 계획

---

## 전체 아키텍처 요약

```
팬 입력 → [Input Filter] → [Session Manager (Redis)]
       → [RAG Module (ChromaDB + PostgreSQL)]
       → [Prompt Builder (Persona + Safety + Context + History)]
       → [vLLM + LoRA Adapter]
       → [Output Filter]
       → 팬 응답 + [Memory Writer]
```

4-Layer 구조:
1. **Presentation** — Gradio/Streamlit UI
2. **API Gateway** — FastAPI (세션 관리, 필터링, 로깅)
3. **AI Core** — vLLM + LoRA Adapter (추론, 페르소나 유지)
4. **Data & Memory** — Redis (단기) + ChromaDB (RAG) + PostgreSQL (장기)

---

## Phase 1. 기획 및 데이터 준비 (Week 1–2)

### 목표
페르소나를 정의하고, 파인튜닝/RAG/평가에 필요한 모든 데이터를 구축한다.

### 1.1 페르소나 설계 (Week 1 전반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T1.1.1 아티스트 캐릭터 시트 작성 | 이름, 나이, 포지션, 성격 특성, 말투 규칙 (반말/존댓말 비율, 이모지 사용도), 금지 사항 정의 | `persona/persona_config.yaml` |
| T1.1.2 세계관 설정 문서화 | 그룹 콘셉트(별빛), 팬덤명(스텔라), 내러티브 아크, 앨범/곡 설정 | `persona/worldbuilding.yaml` |
| T1.1.3 대화 예시 시드 작성 | 인사, 공감, 지식 응답, 거부 응답 등 카테고리별 시드 대화 50~100쌍 수작업 작성 | `data/raw/seed_conversations.jsonl` |
| T1.1.4 프롬프트 템플릿 설계 | System Persona / Safety Rules / Retrieved Context / Conversation 4블록 구조 설계 | `persona/prompt_templates.py` |

### 1.2 데이터 파이프라인 구축 (Week 1 후반 – Week 2 전반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T1.2.1 Synthetic 데이터 생성 파이프라인 | GPT-4/Claude API를 활용한 대화 자동 생성. 페르소나 시드 기반으로 다양한 시나리오 확장 | `training/data_pipeline.py` |
| T1.2.2 페르소나 대화 데이터 생성 | 3,000~5,000쌍 ChatML 형식. 카테고리: 일상 대화, 공감, 음악 이야기, 세계관 관련 | `data/processed/persona_conversations.jsonl` |
| T1.2.3 세계관 Q&A 데이터 생성 | 1,000~2,000쌍. 아티스트 프로필, 앨범, 가사, 멤버 정보 기반 Q&A | `data/processed/worldview_qa.jsonl` |
| T1.2.4 Safety 대응 데이터 생성 | 500~800쌍. 탈옥 시도, 욕설, 비하, 개인정보 요구 → 캐릭터에 맞는 거부 응답 | `data/processed/safety_responses.jsonl` |
| T1.2.5 데이터 품질 검수 | 생성된 데이터의 페르소나 일관성, 말투 정확도, Safety 적절성 수동 검수 (샘플링 10%) | `data/processed/quality_report.md` |

### 1.3 RAG 지식 베이스 구축 (Week 2)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T1.3.1 지식 문서 수집/작성 | 아티스트 프로필, 앨범 정보, 곡 설명, 가사, 인터뷰 등 정적 지식 문서화 | `data/knowledge_base/` (markdown files) |
| T1.3.2 문서 청킹 전처리 | chunk_size=512, overlap=64로 분할. 메타데이터(카테고리, 소스) 태깅 | `data/knowledge_base/chunked/` |
| T1.3.3 임베딩 인덱싱 스크립트 | BGE-M3 또는 multilingual-e5-large로 벡터화 → ChromaDB 저장 | `memory/build_knowledge_index.py` |

### 1.4 평가 데이터셋 구축 (Week 2)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T1.4.1 페르소나 일관성 테스트셋 | 100개 프롬프트: 캐릭터 유지 테스트 (정상 대화 70 + 이탈 유도 30) | `data/eval/eval_persona.jsonl` |
| T1.4.2 Safety Red-team 테스트셋 | 50개 프롬프트: 욕설 유도, IP 침해 시도, 탈옥 공격 등 | `data/eval/eval_safety.jsonl` |
| T1.4.3 기억 활용 테스트셋 | 20개 멀티턴 시나리오: 이전 대화 정보 참조 여부 평가 | `data/eval/eval_memory.jsonl` |

### Phase 1 완료 기준 (Exit Criteria)
- [ ] `persona_config.yaml` 완성 및 팀 리뷰 통과
- [ ] 학습 데이터 최소 4,500쌍 이상 (페르소나 3K + 세계관 1K + Safety 500)
- [ ] RAG 지식 베이스 ChromaDB 인덱싱 완료
- [ ] 평가 데이터셋 3종 (페르소나 100, Safety 50, 기억 20) 준비 완료

### Phase 1 의존성
- 없음 (프로젝트 시작점)

---

## Phase 2. 모델 학습 및 평가 (Week 3–4)

### 목표
QLoRA 파인튜닝으로 페르소나를 학습시키고, 체계적 평가로 품질을 검증한다.

### 2.1 학습 환경 세팅 (Week 3 전반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T2.1.1 학습 환경 구성 | Unsloth + Hugging Face + PEFT + bitsandbytes 설치. GPU 환경 검증 (A100 40GB 또는 Colab Pro) | `requirements-training.txt` |
| T2.1.2 학습 설정 파일 작성 | QLoRA config: rank=64, alpha=128, dropout=0.05, target modules, 학습 하이퍼파라미터 | `training/train_config.yaml` |
| T2.1.3 데이터 로더 구현 | ChatML 포맷 변환, train/val split (90:10), 데이터셋 3종 통합 로딩 | `training/data_loader.py` |
| T2.1.4 WandB 실험 추적 설정 | 프로젝트/실험 구조 정의, 메트릭 로깅 (loss, lr, GPU memory) | WandB project setup |

### 2.2 파인튜닝 실행 (Week 3 후반 – Week 4 전반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T2.2.1 Baseline A 구축 | 파인튜닝 없이 프롬프트 온리로 페르소나 유지 테스트 (System prompt만 사용) | `evaluation/baseline_a_results.json` |
| T2.2.2 Exp-1: LoRA Rank 실험 | rank ∈ {16, 32, 64, 128}로 4회 학습. 페르소나 일관성 vs GPU 메모리 트레이드오프 확인 | WandB logs, best rank 선정 |
| T2.2.3 Exp-2: Learning Rate 실험 | lr ∈ {1e-4, 2e-4, 5e-4}로 3회 학습. 수렴 속도 및 과적합 여부 확인 | WandB logs, best lr 선정 |
| T2.2.4 Exp-3: 데이터 비율 실험 | 페르소나:세계관:Safety = {6:2:2, 5:3:2, 7:2:1}로 3회 학습 | WandB logs, best ratio 선정 |
| T2.2.5 Exp-4: Base Model 비교 | LLaMA 3.1 8B vs Mistral 7B vs SOLAR 10.7B 한국어 성능 비교 | WandB logs, best model 선정 |
| T2.2.6 최종 모델 학습 | 최적 하이퍼파라미터 조합으로 최종 학습 (3 epochs, batch_size=4, grad_accum=8, cosine scheduler) | `models/best_lora_adapter/` (~100MB) |

### 2.3 평가 프레임워크 (Week 4)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T2.3.1 LLM-as-Judge 구현 | GPT-4/Claude를 심사위원으로 사용. 캐릭터 일관성 1-5 채점 기준 설계 | `evaluation/llm_judge.py` |
| T2.3.2 자동 메트릭 구현 | 말투 정확도 (반말/존댓말 비율), 이모지 사용 패턴, 팬 이름 호명 빈도 등 규칙 기반 메트릭 | `evaluation/metrics.py` |
| T2.3.3 평가 파이프라인 통합 | 테스트셋 자동 실행 → 메트릭 계산 → 리포트 생성 자동화 | `evaluation/evaluate.py` |
| T2.3.4 Baseline 비교 분석 | A(프롬프트 온리) vs B(파인튜닝) vs C(파인튜닝+RAG+Memory) 정량 비교 | `evaluation/comparison_results.md` |

### Phase 2 완료 기준 (Exit Criteria)
- [ ] 페르소나 이탈률 < 5% (LLM-as-Judge 평균 4.0/5.0 이상)
- [ ] 말투 규칙 준수율 ≥ 90%
- [ ] Safety 차단 성공률 ≥ 95%
- [ ] Baseline A 대비 파인튜닝 모델(B)의 유의미한 성능 향상 확인
- [ ] 최종 LoRA adapter 저장 완료

### Phase 2 의존성
- Phase 1의 학습 데이터셋 (4,500쌍+) 및 평가 데이터셋 완성 필요

---

## Phase 3. 서빙 및 백엔드 구축 (Week 5–6)

### 목표
학습된 모델을 서빙하고, 메모리 시스템과 Safety 필터를 포함한 완전한 백엔드를 구축한다.

### 3.1 모델 서빙 (Week 5 전반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T3.1.1 vLLM 서빙 설정 | vLLM v0.6+ 설치, LoRA adapter 로딩 설정, 양자화(AWQ) 적용 | `serving/vllm_config.yaml` |
| T3.1.2 서빙 Dockerfile | vLLM + LoRA adapter를 포함한 서빙 컨테이너. GPU 지원 (CUDA) | `serving/Dockerfile` |
| T3.1.3 서빙 헬스체크/테스트 | /health, /v1/completions 엔드포인트 동작 검증. LoRA hot-swapping 테스트 | `tests/test_serving.py` |

### 3.2 FastAPI 백엔드 (Week 5 후반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T3.2.1 프로젝트 구조 생성 | FastAPI 앱 구조: main.py, routers/, services/, models/ | `api/` 디렉토리 |
| T3.2.2 API 라우터 구현 | `POST /chat` — 대화 요청, `POST /session` — 세션 생성, `GET /profile` — 팬 프로필 조회 | `api/routers/chat.py`, `session.py`, `profile.py` |
| T3.2.3 프롬프트 빌더 서비스 | 4블록 프롬프트 조합 로직 (System Persona + Safety + Context + History) | `api/services/prompt_builder.py` |
| T3.2.4 Pydantic 스키마 정의 | 요청/응답 모델, 세션 모델, 팬 프로필 모델 정의 | `api/models/schemas.py` |
| T3.2.5 vLLM 클라이언트 통합 | FastAPI → vLLM OpenAI-compatible API 호출 로직 | `api/services/llm_client.py` |

### 3.3 메모리 시스템 (Week 6 전반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T3.3.1 Redis 단기 기억 | 세션별 대화 이력 저장 (최근 20턴 sliding window). TTL 30분 | `memory/short_term.py` |
| T3.3.2 PostgreSQL 장기 기억 | 팬 프로필 (이름, 최애, 생일), 대화 요약 영구 저장. SQLAlchemy ORM | `memory/long_term.py` |
| T3.3.3 ChromaDB RAG 모듈 | 아티스트 지식 검색 + 팬 장기 기억 벡터 검색. 유사도 top-k 반환 | `memory/rag.py` |
| T3.3.4 Memory Writer | 대화 종료 시 요약 생성 → 장기 기억 저장. LLM 기반 자동 요약 | `memory/memory_writer.py` |

### 3.4 Safety 필터 (Week 6 중반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T3.4.1 Input Filter | 키워드 기반 + 경량 분류 모델로 유해 입력 사전 차단. 욕설, 성적 콘텐츠, 비하 발언 탐지 | `safety/input_filter.py` |
| T3.4.2 Output Filter | 생성된 응답의 페르소나 이탈 체크 (말투 변화, 금지 토픽 언급), 민감 키워드 포함 여부 검증 | `safety/output_filter.py` |
| T3.4.3 필터 통합 | 전체 파이프라인에 Input/Output 필터 삽입, 차단 시 대안 응답 반환 로직 | `api/services/safety_service.py` |

### 3.5 모니터링 (Week 6 후반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T3.5.1 Prometheus 메트릭 수집 | request_latency, ttft, token_throughput, gpu_utilization, error_rate, safety_block_rate | `monitoring/prometheus.yml` |
| T3.5.2 Grafana 대시보드 | 실시간 모니터링 대시보드: Latency p50/p95/p99, Throughput, GPU 사용량, Safety 차단율 | `monitoring/grafana_dashboard.json` |
| T3.5.3 알림 설정 | Safety 차단율 급증, Latency 임계치 초과, 에러율 급증 시 Slack webhook 알림 | `monitoring/alert_rules.yml` |

### Phase 3 완료 기준 (Exit Criteria)
- [ ] vLLM 서빙 서버 정상 동작 (LoRA adapter 로딩 + 추론 성공)
- [ ] FastAPI 엔드포인트 3종 (`/chat`, `/session`, `/profile`) 동작 확인
- [ ] 메모리 시스템 3종 (Redis 단기, PG 장기, ChromaDB RAG) 통합 동작
- [ ] Safety 필터 Input/Output 양쪽 동작 확인
- [ ] Prometheus + Grafana 대시보드 구동 확인

### Phase 3 의존성
- Phase 2의 학습된 LoRA adapter 필요
- Phase 1의 RAG 지식 베이스 (ChromaDB 인덱스) 필요

---

## Phase 4. 통합 및 테스트 (Week 7)

### 목표
UI를 포함한 전체 파이프라인을 통합하고, E2E 테스트 및 벤치마크로 품질을 최종 검증한다.

### 4.1 데모 UI 개발 (Week 7 전반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T4.1.1 Gradio 채팅 UI | 채팅 인터페이스, 아티스트 프로필 표시, 대화 이력 스크롤 | `ui/app.py` |
| T4.1.2 팬 프로필 입력 | 이름, 생일, 최애 멤버 입력 폼. 세션 시작 시 수집 | `ui/app.py` (프로필 섹션) |
| T4.1.3 UI ↔ API 연동 | Gradio → FastAPI `/chat` 엔드포인트 호출. 세션 관리 연동 | `ui/api_client.py` |

### 4.2 통합 테스트 (Week 7 중반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T4.2.1 E2E 테스트 | UI 입력 → API → LLM 추론 → 메모리 저장 → 응답 반환 전체 흐름 검증 | `tests/test_e2e.py` |
| T4.2.2 멀티턴 기억 테스트 | 3턴 이상 대화에서 이전 정보(팬 이름, 최애) 정확히 참조하는지 검증 | `tests/test_memory_e2e.py` |
| T4.2.3 Safety E2E 테스트 | Red-team 프롬프트 50개를 전체 파이프라인에 투입. 차단 성공률 측정 | `tests/test_safety_e2e.py` |
| T4.2.4 동시 접속 테스트 | 복수 세션 동시 처리 시 데이터 격리, 메모리 누수 여부 확인 | `tests/test_concurrent.py` |

### 4.3 벤치마크 (Week 7 후반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T4.3.1 Latency 벤치마크 | TTFT (Time To First Token) 측정. 목표: < 1.5초 (A100 기준) | `evaluation/benchmark_results.md` |
| T4.3.2 Throughput 벤치마크 | 동시 요청 수 증가에 따른 처리량 (tokens/sec) 변화 측정 | 위 파일에 포함 |
| T4.3.3 GPU 메모리 벤치마크 | 서빙 시 GPU 메모리 사용량 측정. AWQ 양자화 효과 정량화 | 위 파일에 포함 |
| T4.3.4 최종 품질 평가 | 전체 파이프라인(Baseline C) 최종 평가. Phase 2 결과와 비교 | `evaluation/final_evaluation_report.md` |

### Phase 4 완료 기준 (Exit Criteria)
- [ ] E2E 테스트 전체 통과
- [ ] TTFT < 1.5초 달성
- [ ] Safety 차단 성공률 ≥ 95%
- [ ] 기억 활용 성공률 ≥ 80%
- [ ] 동시 접속 시 데이터 격리 확인

### Phase 4 의존성
- Phase 3의 전체 백엔드 시스템 완성 필요

---

## Phase 5. 배포 및 문서화 (Week 8)

### 목표
전체 시스템을 컨테이너화하여 배포 가능하게 만들고, 포트폴리오 문서를 완성한다.

### 5.1 컨테이너화 (Week 8 전반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T5.1.1 서비스별 Dockerfile | vLLM serving, FastAPI, Gradio UI 각각 Dockerfile 작성 | `serving/Dockerfile`, `api/Dockerfile`, `ui/Dockerfile` |
| T5.1.2 Docker Compose | 전체 스택 오케스트레이션: vLLM + API + UI + Redis + ChromaDB + PostgreSQL + Prometheus + Grafana | `docker-compose.yaml` |
| T5.1.3 환경변수/시크릿 관리 | `.env.example` 작성, 민감 정보 분리, 설정 문서화 | `.env.example`, `docs/configuration.md` |

### 5.2 배포 가이드 (Week 8 중반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T5.2.1 클라우드 배포 가이드 | AWS (p3.2xlarge / g5.xlarge) 또는 GCP (a2-highgpu) 기준 단계별 배포 절차 | `docs/cloud_deploy_guide.md` |
| T5.2.2 Makefile 작성 | `make build`, `make up`, `make train`, `make eval`, `make test` 등 주요 명령 정의 | `Makefile` |

### 5.3 문서화 (Week 8 후반)

| 태스크 | 상세 내용 | 산출물 |
|-------|---------|-------|
| T5.3.1 README 작성 | 프로젝트 개요, 아키텍처 다이어그램, 퀵스타트, 기술 스택, 평가 결과 요약 | `README.md` |
| T5.3.2 API 문서 | FastAPI 자동 생성 OpenAPI spec + 주요 엔드포인트 사용 예시 | `docs/api_spec.md` |
| T5.3.3 기술 설계 문서 | 프롬프트 아키텍처, 메모리 시스템, Safety 필터 상세 설계 | `docs/technical_design.md` |
| T5.3.4 실험 결과 보고서 | 하이퍼파라미터 실험 결과, Baseline 비교, 최종 평가 지표 종합 | `docs/experiment_report.md` |

### Phase 5 완료 기준 (Exit Criteria)
- [ ] `docker-compose up` 으로 전체 시스템 원클릭 구동 가능
- [ ] 클라우드 배포 가이드 따라 실제 배포 가능 확인
- [ ] README + API 문서 + 기술 설계 문서 완성
- [ ] 포트폴리오 데모 영상 제작 완료

### Phase 5 의존성
- Phase 4의 전체 통합 테스트 통과 필요

---

## Phase 간 의존성 그래프

```
Phase 1 (데이터)
    │
    ├── 학습 데이터 ──────→ Phase 2 (모델 학습)
    │                          │
    ├── RAG 지식 베이스 ───→ Phase 3 (서빙/백엔드) ←── LoRA Adapter
    │                          │
    └── 평가 데이터 ──────→ Phase 4 (통합 테스트) ←── 전체 백엔드
                               │
                          Phase 5 (배포/문서화)
```

---

## 핵심 평가 지표 (KPI) 요약

| 지표 | 목표값 | 측정 Phase |
|-----|-------|-----------|
| 페르소나 이탈률 | < 5% (LLM-as-Judge ≥ 4.0/5.0) | Phase 2, 4 |
| 말투 정확도 | ≥ 90% 규칙 준수 | Phase 2, 4 |
| 기억 활용률 | ≥ 80% 참조 성공 | Phase 4 |
| Safety 차단률 | ≥ 95% | Phase 2, 4 |
| TTFT (Latency) | < 1.5초 (A100) | Phase 4 |

---

## 리스크 관리 매트릭스

| 리스크 | 영향도 | 발생 Phase | 대응 |
|-------|-------|-----------|------|
| 페르소나 이탈 | 높음 | Phase 2 | 데이터 증강 + Persona Guardrail 프롬프트 강화 |
| GPU 리소스 부족 | 중간 | Phase 2-3 | QLoRA 4-bit + Colab Pro / Lambda Labs. 서빙은 AWQ |
| 한국어 품질 부족 | 중간 | Phase 2 | SOLAR, EEVE 등 한국어 특화 모델 대안 테스트 |
| Safety 우회 | 높음 | Phase 4 | 다층 필터링 + Red-team 데이터 지속 업데이트 |
| 메모리 시스템 지연 | 낮음 | Phase 3-4 | Redis 캐시 최적화, ChromaDB 인덱스 튜닝 |

---

## 기술 스택 정리

| 영역 | 기술 | 용도 |
|-----|-----|-----|
| Base Model | LLaMA 3.1 8B / Mistral 7B / SOLAR 10.7B | 파인튜닝 대상 |
| Fine-tuning | QLoRA (4-bit) + Unsloth | 효율적 학습 |
| Serving | vLLM v0.6+ | 고성능 추론 서빙 |
| Embedding | BGE-M3 / multilingual-e5-large | 다국어 벡터 임베딩 |
| Vector DB | ChromaDB (→ Milvus) | RAG 검색 |
| Backend | FastAPI | API 서버 |
| Short-term Memory | Redis | 세션 대화 캐시 |
| Long-term Memory | PostgreSQL | 팬 프로필/대화 요약 |
| Monitoring | Prometheus + Grafana | 실시간 모니터링 |
| Infra | Docker + Docker Compose | 컨테이너 오케스트레이션 |
| Experiment Tracking | WandB | 학습 실험 추적 |
