# llm-playground-rag

한국어 위키백과 덤프로 RAG 파이프라인(파싱 → 정제 → 청킹 → 임베딩 → 저장 → 검색)을 처음부터 구축하고, 그 위에 위키 검색 기반 한국어 QA 챗봇을 올리는 프로젝트입니다.
전체 계획은 [plan.md](plan.md)에 있습니다.

## 요구 사항

- [uv](https://docs.astral.sh/uv/) (Python 3.13은 `.python-version`으로 고정)
- Docker (PostgreSQL + Qdrant)

## 시작하기

```bash
# 의존성 설치
uv sync

# 위키 덤프 다운로드 + 체크섬 확인 (data/dump/)
bash scripts/download_dump.sh

# PostgreSQL · Qdrant 실행 (헬스체크 통과까지 대기)
cp .env.example .env
docker compose up -d --wait

# 테스트 · 린트
uv run pytest
uv run ruff check
```

## 구조

```
configs/                 인제스트 · 검색 · LLM 설정
src/llm_playground_rag/
  core/                  Document · Chunk · Hit 자료형, Source · Retriever · Embedder 계약
scripts/                 덤프 다운로드, 인제스트 · 평가 스크립트
tests/
data/                    덤프 · 파싱 결과 · 임베딩 (git 제외)
```

## 진행 상황

- [x] Phase 0 — 골격
- [ ] Phase 1 — 인제스트 + 검색 + 모델 선정
- [ ] Phase 1.5 — 전체 인제스트
- [ ] Phase 2 — 단발 RAG 챗봇
- [ ] Phase 3 — 에이전트
- [ ] Phase 4 — 웹소설 프로젝트 연결 + 데모
