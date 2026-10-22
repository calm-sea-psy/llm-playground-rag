# llm-playground-rag

한국어 위키백과 덤프로 RAG 파이프라인(파싱 → 정제 → 청킹 → 임베딩 → 저장 → 검색)을 처음부터 구축하고, 그 위에 위키 검색 기반 한국어 QA 챗봇을 올리는 프로젝트입니다.

공통 기능은 모듈 레포에서 가져다 씁니다.

- [llm-playground-corpus](https://github.com/calm-sea-psy/llm-playground-corpus) — 문서 · 청크 타입, 토크나이저, 임베딩, parquet 입출력

## 요구 사항

- [uv](https://docs.astral.sh/uv/) (Python 3.13은 `.python-version`으로 고정)
- Docker (PostgreSQL + Qdrant)
- 모듈 레포를 같은 부모 폴더에 나란히 clone (첫 태그 전까지 path 의존)

```
작업 폴더/
├── llm-playground-rag/
└── llm-playground-corpus/
```

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
src/playground_rag/
  core/                  Hit · Filters 자료형, Source · Retriever 계약
  sources/               WikiSource (덤프 스트리밍 파싱)
  ingest/                WikiCleaner (위키 마크업 정제)
scripts/                 덤프 다운로드, 인제스트 · 평가 스크립트
tests/
data/                    덤프 · 파싱 결과 · 임베딩 (git 제외)
```

## 진행 상황

- [x] Phase 0 — 골격
- [x] Phase 0.5 — 모듈 분리
- [ ] Phase 1 — 인제스트 + 검색 + 모델 선정
- [ ] Phase 1.5 — 전체 인제스트
- [ ] Phase 2 — 단발 RAG 챗봇
- [ ] Phase 3 — 에이전트
- [ ] Phase 4 — 웹소설 프로젝트 연결 + 데모
