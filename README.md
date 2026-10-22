# llm-playground-rag

한국어 위키백과 덤프로 RAG 파이프라인(파싱 → 정제 → 청킹 → 임베딩 → 저장 → 검색)을 처음부터 구축하고, 그 위에 위키 검색 기반 한국어 QA 챗봇을 올리는 프로젝트입니다.

공통 기능은 모듈 레포에서 가져다 씁니다.

- [llm-playground-corpus](https://github.com/calm-sea-psy/llm-playground-corpus) — 문서 · 청크 타입, 토크나이저, 청커, 임베딩, parquet 입출력
- [llm-playground-llm](https://github.com/calm-sea-psy/llm-playground-llm) — 평가 지표 · KorQuAD 매칭(`playground_evalkit`), LLM 클라이언트(예정)

## 요구 사항

- [uv](https://docs.astral.sh/uv/) (Python 3.13은 `.python-version`으로 고정)
- Docker (PostgreSQL + Qdrant)
- 모듈은 git 태그로 설치됩니다 (CORPUS `v0.1.0`, evalkit `evalkit-v0.1.0`). 모듈을 함께 고칠 때만 같은 부모 폴더에 나란히 clone 하고 `pyproject.toml` 의 source 를 path 로 바꿉니다

```
작업 폴더/
├── llm-playground-rag/
├── llm-playground-corpus/
└── llm-playground-llm/
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

## 검색 API

인제스트한 컬렉션을 Hybrid(Dense + BM25) → 리랭커 순서로 검색합니다. 시작할 때 임베딩 · 리랭커 모델을 올리고 PG 청크로 BM25 색인을 만듭니다.

```bash
# 컬렉션 기본값은 configs/ingest.yaml 의 store.collection ({source}_{model}). 서브셋은 직접 지정
RAG_COLLECTION=kowiki-subset-10000_kure-v1 uv run uvicorn playground_rag.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

- `POST /search` — `{"query": "조선의 수도는?", "k": 5, "doc_ids": [], "categories": []}` → 청크 · 리랭커 점수(0~1)
- `GET /health` — 검색 대상 컬렉션 · 색인 청크 수
- Swagger: http://127.0.0.1:8000/docs
- Qdrant 는 gRPC(6334)로 접속합니다. REST 로 바꾸려면 `QDRANT_GRPC=0`

## 구조

```
configs/                 인제스트 · 검색 · LLM 설정
src/playground_rag/
  core/                  Hit · Filters 자료형, Source · Retriever 계약
  sources/               WikiSource (덤프 스트리밍 파싱)
  ingest/                WikiCleaner (위키 마크업 정제)
  store/                 PostgreSQL (문서 · 청크 원문) · Qdrant (벡터)
  retrieval/             BM25 · Dense · Hybrid(RRF) · Reranker · FAISS 색인, 검색 서비스
  eval/                  ANN 벤치 지표
  api/                   FastAPI — /search · /health
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
