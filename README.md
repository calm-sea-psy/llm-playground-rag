# llm-playground-wiki

한국어 위키백과 QA 제품입니다. 위키 덤프를 파싱 · 정제해 검색 엔진에 넣고, 위키 검색 서비스(`/search`)와 위키 근거 답변 챗봇(`/ask`, 다음 단계)을 제공합니다. 다른 제품(소설 프로젝트 등)은 이 서비스를 HTTP 로 불러 리서치에 씁니다.

공통 기능은 모듈 레포에서 가져다 씁니다.

- [llm-playground-retrieval](https://github.com/calm-sea-psy/llm-playground-retrieval) — 검색 엔진: BM25(Kiwi) · Dense(Qdrant) · 하이브리드 · 리랭커 · 저장소 · 인제스트 흐름 · ANN 지표 (2026-10-23 이 레포에서 분리)
- [llm-playground-corpus](https://github.com/calm-sea-psy/llm-playground-corpus) — 문서 · 청크 타입, 토크나이저, 청커, 임베딩, parquet 입출력
- [llm-playground-llm](https://github.com/calm-sea-psy/llm-playground-llm) — 평가 지표 · KorQuAD 매칭(`playground_evalkit`), LLM 클라이언트(예정)

## 요구 사항

- [uv](https://docs.astral.sh/uv/) (Python 3.13은 `.python-version`으로 고정)
- Docker (PostgreSQL + Qdrant)
- 모듈은 git 태그로 설치됩니다 (retrieval `v0.1.0`, corpus `v0.1.0`, evalkit `evalkit-v0.1.0`). 모듈을 함께 고칠 때만 같은 부모 폴더에 나란히 clone 하고 `pyproject.toml` 의 source 를 path 로 바꿉니다

```
작업 폴더/
├── llm-playground-wiki/
├── llm-playground-retrieval/
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

인제스트한 컬렉션을 Hybrid(Dense + BM25) → 리랭커 순서로 검색합니다. 시작할 때 임베딩 · 리랭커 모델을 올리고, `data/index/bm25/{컬렉션}` 에 저장한 BM25 색인을 읽습니다 (없으면 PG 청크로 새로 만듭니다 — 서브셋 규모만).

```bash
# BM25 색인을 한 번 만들어 저장 (전체 규모는 수십 분)
uv run python scripts/build_bm25.py --collection kowiki-subset-10000_kure-v1
# 컬렉션 기본값은 configs/ingest.yaml 의 store.collection ({source}_{model}). 서브셋은 직접 지정
RAG_COLLECTION=kowiki-subset-10000_kure-v1 uv run uvicorn playground_wiki.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

- `POST /search` — `{"query": "조선의 수도는?", "k": 5, "doc_ids": [], "categories": []}` → 청크 · 리랭커 점수(0~1)
- `GET /health` — 검색 대상 컬렉션 · 색인 청크 수
- Swagger: http://127.0.0.1:8000/docs
- Qdrant 는 gRPC(6334)로 접속합니다. REST 로 바꾸려면 `QDRANT_GRPC=0`
- Qdrant Cloud: `.env.example` 의 Qdrant 항목을 `.env.cloud` 에 채우고(git 제외) `--env-file` 로 넘깁니다. PG 는 로컬 그대로입니다

```bash
uv run --env-file .env.cloud python scripts/ingest.py --parsed data/parsed/kowiki-subset-10000
RAG_COLLECTION=kowiki-subset-10000_kure-v1 uv run --env-file .env.cloud uvicorn playground_wiki.api.app:create_app --factory --port 8000
```

## 구조

```
configs/                 인제스트 · 검색 · 답변 정책 · LLM 설정
src/playground_wiki/
  sources/               WikiSource (덤프 스트리밍 파싱)
  ingest/                WikiCleaner (위키 마크업 정제)
  generation/            AnswerPolicy (리랭커 점수로 grounded · partial · refuse, 단계별 생성 설정)
  api/                   FastAPI — /search · /health (검색 엔진은 playground_retrieval)
scripts/                 덤프 다운로드 · 파싱 · 인제스트 · BM25 색인 · 평가 · ANN 벤치
tests/
data/                    덤프 · 파싱 결과 · 임베딩 (git 제외)
```

## 주요 결과

한국어 위키백과 2026-10-01 덤프: 문서 488,963 · 청크 2,393,485 (KURE-v1 · 384토큰). 질문은 KorQuAD 1.0 dev 5,766문항.

| 검색기 | hit@5 (전체) | hit@5 (서브셋 1만) | 리포트 |
|---|---|---|---|
| Dense | 0.691 | 0.846 | |
| BM25 (Kiwi) | 0.723 | 0.859 | |
| Hybrid (RRF) | 0.767 | 0.882 | |
| **Hybrid + 리랭커** | **0.834** | **0.911** | [전체](reports/2026-10-23-retrieval-full.md) · [서브셋](reports/2026-10-22-retrieval-d12.md) |

- ANN: FAISS (Flat · HNSW · IVF · SQ8 · PQ) vs Qdrant (float · int8 · int8+rescore) recall–지연 — [reports/ann-benchmark.md](reports/ann-benchmark.md)
- 서브셋 수치는 검색기 사이 상대 비교용이다 (방해 문서가 적어 높게 나온다)

## 진행 상황

- [x] Phase 0 — 골격
- [x] Phase 0.5 — 모듈 분리
- [x] Phase 1 — 인제스트 + 검색 + 모델 선정
- [x] Phase 1.5 — 전체 인제스트
- [ ] Phase 2 — 단발 RAG 챗봇
- [ ] Phase 3 — 에이전트
- [ ] Phase 4 — 웹소설 프로젝트 연결 + 데모
