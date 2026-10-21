# llm-playground-rag — 프로젝트 계획

한국어 위키백과 덤프로 RAG 파이프라인을 처음부터 끝까지 구축하고, 그 위에 **위키 검색 기반 한국어 QA 챗봇**을 올린다.
파이프라인은 코퍼스 독립적으로 설계해 1호 소비자인 웹소설 집필 에이전트(고증 검색 + 설정집 검색)와 사내 문서 QA에 그대로 재사용한다.

---

## 1. 목표

| 구분 | 내용 | 측정 |
|---|---|---|
| 파이프라인 | 덤프 → 정제 → 청킹 → 임베딩 → 저장 → 검색을 코퍼스 독립 구조로 구현 | `Source` 추가만으로 새 코퍼스 인제스트 가능 |
| 검색 품질 | 하이브리드(dense + BM25) + 리랭킹 | KorQuAD hit@5 · MRR |
| 답변 품질 | 근거 기반 답변 + 출처, 모를 때는 모른다고 답함 | KorQuAD EM/F1, 거부 정확도 |
| ANN 이해 | FAISS·Qdrant 인덱스별 recall–지연–메모리 트레이드오프 측정 | recall@10 vs QPS 곡선 |
| 에이전트 | 다단계 검색으로 단일 검색으로는 안 풀리는 질문 처리 | 자체 "복합 질문" 세트에서 단발 RAG 대비 개선 |
| 재사용 | 웹소설 프로젝트에 `search_wiki` 툴 + 설정집 인제스트로 연결 | 연결 PR 1건 |

## 2. 범위

**포함**
- 한국어 위키백과 공식 덤프 파싱·정제 (직접 구현)
- 섹션 인지 청킹, 임베딩 모델 비교·선정
- PG(원문·메타) + Qdrant(벡터) 저장, FAISS 벤치마크
- 하이브리드 검색, 리랭커, 3단계 답변 정책
- FastAPI `/search` `/ask`, LangGraph 에이전트
- KorQuAD 기반 평가 하니스, ANN 벤치마크 리포트
- 웹소설 프로젝트 연결 (`search_wiki` 툴, `NovelSource`)

**제외**
- 임베딩·리랭커 모델 파인튜닝
- 덤프 자동 갱신(증분 인제스트) — 수동 재실행으로 충분
- 인증·멀티테넌시·과금
- 정식 프론트엔드 (Gradio 데모 1개까지만)

## 3. 확정된 결정

| 항목 | 결정 | 비고 |
|---|---|---|
| 데이터 | 공식 덤프 `kowiki-20261001-pages-articles.xml.bz2` (2026-10-01 생성, 1.4GB, 다운로드 완료 → `data/dump/`) | 정제본(HF) 대신 직접 파싱 — 전처리 경험이 목적. `dump_date = 20261001` 고정 |
| 임베딩 | bge-m3 · KURE-v1 · KoE5 **비교 후 선정** | Phase 1에서 서브셋 + FAISS Flat + KorQuAD hit@5로 결정 |
| 벡터 저장 | 로컬 Docker Qdrant(전체) + Qdrant Cloud 무료(서브셋 데모) | Cloud는 GCP Iowa 클러스터 (생성 완료) |
| 원문·메타 | PostgreSQL | pgvector는 쓰지 않음 (검색은 Qdrant) |
| FAISS | 평가·벤치마크 전용 (서빙 아님) | `Retriever` 인터페이스로 Qdrant와 교체 가능 |
| 키워드 검색 | BM25 + Kiwi 형태소 분석 | 한국어 조사 처리 때문에 필수 |
| 리랭커 | bge-reranker-v2-m3 | 임계값은 캘리브레이션으로 결정 |
| LLM | Ollama + gemma4 | OpenAI 호환 클라이언트로 감싸 vLLM 전환은 설정만 변경 |
| 에이전트 | LangGraph | Phase 3 |
| API | FastAPI, `/ask`는 SSE 스트리밍 | |
| UI | Phase 1~3 없음(Swagger), Phase 4 Gradio 데모 1개 | |
| 평가 데이터 | KorQuAD 1.0 (위키 기반 QA 6만 쌍) | 검색·생성 정답지로 사용 |
| 패키지 관리 | uv (`pyproject.toml` + `uv.lock`, `.venv` 프로젝트 내부), Python 3.13 고정(`.python-version`) | `uv run`으로 스크립트 실행, lock 파일 커밋 |
| 브랜치 | 로컬은 `main` 하나 | 원격은 공개 저장소 `calm-sea-psy/llm-playground-rag` |
| 코드 규칙 | 전역 스킬 `python-conventions` + ruff(base + fastapi + numpy) | frozen dataclass, pydantic은 API 경계에서만 |
| PR | 전역 스킬 `pr-create` + `pr-lint` 워크플로 | 검수 후 생성 |

## 4. 아키텍처

```
[인제스트 — 배치]
 Source ──→ Document ──→ Chunker ──→ Embedder ──→ Store
  │                                                │
  ├ WikiSource (XML 스트리밍 파싱·정제)              ├ PG: documents, chunks
  ├ NovelSource (설정집 md·회차 txt)   ← Phase 4    └ Qdrant: collection별 벡터 + payload
  └ (PdfSource 등 추후)

[서빙 — 온라인]
 /search ─→ HybridRetriever ─→ Reranker ─→ top-k 청크 + 점수
              ├ DenseRetriever (Qdrant)
              └ Bm25Retriever (Kiwi 토큰, 인메모리 또는 PG)
 /ask    ─→ 질문 ─→ /search ─→ 답변 정책(3단계) ─→ LLM(gemma4) ─→ SSE 스트림 + 출처
 /ask?mode=agent ─→ LangGraph: 분해 → 검색 → 충분? → 재검색(≤3) → 답변

[평가 — 오프라인]
 FAISS Flat(정답) · HNSW · IVF · IVF+PQ  vs  Qdrant HNSW · 양자화 · on_disk
 KorQuAD → hit@k, MRR, EM/F1, 거부 정확도
```

### 4.1 핵심 인터페이스

```python
@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    sections: tuple[Section, ...]
    meta: Mapping[str, str]        # source, url, categories, dump_date ...

@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str                      # "제목 > 섹션\n본문" 형태로 임베딩
    seq: int

class Source(Protocol):
    def documents(self) -> Iterator[Document]: ...

class Retriever(Protocol):
    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]: ...

class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...
```

규칙: `Source`는 `Document`를 내놓는 것까지만 책임진다. 파싱 코드는 청킹 이후를 직접 호출하지 않는다.

## 5. 데이터 파이프라인

### 5.1 파싱 (`WikiSource`)
- `bz2.open` + `lxml.iterparse`로 스트리밍. 전체를 메모리에 올리지 않음
- namespace 0만. `#넘겨주기`(리다이렉트), 동음이의 문서, 본문 200자 미만 stub 제외
- `multiprocessing`으로 페이지 단위 병렬 처리 → parquet 샤드로 저장 (`data/parsed/*.parquet`)
- 덤프 날짜를 `meta.dump_date`에 기록. 답변에 "이 날짜 이후 정보 없음"을 표시하는 데 사용

### 5.2 정제
- `mwparserfromhell.strip_code()` 후 정규식 후처리
  - 틀 잔여물 `{{...}}`, 파일·분류 링크, 각주 `[1]`, 표 마크업, HTML 태그·엔티티
  - 연속 공백·빈 줄 정리
- 섹션 헤더(`== 제목 ==`)는 제거하지 않고 `Section(level, title, text)`로 구조화
- 분류(`[[분류:...]]`)는 본문에서 빼서 `meta.categories`로
- 정제 품질 점검: 랜덤 50개 문서를 눈으로 확인하는 체크리스트를 `docs/cleaning-checklist.md`에 기록

### 5.3 청킹
- 섹션 단위로 먼저 분할 → 섹션이 길면 선정 모델의 토크나이저 기준 256~512 토큰, 오버랩 10~20%
- 각 청크 앞에 `"{문서 제목} > {섹션 제목}"` 헤더 삽입
- 50자 미만 청크 제거
- 청크 크기(256/384/512)는 Phase 1 평가 항목

### 5.4 임베딩
- 후보: bge-m3(1024d) · KURE-v1(1024d) · KoE5(1024d)
- fp16, 길이순 정렬 배치, 결과는 `data/emb/{model}/*.npy` 샤드 + id 매핑 — **재계산 금지**
- 선정 기준: 서브셋(문서 1만) KorQuAD hit@5 1순위, 처리량 2순위
- 전체 임베딩은 선정 모델 1개만. 예상 2~4시간 (RTX 5080 Laptop 16GB)

### 5.5 저장
- PG
  - `documents(id, title, url, categories, dump_date, source)`
  - `chunks(id, doc_id, seq, section, text, token_count)`
  - `ingest_runs(id, source, model, started_at, finished_at, chunk_count)`
- Qdrant
  - 컬렉션명 `{source}_{model}` (예: `kowiki_bge-m3`)
  - payload: `doc_id, title, section, seq, categories, dump_date`
  - 로컬: int8 스칼라 양자화 + `on_disk: true`. 전체 300~400만 청크를 RAM 3~4GB 선에서 유지
  - Cloud 무료(1GB): 서브셋 컬렉션만 (범위는 미결 사항 참고)

## 6. 검색·생성

### 6.1 하이브리드 검색
1. Dense: Qdrant top-50
2. BM25: Kiwi로 명사·어간 토큰화 → `rank_bm25` 인메모리(서브셋) / PG 테이블(전체) top-50
3. RRF(k=60)로 합산 → top-20
4. 리랭커(bge-reranker-v2-m3) → top-5 + 점수

### 6.2 답변 정책 (3단계)

| 상황 | 판단 | 응답 |
|---|---|---|
| 관련 문서 있음 | 리랭커 최고 점수 ≥ T_high | 근거 기반 답변 + 출처 링크 |
| 약하게 관련 | T_low ≤ 점수 < T_high | "위키에 직접적인 내용은 없음. 관련 문서: A, B" + 일반 지식 답변을 **명시적으로 분리** |
| 관련 없음 | 점수 < T_low | 모른다고 답하고 시도한 검색어 표시 |

T_high, T_low는 KorQuAD 질문(답 있음) + 위키 밖 질문 100개(최신 뉴스·개인사·계산)로 거부 정확도가 최대가 되는 값으로 캘리브레이션.

### 6.3 생성
- 프롬프트: 컨텍스트 안에서만 답변, 출처는 `[문서 제목]`으로 표기, 덤프 날짜 이후 정보 없음 고지
- 출처 URL: `https://ko.wikipedia.org/wiki/{title}` 로 직접 생성
- SSE 스트리밍, 답변 후 `sources` 이벤트로 청크 목록 전송

### 6.4 에이전트 (Phase 3)
```
decompose(질문 → 하위 질문 목록)
  → for each: search → collect
  → judge(수집된 근거로 충분한가?)
      yes → answer(출처 포함)
      no  → refine(새 검색어) → search (최대 3회)
```
- 상태: `question, sub_questions, evidence[], iterations, answer`
- 단발 RAG와 동일한 `/search`를 툴로 사용. 검색 로직 중복 없음

## 7. 평가

| 대상 | 데이터 | 지표 | 비교 축 |
|---|---|---|---|
| 검색 | KorQuAD 1.0 dev (질문 → 정답 문단) | hit@1/5/10, MRR | 임베딩 모델, 청크 크기, BM25 유무, 리랭커 유무 |
| 생성 | KorQuAD 1.0 dev | EM, F1 | 단발 RAG vs 에이전트 |
| 거부 | KorQuAD 100 + 위키 밖 질문 100 | 거부 정확도(precision/recall) | 임계값 |
| ANN | 선정 임베딩 전체, 질의 1,000개 | recall@10, p50/p99 지연, QPS, 메모리, 빌드 시간 | FAISS Flat/HNSW/IVF/IVF+PQ, Qdrant HNSW/양자화/on_disk, 파라미터 스윕(ef, nprobe, m) |
| 복합 질문 | 직접 작성 50개 (다단계 필요) | 정답률(수동 채점), 평균 검색 횟수 | 단발 vs 에이전트 |

- KorQuAD 정답 문단 ↔ 우리 청크 매칭은 **문서 제목 + 정답 문자열 포함 여부**로 판정
- 모든 평가는 `scripts/eval_*.py`로 재현 가능하게, 결과는 `reports/`에 날짜별 저장
- ANN 결과는 recall–QPS 곡선 그래프로 `reports/ann-benchmark.md`에

## 8. 레포 구조

```
llm-playground-rag/
├── plan.md
├── README.md
├── .python-version                # 3.13
├── .gitignore                     # data/, docs/
├── pyproject.toml                 # uv 관리
├── uv.lock                        # 커밋
├── ruff.toml                      # python-conventions 스킬의 base + fastapi + numpy
├── docker-compose.yml             # postgres, qdrant
├── configs/
│   ├── ingest.yaml                # 청킹·임베딩·컬렉션 설정
│   ├── retrieval.yaml             # top-k, RRF, 리랭커 임계값
│   └── llm.yaml                   # Ollama/vLLM 엔드포인트, 모델명
├── src/llm_playground_rag/
│   ├── core/                      # Document, Chunk, Hit, Protocol 정의
│   ├── sources/                   # wiki.py, novel.py (Phase 4)
│   ├── ingest/                    # cleaner.py, chunker.py, embedder.py, pipeline.py
│   ├── store/                     # pg.py, qdrant.py
│   ├── retrieval/                 # dense.py, bm25.py, hybrid.py, rerank.py, faiss_retriever.py
│   ├── generation/                # llm_client.py, prompts.py, policy.py
│   ├── agent/                     # graph.py, nodes.py (Phase 3)
│   ├── eval/                      # korquad.py, metrics.py, ann_bench.py
│   └── api/                       # app.py, routes/search.py, routes/ask.py, schemas.py (pydantic)
├── scripts/
│   ├── download_dump.sh
│   ├── ingest.py                  # --source wiki --subset 10000 / --full
│   ├── embed_compare.py           # 임베딩 모델 비교
│   ├── eval_retrieval.py
│   ├── eval_answer.py
│   ├── calibrate_threshold.py
│   └── bench_ann.py
├── tests/
├── data/                          # .gitignore — dump/, parsed/, emb/
├── reports/                       # 평가·벤치마크 결과 (커밋)
└── docs/                          # .gitignore — 로컬 전용
    ├── cleaning-checklist.md
    ├── decisions.md               # ADR 형식 결정 기록
    └── integration-novel.md       # Phase 4
```

## 9. 단계별 계획

기간은 정하지 않는다. 각 단계는 완료 기준을 만족하면 다음으로 넘어간다.

### Phase 0 — 골격
- 레포 생성, `uv init` + pyproject, ruff, docker-compose(PG + Qdrant), `core/` 인터페이스
- `scripts/download_dump.sh` — 덤프 다운로드 + 체크섬
- **완료 기준**: `docker compose up` 후 PG·Qdrant 헬스체크 통과, `uv run pytest` 빈 테스트 통과

### Phase 1 — 인제스트 + 검색 + 모델 선정
- `WikiSource` 파싱·정제, 서브셋(문서 1만) parquet 생성
- 청킹, 임베딩 3종 비교(`embed_compare.py`), 선정 → `docs/decisions.md`
- PG·Qdrant 저장, `DenseRetriever` · `Bm25Retriever` · `HybridRetriever` · `Reranker`
- `FaissRetriever` + `bench_ann.py` 1차(서브셋)
- `/search` API
- `eval_retrieval.py` — KorQuAD hit@k, MRR (서브셋에 해당하는 질문만)
- **완료 기준**: 서브셋에서 하이브리드+리랭커 hit@5 ≥ dense 단독 대비 개선 확인, 임베딩 모델 결정 기록

### Phase 1.5 — 전체 인제스트
- 선정 모델로 전체 임베딩 (하룻밤), 로컬 Qdrant 전체 적재 (양자화 + on_disk)
- Cloud 무료 클러스터에 서브셋 적재
- `bench_ann.py` 전체 규모 실행 → `reports/ann-benchmark.md`
- **완료 기준**: 전체 KorQuAD dev hit@5·MRR 기록, ANN 리포트 작성

### Phase 2 — 단발 RAG 챗봇
- `llm_client.py`(OpenAI 호환), 프롬프트, 3단계 답변 정책
- `calibrate_threshold.py` — 위키 밖 질문 100개 작성 + 임계값 결정
- `/ask` SSE
- `eval_answer.py` — EM/F1, 거부 정확도
- **완료 기준**: `/ask`가 출처 포함 답변·분리 답변·거부 세 경우를 모두 올바르게 냄, 지표 기록

### Phase 3 — 에이전트
- LangGraph 그래프(분해 → 검색 → 판단 → 재검색 → 답변), `/ask?mode=agent`
- 복합 질문 50개 작성, 단발 vs 에이전트 비교
- **완료 기준**: 복합 질문 세트에서 에이전트가 단발 대비 개선, 평균 검색 횟수·지연 기록

### Phase 4 — 연결 + 데모
- 웹소설 프로젝트: `search_wiki` 툴 등록 (`/search` 호출), `NovelSource`(설정집 md·회차 txt) → `novel_bible` 컬렉션
- Gradio 채팅 데모 (출처 표시)
- `docs/integration-novel.md`
- (선택) 질의 라우터 분류기 — LLM 판단을 소형 분류기로 증류해 지연 비교
- **완료 기준**: 웹소설 프로젝트에서 위키 검색·설정집 검색이 같은 파이프라인으로 동작

## 10. 리소스 추정

| 항목 | 추정 |
|---|---|
| 덤프 | 압축 1.4GB (20261001), 풀면 6~7GB, 본문 문서 65~70만 |
| 청크 수 | 300~400만 (256~512 토큰 기준) |
| 벡터 원본 | 1024d float32 × 350만 ≈ 14GB → npy 샤드로 디스크 보관 |
| Qdrant 로컬 RAM | int8 양자화 + on_disk → 3~4GB |
| 임베딩 시간 | 모델 비교(서브셋 3종) 수십 분, 전체(1종) 2~4시간 |
| 파싱 시간 | 멀티프로세스 10~20분 |
| LLM | gemma4 (Ollama), 임베딩·리랭커와 VRAM 동시 사용 시 모델 크기 조정 |

## 11. 리스크

| 리스크 | 대응 |
|---|---|
| 정제 품질 — 틀 잔여물·표 깨짐이 검색 품질을 깎음 | 체크리스트로 샘플 검수, 정제 규칙은 테스트로 고정 |
| VRAM 16GB에 임베딩 + 리랭커 + LLM 동시 적재 불가 | 인제스트(임베딩)와 서빙(리랭커+LLM)을 시간 분리, 서빙 시 리랭커는 fp16·배치 1 |
| KorQuAD 문단 ↔ 청크 매칭 오류로 지표 왜곡 | 매칭 규칙을 문서화하고 샘플 100개 수동 확인 |
| 소형 LLM이 무관한 컨텍스트에도 그럴듯하게 답함 | 3단계 정책의 임계값 + "컨텍스트 밖이면 모른다" 프롬프트 + 거부 정확도 지표로 감시 |
| 전체 인제스트 중 중단 | parquet·npy 샤드 단위 재개, `ingest_runs` 테이블로 진행 추적 |
| 범위 확장 (UI·파인튜닝·증분 갱신) | §2 비범위 고정. 새 항목은 `docs/decisions.md`에 기록 후에만 |

## 12. 미결 사항

Phase 0 시작에 필요한 항목은 모두 결정됨. 아래는 해당 단계에서 결정한다.

- Qdrant Cloud 서브셋 범위 — 특정 분류(역사·과학 등) vs 랜덤 1만 문서
- BM25 전체 규모 저장 방식 — PG 테이블 vs 별도 인메모리 서비스
- 복합 질문 50개·위키 밖 질문 100개 작성 주체와 형식

## 13. 웹소설 프로젝트 연결 설계 (Phase 4 상세)

```
웹소설 에이전트
 ├─ 리서처 노드 ── search_wiki(query) ──→ llm-playground-rag /search?collection=kowiki
 │                                          (고증: 역사·지리·문화·용어)
 └─ 설정 검수 노드 ── search_bible(query) ─→ llm-playground-rag /search?collection=novel_bible
                                            (설정 일관성: 인물·사건·세계관)

인제스트: 회차 완성 시 NovelSource(회차 txt) → novel_bible 컬렉션에 추가
```

- `/search`에 `collection` 파라미터 추가 — 같은 API로 코퍼스만 선택
- `NovelSource`: 설정집 md는 헤더 단위 섹션, 회차 txt는 장면 단위 섹션으로 `Document` 생성
- 설정집 컬렉션은 수백~수천 청크 규모라 BM25 인메모리 + Qdrant Cloud 무료로 충분
