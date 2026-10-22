"""ANN 벤치마크: FAISS(Flat · HNSW · IVF · IVF+SQ8 · IVF+PQ) vs Qdrant(전수 · HNSW float · HNSW int8).

정답은 FAISS Flat 전수 검색의 상위 10건이다. 지표는 정답 대비 recall@10, KorQuAD hit@5 · MRR@10(evalkit 매칭),
질의 하나씩 보낸 지연 p50 · p99, 질의를 한꺼번에 보낸 처리량(QPS), 색인 크기, 색인 시간.

- 문서 벡터는 scripts/ingest.py 가 남긴 임베딩 캐시를, 청크 원문은 PG 를 읽는다 (같은 컬렉션)
- 질문은 정답 문서가 컬렉션에 있는 KorQuAD dev 질문 중 무작위 --queries 개. 질의 벡터는 캐시에 남긴다
- FAISS 지연은 스레드 1개, 처리량은 전체 스레드. Qdrant 지연은 REST 왕복 포함
- 서브셋 수치는 1차 확인용. 전체 규모는 Phase 1.5 (reports/ann-benchmark.md)

출력:
  reports/{run_id}.md   결과 표 (커밋)

사용법:
  uv run python scripts/bench_ann.py --collection kowiki-subset-10000_kure-v1
"""

import argparse
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray
from playground_corpus import (
    EMBEDDING_MODELS,
    EmbeddingCache,
    SentenceTransformerEmbedder,
)
from playground_evalkit import (
    KorQuADItem,
    contains_answer,
    hit_at_k,
    load_korquad,
    mrr,
    render_table,
)
from qdrant_client import models

from playground_rag.eval import latency_summary, recall_at_k
from playground_rag.retrieval import FaissIndex
from playground_rag.store import PgConfig, PgStore, QdrantStore

SUBSET = Path("data/parsed/kowiki-subset-10000")
SOURCE = "kowiki"
EMB_DIR = Path("data/emb")
TOP_K = 10
WARMUP = 20  # 측정 전에 버리는 질의 수 (캐시 · 스레드 풀 준비)
QDRANT_BATCH = 100  # 처리량 측정 때 요청 하나에 담는 질의 수
NLIST = 1024  # IVF 목록 수 ≈ 4·√n (n = 5만). 전체 규모에서는 다시 정한다
# (index_factory 문자열, 바꿔 볼 검색 인자, 값들)
FAISS_SPECS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("Flat", "", (0,)),
    ("HNSW32", "efSearch", (16, 32, 64, 128, 256)),
    (f"IVF{NLIST},Flat", "nprobe", (1, 4, 16, 64)),
    (f"IVF{NLIST},SQ8", "nprobe", (4, 16, 64)),
    (f"IVF{NLIST},PQ128", "nprobe", (4, 16, 64)),  # 벡터당 128바이트 (float32 의 1/32)
)
QDRANT_EF = (16, 32, 64, 128, 256)
COLUMNS = [
    "engine",
    "index",
    "param",
    "recall_at_10",
    "hit_at_5",
    "mrr_at_10",
    "p50_ms",
    "p99_ms",
    "qps",
    "build_sec",
    "index_mb",
]


@dataclass(frozen=True)
class RunResult:
    """색인 설정 하나로 질의 전부를 돌린 결과."""

    found: list[list[str]]
    """질의마다 청크 id 상위 10 (순위순)."""
    latencies: list[float]
    """질의 하나씩 보낸 검색 시간 (ms)."""
    qps: float
    """질의를 한꺼번에 보냈을 때 초당 질의 수."""


def main() -> None:
    args = _parse_args()
    run_id = args.run_id or f"{datetime.now(tz=UTC):%Y-%m-%d}-ann-subset"
    chunk_ids, vectors = EmbeddingCache(
        EMB_DIR / args.model / f"{args.collection}-{args.max_tokens}"
    ).load()
    with PgStore.connect(PgConfig.from_env()) as pg:
        chunks = {chunk.id: chunk for chunk in pg.all_chunks(args.collection, SOURCE)}
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunks]
    if missing:
        raise SystemExit(
            f"PG 에 없는 청크 {len(missing)}개 (예: {missing[0]}) — 같은 컬렉션인지 확인"
        )
    doc_ids = [chunks[chunk_id].doc_id for chunk_id in chunk_ids]
    items = _sample_questions(args, {chunk.title for chunk in chunks.values()})
    queries = _query_vectors(args, items)
    print(
        f"vectors={len(chunk_ids):,} queries={len(items):,} threads={faiss.omp_get_max_threads()}"
    )

    def judge(found: list[list[str]]) -> list[list[bool]]:
        return [
            [contains_answer(chunks[c].title, chunks[c].text, item) for c in ids]
            for item, ids in zip(items, found, strict=True)
        ]

    rows: list[dict[str, float | str | int]] = []
    truth: list[list[str]] = []
    for factory, param, values in FAISS_SPECS:
        start = time.perf_counter()
        index = FaissIndex.build(
            vectors, chunk_ids, doc_ids, collection=args.collection, factory=factory
        )
        build_sec = time.perf_counter() - start
        index_mb = faiss.serialize_index(index.index).nbytes / 2**20
        for value in values:
            if param:
                faiss.ParameterSpace().set_index_parameter(index.index, param, value)
            run = _faiss_run(index, queries)
            if factory == "Flat":
                truth = run.found
            label = f"{param}={value}" if param else "-"
            row = _row("faiss", factory, label, run, truth=truth, judge=judge)
            rows.append({**row, "build_sec": round(build_sec, 1), "index_mb": round(index_mb, 1)})
            _print(row)

    qdrant = QdrantStore.from_env()
    for label, param, params in _qdrant_specs():
        run = _qdrant_run(qdrant, args.collection, queries, params)
        row = _row("qdrant", label, param, run, truth=truth, judge=judge)
        rows.append({**row, "build_sec": "-", "index_mb": "-"})
        _print(row)

    _write_report(run_id, rows, len(chunk_ids), len(items), args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", required=True, help="scripts/ingest.py 로 적재한 컬렉션")
    parser.add_argument("--model", default="kure-v1", choices=list(EMBEDDING_MODELS))
    parser.add_argument(
        "--max-tokens", type=int, default=384, help="임베딩 캐시 폴더 이름의 청크 크기"
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=SUBSET,
        help="subset.json 이 있는 폴더 (KorQuAD 경로 · 제목 보정)",
    )
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def _sample_questions(args: argparse.Namespace, titles: set[str]) -> list[KorQuADItem]:
    summary = json.loads((args.subset / "subset.json").read_text(encoding="utf-8"))
    items = [
        item
        for item in load_korquad(
            Path(summary["korquad_dev"]), title_aliases=summary["korquad_title_map"]
        )
        if item.title in titles
    ]
    return random.Random(args.seed).sample(items, min(args.queries, len(items)))


def _query_vectors(args: argparse.Namespace, items: Sequence[KorQuADItem]) -> NDArray[np.float32]:
    """질의 벡터. 같은 질문 · 모델이면 캐시를 다시 쓴다."""
    cache = EmbeddingCache(EMB_DIR / args.model / f"korquad-dev-q{len(items)}-seed{args.seed}")
    ids = [item.id for item in items]
    if not (cache.directory / "manifest.json").exists():
        embedder = SentenceTransformerEmbedder.load(EMBEDDING_MODELS[args.model])
        cache.write(embedder, ids, [item.question for item in items], kind="query")
    cached_ids, vectors = cache.load()
    if cached_ids != ids:
        raise SystemExit(f"질의 캐시가 질문과 다르다: {cache.directory} 를 지우고 다시 실행")
    return vectors


def _faiss_run(index: FaissIndex, queries: NDArray[np.float32]) -> RunResult:
    normalized = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        index.index.search(normalized[:WARMUP], TOP_K)
        latencies = _timed(lambda q: index.index.search(q.reshape(1, -1), TOP_K), normalized)
    finally:
        faiss.omp_set_num_threads(threads)
    start = time.perf_counter()
    _, rows = index.index.search(normalized, TOP_K)
    qps = len(normalized) / (time.perf_counter() - start)
    found = [[index.chunk_ids[row] for row in query_rows if row >= 0] for query_rows in rows]
    return RunResult(found=found, latencies=latencies, qps=qps)


def _qdrant_specs() -> list[tuple[str, str, models.SearchParams]]:
    """(색인 이름, 인자 표기, 검색 인자). 컬렉션은 HNSW(m=16) + int8 스칼라 양자화, 원본 벡터 on_disk."""
    original = models.QuantizationSearchParams(ignore=True)
    variants = (
        ("HNSW float", original),  # 양자화를 쓰지 않고 원본 벡터로
        ("HNSW int8", models.QuantizationSearchParams(ignore=False, rescore=False)),
        # 재채점은 후보 안의 순서만 바꾸므로 후보를 2배로 뽑아야 recall 이 오른다
        (
            "HNSW int8+rescore x2",
            models.QuantizationSearchParams(ignore=False, rescore=True, oversampling=2.0),
        ),
    )
    specs = [("exact float", "-", models.SearchParams(exact=True, quantization=original))]
    for label, quantization in variants:
        specs.extend(
            (label, f"hnsw_ef={ef}", models.SearchParams(hnsw_ef=ef, quantization=quantization))
            for ef in QDRANT_EF
        )
    return specs


def _qdrant_run(
    store: QdrantStore, collection: str, queries: NDArray[np.float32], params: models.SearchParams
) -> RunResult:
    for query in queries[:WARMUP]:
        store.search(collection, query, TOP_K, params=params)
    latencies = _timed(lambda q: store.search(collection, q, TOP_K, params=params), queries)
    start = time.perf_counter()
    found: list[list[str]] = []
    for offset in range(0, len(queries), QDRANT_BATCH):
        batch = store.search_batch(
            collection, queries[offset : offset + QDRANT_BATCH], TOP_K, params=params
        )
        found.extend([hit.chunk_id for hit in hits] for hits in batch)
    qps = len(queries) / (time.perf_counter() - start)
    return RunResult(found=found, latencies=latencies, qps=qps)


def _timed(
    search: Callable[[NDArray[np.float32]], object], queries: NDArray[np.float32]
) -> list[float]:
    latencies = []
    for query in queries:
        start = time.perf_counter()
        search(query)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def _row(
    engine: str,
    index: str,
    param: str,
    run: RunResult,
    *,
    truth: list[list[str]],
    judge: Callable[[list[list[str]]], list[list[bool]]],
) -> dict[str, float | str | int]:
    relevance = judge(run.found)
    latency = latency_summary(run.latencies)
    return {
        "engine": engine,
        "index": index,
        "param": param,
        "recall_at_10": recall_at_k(run.found, truth, TOP_K),
        "hit_at_5": hit_at_k(relevance, 5),
        "mrr_at_10": mrr(relevance),
        "p50_ms": latency.p50_ms,
        "p99_ms": latency.p99_ms,
        "qps": round(run.qps),
    }


def _print(row: dict[str, float | str | int]) -> None:
    print(
        f"{row['engine']} {row['index']} {row['param']}: recall@10={row['recall_at_10']:.3f} "
        f"hit@5={row['hit_at_5']:.3f} p50={row['p50_ms']}ms qps={row['qps']}"
    )


def _write_report(
    run_id: str,
    rows: list[dict[str, float | str | int]],
    vectors: int,
    questions: int,
    args: argparse.Namespace,
) -> None:
    report = (
        f"# ANN 벤치마크 — {run_id}\n\n"
        f"- 컬렉션: `{args.collection}` 벡터 {vectors:,}개 ({args.model}, 청크 {args.max_tokens}토큰) · "
        f"KorQuAD dev 질문 {questions:,}개 (정답 문서가 컬렉션에 있는 질문 중 무작위, seed {args.seed})\n"
        "- `recall_at_10`: FAISS Flat(전수 비교) 상위 10건 대비. `hit_at_5` · `mrr_at_10`: evalkit `contains_answer` 정답 판정\n"
        f"- FAISS: CPU, 지연은 스레드 1개 · 질의 하나씩, `qps` 는 질의 {questions:,}개 일괄 · 스레드 {faiss.omp_get_max_threads()}개. "
        "`index_mb` 는 직렬화한 색인 크기\n"
        f"- Qdrant: 로컬 Docker, HNSW m=16 · ef_construct=100 + int8 스칼라 양자화(RAM) · 원본 벡터 on_disk. "
        f"지연은 gRPC 왕복 포함 (REST 는 질의마다 약 16ms 더 든다), `qps` 는 요청 하나에 {QDRANT_BATCH}개씩 묶어 보냄. "
        "색인 시간 · 크기는 적재 때라 비움\n"
        "- **서브셋 1차 확인용** — 전체 규모(약 230만 벡터)는 Phase 1.5\n\n"
        + render_table(rows, COLUMNS)
    )
    Path("reports").mkdir(exist_ok=True)
    # Windows 에서도 LF 로 (리포트는 커밋하고 .gitattributes 가 LF)
    (Path("reports") / f"{run_id}.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"report: reports/{run_id}.md")


if __name__ == "__main__":
    main()
