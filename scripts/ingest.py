"""파싱한 문서를 청킹 · 임베딩해 PostgreSQL(원문) · Qdrant(벡터)에 넣는다.

순서: 문서 → PG documents → 청킹 → PG chunks → 임베딩(npy 캐시, 다시 계산 안 함) → Qdrant → 실행 기록.
같은 키로 다시 넣으면 덮어쓰므로 중단된 뒤 다시 돌려도 된다.

사용법:
  uv run python scripts/ingest.py --parsed data/parsed/kowiki-subset-10000          # 선정 모델 (configs)
  uv run python scripts/ingest.py --parsed data/parsed/kowiki-subset-10000 --model bge-m3 --limit-docs 300 --collection smoke
"""

import argparse
import time
from pathlib import Path

import yaml
from playground_corpus import (
    EMBEDDING_MODELS,
    Chunker,
    ChunkingConfig,
    EmbeddingCache,
    HFTokenizer,
    SentenceTransformerEmbedder,
    read_documents,
)

from playground_rag.store import PgConfig, PgStore, QdrantStore


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = config["source"]["name"]
    chunking = config["chunking"]
    spec = EMBEDDING_MODELS[args.model or config["embedding"]["model"]]
    collection = args.collection or config["store"]["collection"].format(
        source=args.parsed.name, model=spec.name
    )

    documents = [
        doc for shard in sorted(args.parsed.glob("docs-*.parquet")) for doc in read_documents(shard)
    ]
    if args.limit_docs:
        documents = documents[: args.limit_docs]
    tokenizer = HFTokenizer.from_pretrained(spec.model_id)
    max_tokens = min(
        args.max_tokens or chunking["max_tokens"],
        spec.max_chunk_tokens(tokenizer, tokenizer.special_token_count),
    )
    chunker = Chunker(
        tokenizer,
        ChunkingConfig(
            max_tokens=max_tokens,
            overlap_ratio=chunking["overlap_ratio"],
            min_chars=chunking["min_chars"],
        ),
    )
    print(
        f"collection={collection} documents={len(documents):,} model={spec.name} max_tokens={max_tokens}"
    )

    with PgStore.connect(PgConfig.from_env()) as pg:
        pg.create_schema()
        run_id = pg.start_run(source, collection, spec.name, max_tokens)
        pg.upsert_documents(source, documents)
        start = time.perf_counter()
        chunks = [chunk for doc in documents for chunk in chunker.chunk(doc)]
        print(f"chunks={len(chunks):,} ({time.perf_counter() - start:.0f}s)")
        pg.upsert_chunks(collection, chunks)

        cache = EmbeddingCache(
            Path(config["embedding"]["output_dir"]) / spec.name / f"{collection}-{max_tokens}"
        )
        start = time.perf_counter()
        embedder = SentenceTransformerEmbedder.load(
            spec, fp16=config["embedding"]["fp16"], batch_size=config["embedding"]["batch_size"]
        )
        cache.write(embedder, [chunk.id for chunk in chunks], [chunk.text for chunk in chunks])
        ids, vectors = cache.load()
        print(f"embedded {len(ids):,} ({time.perf_counter() - start:.0f}s)")

        qdrant = QdrantStore.from_env()
        qdrant.ensure_collection(
            collection, dim=spec.dim, on_disk=config["store"]["qdrant"]["on_disk"]
        )
        categories = {doc.id: doc.categories for doc in documents}
        dump_date = config["source"]["dump_date"]
        qdrant.upsert(collection, chunks, vectors, categories, dump_date=str(dump_date))
        pg.finish_run(run_id, doc_count=len(documents), chunk_count=len(chunks))
        print(f"done: run_id={run_id} qdrant points={qdrant.count(collection):,}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/ingest.yaml"))
    parser.add_argument("--parsed", type=Path, required=True, help="docs-*.parquet 가 있는 폴더")
    parser.add_argument(
        "--model",
        default="",
        choices=["", *EMBEDDING_MODELS],
        help="비우면 configs/ingest.yaml 의 embedding.model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="0 이면 configs/ingest.yaml 의 chunking.max_tokens",
    )
    parser.add_argument(
        "--limit-docs", type=int, default=0, help="앞에서부터 이만큼 문서만 (0 = 전체, 시험용)"
    )
    parser.add_argument(
        "--collection", default="", help="기본: store.collection 템플릿 ({source}_{model})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
