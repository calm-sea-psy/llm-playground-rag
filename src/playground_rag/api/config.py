"""API 설정: configs/*.yaml 과 환경변수를 읽어 서비스 구성으로 바꾼다."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml

from playground_rag.retrieval import SearchSettings
from playground_rag.store import PgConfig

CONFIG_DIR = Path("configs")  # 레포 루트에서 실행할 때. RAG_CONFIG_DIR 로 바꾼다
BM25_INDEX_DIR = Path("data/index/bm25")  # scripts/build_bm25.py 가 {컬렉션} 폴더로 저장한다


@dataclass(frozen=True)
class ApiConfig:
    """API 프로세스 하나의 구성."""

    search: SearchSettings
    pg: PgConfig

    @classmethod
    def from_env(cls) -> Self:
        """RAG_CONFIG_DIR(기본 configs/) 의 ingest.yaml · retrieval.yaml 과 환경변수로 만든다.

        컬렉션은 RAG_COLLECTION, 없으면 ingest.yaml 의 store.collection 템플릿(`{source}_{model}`).
        서브셋을 띄울 때는 RAG_COLLECTION=kowiki-subset-10000_kure-v1 처럼 지정한다.
        BM25 색인은 RAG_BM25_INDEX_DIR(기본 data/index/bm25) 아래 컬렉션 이름 폴더.
        """
        config_dir = Path(os.environ.get("RAG_CONFIG_DIR", CONFIG_DIR))
        ingest = yaml.safe_load((config_dir / "ingest.yaml").read_text(encoding="utf-8"))
        retrieval = yaml.safe_load((config_dir / "retrieval.yaml").read_text(encoding="utf-8"))
        source = ingest["source"]["name"]
        model = ingest["embedding"]["model"]
        collection = os.environ.get("RAG_COLLECTION") or ingest["store"]["collection"].format(
            source=source, model=model
        )
        # HybridRetriever 는 검색기마다 같은 후보 수를 쓴다
        if retrieval["dense"]["top_k"] != retrieval["bm25"]["top_k"]:
            raise ValueError("retrieval.yaml: dense.top_k and bm25.top_k must be equal")
        search = SearchSettings(
            collection=collection,
            source=source,
            model=model,
            candidates=retrieval["dense"]["top_k"],
            rrf_k=retrieval["fusion"]["rrf_k"],
            rerank_candidates=retrieval["fusion"]["top_k"],
            reranker_model=retrieval["reranker"]["model"],
            top_k=retrieval["reranker"]["top_k"],
            bm25_index=Path(os.environ.get("RAG_BM25_INDEX_DIR", BM25_INDEX_DIR)) / collection,
        )
        return cls(search=search, pg=PgConfig.from_env())
