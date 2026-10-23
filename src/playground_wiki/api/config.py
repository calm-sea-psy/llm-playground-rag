"""API 설정: configs/*.yaml 과 환경변수를 읽어 서비스 구성으로 바꾼다."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import yaml
from playground_llm import ModelSpec
from playground_retrieval.search import SearchSettings
from playground_retrieval.store import PgConfig

from playground_wiki.generation import AnswerPolicy, Profile, Thresholds

CONFIG_DIR = Path("configs")  # 레포 루트에서 실행할 때. RAG_CONFIG_DIR 로 바꾼다
BM25_INDEX_DIR = Path("data/index/bm25")  # scripts/build_bm25.py 가 {컬렉션} 폴더로 저장한다
ANSWER_CONFIG = "answer.yaml"
# answer.yaml 프로필 한 행의 키. 오타 난 키가 조용히 무시되지 않게 정확히 이 집합이어야 한다
PROFILE_KEYS = frozenset({"provider", "model", "params", "prompts", "timeout_sec"})


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


def load_answer_policy(config_dir: Path) -> AnswerPolicy:
    """config_dir 의 answer.yaml(임계값 · 단계별 프로필)로 답변 정책을 만든다.

    Raises:
      ValueError: 프로필 값이 틀렸거나, 임계값을 아직 정하지 않았을 때(null).
    """
    raw = yaml.safe_load((config_dir / ANSWER_CONFIG).read_text(encoding="utf-8"))
    profiles = parse_profiles(raw["profiles"])
    high, low = raw["thresholds"]["high"], raw["thresholds"]["low"]
    if high is None or low is None:
        raise ValueError(
            f"{ANSWER_CONFIG}: thresholds are not calibrated yet — run scripts/calibrate_threshold.py"
        )
    return AnswerPolicy(Thresholds(high=float(high), low=float(low)), profiles)


def parse_profiles(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, Profile]:
    """answer.yaml 의 profiles 표를 단계별 Profile 로 바꾼다. YAML 목록은 튜플로."""
    return {stage: _profile(stage, row) for stage, row in raw.items()}


def _profile(stage: str, row: Mapping[str, Any]) -> Profile:
    if set(row) != PROFILE_KEYS:
        raise ValueError(
            f"{ANSWER_CONFIG}: profile {stage!r} keys must be {sorted(PROFILE_KEYS)}, "
            f"got {sorted(row)}"
        )
    params = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in row["params"].items()
    }
    return Profile(
        spec=ModelSpec(row["provider"], row["model"], params),
        prompts=tuple(row["prompts"]),
        timeout_sec=float(row["timeout_sec"]),
    )
