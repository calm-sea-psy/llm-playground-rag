"""3단계 답변 정책: 리랭커 1위 점수로 grounded · partial · refuse 를 고르고 단계별 생성 설정을 붙인다.

의도 분류기를 두지 않고 이 정책이 라우터 역할을 한다 (ADR-021). refuse 는 LLM 을 부르지 않는다.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, get_args

from playground_llm import ModelSpec
from playground_retrieval.core import Hit

Stage = Literal["grounded", "partial", "refuse"]
STAGES: tuple[str, ...] = get_args(Stage)
LLM_STAGES: frozenset[str] = frozenset(STAGES) - {"refuse"}
"""프로필(생성 설정)이 있어야 하는 단계."""


@dataclass(frozen=True)
class Thresholds:
    """리랭커 점수(0~1) 경계. scripts/calibrate_threshold.py 로 정한다."""

    high: float
    """이 이상이면 grounded."""
    low: float
    """이 이상이면 partial, 미만이면 refuse."""

    def __post_init__(self) -> None:
        if not 0 <= self.low <= self.high <= 1:
            raise ValueError(
                f"thresholds must satisfy 0 <= low <= high <= 1, got {self.low=} {self.high=}"
            )

    def stage(self, score: float | None) -> Stage:
        """리랭커 1위 점수의 단계. 검색 결과가 없으면(None) refuse."""
        if score is None or score < self.low:
            return "refuse"
        return "grounded" if score >= self.high else "partial"


@dataclass(frozen=True)
class Profile:
    """단계 하나의 생성 설정. (spec_id, 프롬프트 버전) 이 벤치 · 평가 기록의 키다."""

    spec: ModelSpec
    """모델과 생성 설정 (think · max_tokens · temperature …). think 를 바꾸면 spec_id 가 달라진다."""
    prompts: tuple[str, ...]
    """이어 붙일 프롬프트 `이름@버전`. 공통 베이스가 먼저, 단계 블록이 뒤."""
    timeout_sec: float
    """답 하나를 기다리는 상한."""

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("prompts must not be empty")
        if self.timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be > 0, got {self.timeout_sec=}")


@dataclass(frozen=True)
class Decision:
    """질의 하나에 대한 정책 판단."""

    stage: Stage
    top_score: float | None
    """리랭커 1위 점수. 검색 결과가 없으면 None."""
    context: tuple[Hit, ...]
    """답변 근거로 넘길 청크 — low 이상인 것만, 점수 내림차순. refuse 면 비어 있다."""
    profile: Profile | None
    """생성 설정. refuse 면 None (LLM 을 부르지 않는다)."""


class AnswerPolicy:
    """검색 결과의 리랭커 점수를 보고 답변 단계 · 근거 · 생성 설정을 고른다.

    예)
        policy = AnswerPolicy(Thresholds(high=0.99, low=0.5), {"grounded": g, "partial": p})
        decision = policy.decide(result.hits)
        if decision.profile is None:
            ...  # refuse — 고정 문구와 시도한 검색어만 돌려준다
    """

    def __init__(self, thresholds: Thresholds, profiles: Mapping[str, Profile]) -> None:
        if set(profiles) != LLM_STAGES:
            raise ValueError(
                f"profiles must be exactly {sorted(LLM_STAGES)} (refuse calls no LLM), "
                f"got {sorted(profiles)}"
            )
        self.thresholds = thresholds
        self.profiles: Mapping[str, Profile] = MappingProxyType(dict(profiles))

    def decide(self, hits: Sequence[Hit]) -> Decision:
        """리랭커 점수(0~1)가 붙은 검색 결과로 단계를 정한다. low 미만 청크는 근거에서 뺀다."""
        top_score = max((hit.score for hit in hits), default=None)
        stage = self.thresholds.stage(top_score)
        relevant = [hit for hit in hits if hit.score >= self.thresholds.low]
        return Decision(
            stage=stage,
            top_score=top_score,
            context=tuple(sorted(relevant, key=lambda hit: hit.score, reverse=True)),
            profile=self.profiles.get(stage),
        )
