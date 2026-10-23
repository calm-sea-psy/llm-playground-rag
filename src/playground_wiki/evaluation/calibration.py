"""답변 정책 임계값 고르기: 리랭커 1위 점수로 `high` · `low` 를 정하는 규칙 (ADR-024).

규칙은 라벨(grounded · partial · refuse)별 비율만 쓴다 — 정확도 · precision 은 라벨이 섞인 비율에 따라
달라지는데, 고르는 데 쓴 세트의 섞인 비율은 실제 질문과 다르다.
점수는 0.99 이상에 몰리므로 격자는 로짓 축에 놓는다 (0.99 와 0.999 는 로짓 4.6 대 6.9).
"""

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

LOGIT_STEP = 0.05  # 격자 간격 (로짓). 0.99 ~ 0.999 사이에 약 46칸
EPS = 1e-7  # 점수를 (EPS, 1 − EPS) 로 잘라 로짓이 유한하게
HIGH_RECALL = 0.99  # grounded 의 이 비율 이상이 high 위로 — 캐스케이드라 high 는 지연 손잡이 (임시, F9 에서 확정)
MAX_FALSE_REFUSAL = 0.01  # grounded 를 거절하는 비율 상한 — 가장 비싼 실수
# partial 잘못 거절(답 하나를 잃음)의 무게 — refuse 를 통과시킨 비용을 1 로 둔 상대값, w ≈ 1/r.
# r = refuse 질문이 partial 로 갔을 때 해로운 답이 나오는 비율 (범위 밖 질문에 일반 지식으로 답하는 건 비용이 아니다 — ADR-025).
# 2026-10-23 probe(partial think off): fit 대상 39 중 1 → r 2.6%, CP 95% 상한 13.5% → w ≥ 7.4.
# 상한 쪽(가장 작은 w)을 쓴다 — 점추정(39)과 같은 low 가 나온다
PARTIAL_WEIGHT = 7.4


@dataclass(frozen=True)
class LabeledScores:
    """라벨별 리랭커 1위 점수 (0~1)."""

    grounded: tuple[float, ...]
    """상위 5 에 정답이 든 질문."""
    partial: tuple[float, ...]
    """관련 문서는 있지만 문맥에 답이 없는 질문 (검색이 정답을 놓친 KorQuAD 포함)."""
    refuse: tuple[float, ...]
    """위키로 답할 근거가 없는 질문."""

    def __post_init__(self) -> None:
        if not (self.grounded and self.partial and self.refuse):
            raise ValueError("every label needs at least one score")


@dataclass(frozen=True)
class Fit:
    """고른 임계값 (확률). score ≥ high → grounded 먼저, score < low → refuse, 그 사이 → partial."""

    high: float
    low: float
    objective: float
    """low 에서의 `refuse 거절 비율 − partial_weight × partial 거절 비율`."""


def logit(score: float) -> float:
    """확률 → 로짓. 0 · 1 근처는 EPS 로 잘라 유한하게."""
    clipped = min(max(score, EPS), 1 - EPS)
    return math.log(clipped / (1 - clipped))


def sigmoid(value: float) -> float:
    """로짓 → 확률."""
    return 1 / (1 + math.exp(-value))


def logit_grid(scores: LabeledScores) -> tuple[float, ...]:
    """모든 점수를 덮는 로짓 격자 (`LOGIT_STEP` 간격, 오름차순)."""
    values = [logit(s) for s in (*scores.grounded, *scores.partial, *scores.refuse)]
    start = math.floor(min(values) / LOGIT_STEP)
    # 최댓값보다 한 칸 위까지 — 모든 점수를 아래에 두는 경우도 후보로
    stop = math.ceil(max(values) / LOGIT_STEP) + 1
    return tuple(i * LOGIT_STEP for i in range(start, stop + 1))


def share_below(scores: Sequence[float], threshold: float) -> float:
    """로짓 threshold 미만인 점수의 비율 (`Thresholds.stage` 의 refuse 조건과 같은 쪽)."""
    return sum(logit(s) < threshold for s in scores) / len(scores)


def fit_thresholds(scores: LabeledScores, *, partial_weight: float = PARTIAL_WEIGHT) -> Fit:
    """규칙대로 high · low 를 고른다.

    - high: grounded 의 `HIGH_RECALL` 이상이 high 위에 남는 가장 높은 격자점
    - low: grounded 거절 ≤ `MAX_FALSE_REFUSAL` 이고 low ≤ high 인 격자점 중
      `refuse 거절 비율 − partial_weight × partial 거절 비율` 이 가장 큰 것, 같으면 낮은 쪽 (답을 덜 잃는 쪽)
    """
    grid = logit_grid(scores)
    high = max(t for t in grid if 1 - share_below(scores.grounded, t) >= HIGH_RECALL)
    candidates = [
        t for t in grid if t <= high and share_below(scores.grounded, t) <= MAX_FALSE_REFUSAL
    ]
    best_low, best_value = candidates[0], -math.inf
    for t in candidates:  # 오름차순이라 같은 값이면 먼저 나온 낮은 쪽이 남는다
        value = share_below(scores.refuse, t) - partial_weight * share_below(scores.partial, t)
        if value > best_value:
            best_low, best_value = t, value
    return Fit(high=sigmoid(high), low=sigmoid(best_low), objective=best_value)


def bootstrap_fits(
    scores: LabeledScores, *, rounds: int, seed: int, partial_weight: float = PARTIAL_WEIGHT
) -> list[Fit]:
    """라벨마다 따로 복원추출해 다시 고른 임계값들 (층화 bootstrap). 구간을 내는 데 쓴다."""
    rng = random.Random(seed)
    fits = []
    for _ in range(rounds):
        resampled = LabeledScores(
            grounded=tuple(rng.choices(scores.grounded, k=len(scores.grounded))),
            partial=tuple(rng.choices(scores.partial, k=len(scores.partial))),
            refuse=tuple(rng.choices(scores.refuse, k=len(scores.refuse))),
        )
        fits.append(fit_thresholds(resampled, partial_weight=partial_weight))
    return fits


def clopper_pearson(successes: int, trials: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """이항 비율의 정확한 양쪽 (1 − alpha) 구간. 0건이면 하한 0, 전부면 상한 1."""
    if not 0 <= successes <= trials or trials == 0:
        raise ValueError(
            f"need 0 <= successes <= trials and trials > 0, got {successes=} {trials=}"
        )
    lower = (
        0.0
        if successes == 0
        else _solve(lambda p: 1 - _binom_cdf(successes - 1, trials, p), alpha / 2)
    )
    upper = (
        1.0
        if successes == trials
        else _solve(lambda p: _binom_cdf(successes, trials, p), alpha / 2, decreasing=True)
    )
    return lower, upper


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _solve(f: Callable[[float], float], target: float, *, decreasing: bool = False) -> float:
    # f 가 p 에 대해 단조라 이분법으로 f(p) = target 인 p 를 찾는다
    low, high = 0.0, 1.0
    for _ in range(100):
        mid = (low + high) / 2
        if (f(mid) > target) != decreasing:
            high = mid
        else:
            low = mid
    return (low + high) / 2
