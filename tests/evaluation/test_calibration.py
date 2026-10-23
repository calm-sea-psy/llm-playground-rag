import math

import pytest

from playground_wiki.evaluation import (
    LabeledScores,
    bootstrap_fits,
    clopper_pearson,
    fit_thresholds,
    logit,
    share_below,
    sigmoid,
)


def _scores(grounded: list[float], partial: list[float], refuse: list[float]) -> LabeledScores:
    return LabeledScores(grounded=tuple(grounded), partial=tuple(partial), refuse=tuple(refuse))


def test_logit_round_trips_and_stays_finite_at_the_edges():
    assert sigmoid(logit(0.999)) == pytest.approx(0.999)
    assert math.isfinite(logit(1.0))
    assert math.isfinite(logit(0.0))
    assert logit(0.999) - logit(0.99) > 2  # 확률로는 0.009 차이지만 로짓으로는 크게 벌어진다


def test_high_keeps_99_percent_of_grounded_above_it():
    fit = fit_thresholds(_scores([0.99] * 99 + [0.6], [0.3], [0.1]))

    assert 0.98 < fit.high <= 0.99  # 맨 아래 1개(0.6)는 high 아래로 보내도 된다
    assert share_below((0.99,), logit(fit.high)) == 0


def test_low_never_refuses_more_than_1_percent_of_grounded():
    grounded = [0.4, 0.45] + [0.99] * 98  # 2% 가 낮다
    fit = fit_thresholds(_scores(grounded, [0.9] * 10, [0.42] * 10))

    assert share_below(grounded, logit(fit.low)) <= 0.01


def test_partial_weight_moves_low():
    # 가중치 1 이면 (0.5, 0.9] 가 최선(1.0 − 0.4)이고, partial 거절을 2배로 치면 (0.2, 0.4] 가 낫다
    scores = _scores([0.99] * 100, [0.4] * 4 + [0.9] * 6, [0.2] * 10 + [0.5] * 10)

    plain = fit_thresholds(scores, partial_weight=1.0)
    heavy = fit_thresholds(scores, partial_weight=2.0)

    assert 0.5 < plain.low <= 0.9
    assert plain.objective == pytest.approx(0.6)
    assert 0.2 < heavy.low <= 0.4
    assert heavy.objective == pytest.approx(0.5)


def test_low_takes_the_lower_value_on_ties():
    scores = _scores([0.99] * 100, [0.4] * 4 + [0.9] * 6, [0.2] * 10 + [0.5] * 10)
    fit = fit_thresholds(scores, partial_weight=2.0)

    assert fit.low < 0.21  # (0.2, 0.4] 가 모두 같은 값 — 그중 가장 낮은 격자점


def test_bootstrap_is_repeatable_with_a_seed():
    scores = _scores([0.99, 0.98, 0.9, 0.97], [0.5, 0.6, 0.3], [0.1, 0.2, 0.7])

    first = bootstrap_fits(scores, rounds=20, seed=3)
    again = bootstrap_fits(scores, rounds=20, seed=3)

    assert first == again
    assert len(first) == 20


def test_clopper_pearson_matches_known_bounds():
    assert clopper_pearson(0, 162) == (0.0, pytest.approx(0.0225, abs=5e-4))
    assert clopper_pearson(1, 162)[1] == pytest.approx(0.0339, abs=5e-4)
    lower, upper = clopper_pearson(5, 10)
    assert lower == pytest.approx(0.187, abs=1e-3)
    assert upper == pytest.approx(0.813, abs=1e-3)


def test_scores_need_every_label():
    with pytest.raises(ValueError, match="every label"):
        _scores([0.9], [], [0.1])
