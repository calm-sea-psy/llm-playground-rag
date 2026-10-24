"""답 라벨: 시간이 걸린 질문의 답이 오늘 · 옛 값 · '지금은 모른다'를 어떻게 다루는지 (프롬프트 v3 · v4 비교).

라벨은 작은 물음에서 계산한다 — 판정 모델에 복합 · 부정형 라벨을 바로 물으면 극성을 헷갈렸다 (2026-10-24, 148답).
- 문항 속성 (사람이 문항마다 한 번 정해 고정): `Traits`
- 답에 대한 물음 (판정 모델 — `answer_judge`): `JudgeAnswers`
- 날짜 오류는 규칙(`date_error`) — 같은 라벨을 판정 모델에 맡기면 사람 양성 4 중 2 · 오판 8 이었다
"""

import re
from dataclasses import dataclass
from datetime import date

# 오늘 · 현재를 가리키는 말에 바로 붙은 날짜 ("현재 시점(2026년 10월 1일 기준)" · "오늘이 2025년 1월 26일")
_ANCHORED_DATE = re.compile(
    r"(오늘|현재 날짜|현재 시점|현재 시각|오늘 날짜)\s*[은는이가]?\s*[(（]?\s*(?:인\s*)?"
    r"(\d{4})년\s*(\d{1,2})월(?:\s*(\d{1,2})일)?"
)
# 날짜 바로 뒤의 의문형 — "현재 시점이 2026년 10월 1일 이전인지 이후인지" 는 오늘을 모른다는 말이다
_QUESTION_FORM = re.compile(r"^[^.]{0,12}인지")


@dataclass(frozen=True)
class Traits:
    """문항 속성. 답이 아니라 질문의 것이라 판정 모델에 묻지 않고 사람이 정한다."""

    should_say_unknown: bool
    """올바른 답이 "지금 값은 모른다"여야 하는가 — 시세 · 날씨처럼 덤프가 답할 수 없는 것.
    "대통령은 누구야?"는 지금 값을 묻지만 올바른 답은 "2026년 10월 1일 기준 X"라 아니다."""
    answer_in_context: bool
    """검색된 문서에 답이 실제로 들어 있나. should_say_unknown 이 아닌 문항의 과잉 거절 분모를 정한다."""


@dataclass(frozen=True)
class JudgeAnswers:
    """판정 모델이 답 하나에 대해 답한 물음 셋."""

    gives_value: bool
    """질문이 묻는 값(순위 · 가격 · 수치 · 이름 · 결과 · 상황)을 내놓았나 — 시점이 붙었든 아니든."""
    undated_value: bool
    """그 값을 언제 것인지 밝히지 않은 채 내놓았나."""
    says_unknown: bool
    """지금 값 · 상황을 알 수 없다거나, 문서에 없다거나, 다른 곳에서 확인해야 한다는 말이 있나."""


@dataclass(frozen=True)
class AnswerLabels:
    """답 하나의 라벨. 해당하지 않는 문항이면 None."""

    date_error: bool
    """오늘을 실제 오늘이 아닌 날로 단정하거나 다른 날짜를 오늘로 상정했다 (규칙)."""
    stale_as_current: bool | None
    """옛 값을 지금 값처럼 — should_say_unknown 문항에서 시점 없는 값을 내놓고 모른다는 말도 없다."""
    no_current_disclaimer: bool | None
    """지금 모름 미표시 — should_say_unknown 문항에서 지금 값은 모른다는 말이 없다 (날짜를 밝혔어도)."""
    over_refusal: bool | None
    """과잉 거절 — 문서에 답이 있는 문항에서 값을 내놓지 않고 모른다고만 했다."""
    extra_caution: bool | None
    """불필요한 모름 — should_say_unknown 이 아닌 문항에서 값을 내놓고도 모른다는 말을 붙였다."""


def date_error(text: str, *, today: date, dump_date: date) -> bool:
    """오늘 · 현재에 붙은 날짜가 실제 오늘과 다르면 날짜 오류 (2026-10-24 사람 확인으로 정한 규칙).

    덤프 날짜가 아닌 날짜를 오늘에 붙이면 확언이든 의문이든 오류 — 오늘을 문서의 시점에 묶은 것이다.
    덤프 날짜는 단정("기준이므로")이면 오류, 의문형("이전인지 이후인지")이면 오늘을 모른다는 맞는 추론이라 뺀다.
    월만 쓰고 그 달이 오늘의 달이면 맞은 것으로 본다.
    """
    for match in _ANCHORED_DATE.finditer(text):
        year, month = int(match[2]), int(match[3])
        day = int(match[4]) if match[4] else None
        if (year, month) == (today.year, today.month) and day in (None, today.day):
            continue
        if (year, month, day) == (
            dump_date.year,
            dump_date.month,
            dump_date.day,
        ) and _QUESTION_FORM.search(text[match.end() :]):
            continue
        return True
    return False


def label_answer(
    text: str, traits: Traits, judge: JudgeAnswers, *, today: date, dump_date: date
) -> AnswerLabels:
    """문항 속성 · 판정 물음 · 날짜 규칙으로 답의 라벨을 계산한다."""
    should, unknown = traits.should_say_unknown, judge.says_unknown
    answerable = not should and traits.answer_in_context
    return AnswerLabels(
        date_error=date_error(text, today=today, dump_date=dump_date),
        stale_as_current=(judge.undated_value and not unknown) if should else None,
        no_current_disclaimer=(not unknown) if should else None,
        over_refusal=(unknown and not judge.gives_value) if answerable else None,
        extra_caution=(unknown and judge.gives_value) if not should else None,
    )
