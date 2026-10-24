import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import date

import pytest
from fastapi.testclient import TestClient
from playground_corpus import Chunk
from playground_llm import (
    FakeLLM,
    Generation,
    GenerationError,
    Message,
    ModelSpec,
    PromptLibrary,
    StreamDelta,
    Usage,
)
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.api.answer import CASCADE_MESSAGE, REFUSAL, AnswerEvent, AnswerService
from playground_wiki.api.app import create_app
from playground_wiki.gate import QueryGate, Token
from playground_wiki.generation import PROMPT_DIR, AnswerPolicy, Profile, Thresholds

LIBRARY = PromptLibrary.load(PROMPT_DIR)
SPEC = ModelSpec("ollama", "fake", {"think": "off", "max_tokens": 64})
GROUNDED = Profile(SPEC, system=("base@v4", "grounded@v2"), user="question@v1", timeout_sec=5)
PARTIAL = Profile(SPEC, system=("base@v4", "partial@v2"), user="question@v1", timeout_sec=5)
QUESTION = "경복궁을 중건한 사람은?"


class _NoTokens:
    """형태소가 없는 분석기 — 실시간 규칙은 걸리지 않고 잡담 사전만 본다 (Kiwi 를 올리지 않는다)."""

    def tokenize(self, text: str) -> list[Token]:
        return []


class FakeSearch:
    def __init__(self, score: float | None) -> None:
        self.score = score
        self.calls: list[str] = []

    async def search(self, query: str) -> SearchResult:
        self.calls.append(query)
        if self.score is None:
            return SearchResult(hits=(), took_ms=1.0)
        chunk = Chunk(
            id="1-0",
            doc_id="1",
            title="경복궁",
            section="역사",
            text="흥선대원군이 중건했다.",
            seq=0,
            token_count=4,
        )
        return SearchResult(hits=(Hit(chunk=chunk, score=self.score),), took_ms=1.0)


def _by_stage(
    grounded: str, partial: str = "### 위키백과 문서에서\n직접적인 내용은 없습니다."
) -> FakeLLM:
    """system 프롬프트로 단계를 알아보고 단계마다 정해 둔 답을 낸다."""
    return FakeLLM(
        [lambda spec, messages: grounded if "근거 답변" in messages[0].content else partial]
    )


def _service(
    llm: object, score: float | None = 0.95, **kwargs: object
) -> tuple[AnswerService, FakeSearch]:
    search = FakeSearch(score)
    policy = AnswerPolicy(Thresholds(high=0.9, low=0.5), {"grounded": GROUNDED, "partial": PARTIAL})
    service = AnswerService(
        search=search,
        policy=policy,
        library=LIBRARY,
        llm=llm,
        gate=QueryGate(_NoTokens()),
        dump_date="2026년 10월 1일",
        today=lambda: date(2026, 10, 24),
        **kwargs,
    )
    return service, search


def _events(service: AnswerService, question: str = QUESTION) -> list[AnswerEvent]:
    async def collect() -> list[AnswerEvent]:
        return [event async for event in service.answer(question)]

    return asyncio.run(collect())


def _names(events: list[AnswerEvent]) -> list[str]:
    return [event.name for event in events]


def _text(events: list[AnswerEvent]) -> str:
    return "".join(str(event.data["text"]) for event in events if event.name == "delta")


def _done(events: list[AnswerEvent]) -> dict:
    return next(event.data for event in events if event.name == "done")


def test_smalltalk_is_answered_by_the_gate_without_search():
    service, search = _service(_by_stage("찾음\n답"))

    events = _events(service, "안녕!")

    assert _names(events) == ["stage", "delta", "done"]
    assert events[0].data["stage"] == "gate"
    assert search.calls == []
    assert _done(events)["diagnostics"]["gate"]["kind"] == "greeting"


def test_no_hits_is_a_fixed_refusal_with_the_query():
    service, _ = _service(_by_stage("찾음\n답"), score=None)

    events = _events(service)

    assert _names(events) == ["stage", "delta", "done"]
    assert events[0].data["stage"] == "refuse"
    assert _text(events) == REFUSAL.format(query=QUESTION)


def test_grounded_answer_drops_the_marker_and_out_of_range_citations():
    service, _ = _service(_by_stage("찾음\n흥선대원군이 중건했다 [1][7]."))

    events = _events(service)

    assert _names(events)[:3] == ["sources", "stage", "delta"]
    assert events[0].data["sources"][0]["title"] == "경복궁"
    assert _text(events) == "흥선대원군이 중건했다 [1]."
    done = _done(events)
    assert (done["stage_path"], done["cited"], done["dropped"]) == (["grounded"], [1], [7])
    assert done["diagnostics"]["marker"] == "found"


def test_not_found_cascades_to_partial_before_any_text():
    llm = _by_stage("못 찾음", partial="### 위키백과 문서에서\n직접적인 내용은 없습니다.")
    service, _ = _service(llm)

    events = _events(service)

    stages = [event.data for event in events if event.name == "stage"]
    assert [s["stage"] for s in stages] == ["grounded", "partial"]
    assert stages[1]["message"] == CASCADE_MESSAGE
    assert _text(events).startswith("### 위키백과 문서에서")
    assert _done(events)["stage_path"] == ["grounded", "partial"]
    assert len(llm.calls) == 2


def test_empty_grounded_answer_also_cascades():
    service, _ = _service(_by_stage("찾음\n\n"))

    assert _done(_events(service))["stage_path"] == ["grounded", "partial"]


def test_empty_partial_answer_ends_with_the_fixed_refusal():
    service, _ = _service(
        _by_stage("찾음\n답", partial=""), score=0.6
    )  # low ≤ 0.6 < high → partial

    events = _events(service)

    assert _done(events)["stage_path"] == ["partial"]
    assert _text(events) == REFUSAL.format(query=QUESTION)


class ScriptedLLM:
    """조각을 정해 둔 대로(사이에 쉬기도 하며) 흘리는 대역. 응답 스트림이 닫혔는지 기록한다."""

    def __init__(self, pieces: Sequence[str | float | Exception]) -> None:
        self.pieces = pieces
        self.closed = threading.Event()
        self.produced = 0

    def stream(
        self, spec: ModelSpec, messages: Sequence[Message]
    ) -> Iterator[StreamDelta | Generation]:
        try:
            for piece in self.pieces:
                if isinstance(piece, float):
                    time.sleep(piece)
                elif isinstance(piece, Exception):
                    raise piece
                else:
                    self.produced += 1
                    yield StreamDelta("text", piece)
            yield Generation(
                text="", model=spec, usage=Usage(prompt_tokens=1, completion_tokens=1, total_ms=0)
            )
        finally:
            self.closed.set()


def test_failure_before_any_text_is_an_error_event():
    service, _ = _service(ScriptedLLM([GenerationError("server down")]))

    events = _events(service)

    assert _names(events)[-1] == "error"
    assert "done" not in _names(events)


def test_after_text_is_sent_a_timeout_closes_with_done_not_a_new_stage():
    slow = ScriptedLLM(["찾음\n", "흥선대원군이 ", 0.5, "중건했다."])
    profile = Profile(SPEC, system=GROUNDED.system, user=GROUNDED.user, timeout_sec=0.2)
    service, _ = _service(slow)
    service._policy = AnswerPolicy(
        Thresholds(high=0.9, low=0.5), {"grounded": profile, "partial": PARTIAL}
    )

    events = _events(service)

    assert [e.data["stage"] for e in events if e.name == "stage"] == [
        "grounded"
    ]  # partial 로 넘어가지 않는다
    assert "error" not in _names(events)
    done = _done(events)
    assert (done["timed_out"], done["stage_path"]) == (True, ["grounded"])
    assert _text(events) == "흥선대원군이 "


def test_not_found_closes_the_model_stream_instead_of_reading_it_to_the_end():
    # 못 찾음 뒤에 긴 출력이 이어져도 끝까지 읽지 않고 응답 스트림을 닫는다 — 작업 스레드가 다음 조각에서 멈춘다
    chatty = ScriptedLLM(["못 찾음\n", *["더 쓰는 말 "] * 200])
    service, _ = _service(chatty)
    service._llm = _Switch(
        grounded=chatty, partial=_by_stage("", partial="### 위키백과 문서에서\n없음")
    )

    _events(service)

    assert chatty.closed.wait(2)
    assert chatty.produced < 200


class _Switch:
    """단계마다 다른 LLM 대역을 쓴다 (system 프롬프트로 단계를 알아본다)."""

    def __init__(self, grounded: object, partial: object) -> None:
        self.grounded, self.partial = grounded, partial

    def stream(
        self, spec: ModelSpec, messages: Sequence[Message]
    ) -> Iterator[StreamDelta | Generation]:
        target = self.grounded if "근거 답변" in messages[0].content else self.partial
        return target.stream(spec, messages)


def test_question_text_is_not_logged_unless_asked(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="playground_wiki.api.answer")
    quiet, _ = _service(_by_stage("찾음\n답"))
    loud, _ = _service(_by_stage("찾음\n답"), log_questions=True)

    _events(quiet)
    quiet_record = json.loads(caplog.records[-1].getMessage().removeprefix("ask "))
    _events(loud)
    loud_record = json.loads(caplog.records[-1].getMessage().removeprefix("ask "))

    assert "question" not in quiet_record
    assert quiet_record["question_chars"] == len(QUESTION)
    assert len(quiet_record["question_sha256"]) == 64
    assert loud_record["question"] == QUESTION


def test_ask_route_streams_server_sent_events():
    app = create_app()
    app.state.answer_service, _ = _service(_by_stage("찾음\n흥선대원군이 중건했다 [1]."))
    client = TestClient(app)  # with 없이 — lifespan(모델 · 저장소)을 돌리지 않는다

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in response.text
    assert client.post("/ask", json={"question": "  "}).status_code == 422
