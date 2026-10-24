"""답변 생성에 넘길 대화: 프롬프트 파일의 지시는 system 턴, 검색한 문서 · 질문은 user 턴.

문서를 지시와 다른 턴에 두는 것은 간접 인젝션 방어의 일부다 — 문서 안의 문장이 지시처럼 읽히지 않게.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from playground_llm import Message, PromptLibrary, PromptTemplate, compose
from playground_retrieval.core import Hit

from playground_wiki.generation.policy import Profile

PROMPT_DIR = Path(__file__).parent / "prompts"
"""위키 프롬프트 파일 (`{이름}/{버전}.md`). 쓰는 쪽이 시작할 때 `PromptLibrary.load(PROMPT_DIR)` 로 읽는다."""
NOT_FOUND = "주어진 문서에서 답을 찾을 수 없습니다."
"""grounded 프롬프트가 답을 못 찾으면 이 문장만 쓰라고 시킨다. 줄글 답의 거절 판정에 그대로 쓴다."""
SYSTEM_VARIABLES = frozenset({"dump_date", "not_found"})
"""system 조각이 쓸 수 있는 자리표시자. 문서 · 질문은 여기에 들어가지 않는다."""
USER_VARIABLES = frozenset({"context", "question"})
"""user 조각이 정확히 써야 하는 자리표시자."""
# 문서 본문이 블록 경계를 흉내 내지 못하게 지운다 (본문의 </document> 가 블록을 닫고 그 뒤를 지시처럼 보이게 하는 것)
_DOCUMENT_TAG = re.compile(r"</?documents?\b[^>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class ChatPrompt:
    """LLM 에 넘길 대화와 그 프롬프트 버전."""

    messages: tuple[Message, ...]
    """system(지시) → user(문서 · 질문)."""
    version: str
    """조각마다 `이름@버전:해시` 를 `+` 로 이은 것 (system 조각들 → user 조각). 평가 기록의 prompt_version."""


def build_prompt(
    profile: Profile,
    library: PromptLibrary,
    *,
    question: str,
    context: Sequence[Hit],
    dump_date: str,
) -> ChatPrompt:
    """프로필의 프롬프트 조각으로 system · user 턴을 만든다.

    Args:
      profile: 답변 단계의 생성 설정 (system 조각들 · user 조각).
      library: `PROMPT_DIR` 에서 읽은 프롬프트 모음.
      question: 사용자 질문 원문.
      context: 답변 근거로 넘길 검색 결과 (`Decision.context`).
      dump_date: 덤프 기준일. 사람이 읽는 꼴로 (예: "2026년 10월 1일").

    Raises:
      PromptError: 프로필이 가리키는 프롬프트가 없거나 자리표시자를 채울 수 없을 때.
    """
    system_parts = [library.get(ref) for ref in profile.system]
    # 단계마다 쓰는 값이 달라(not_found 는 grounded 만) 조각들이 쓰는 것만 넘긴다. 빠진 값은 compose 가 잡는다
    system = compose(
        system_parts, _used(system_parts, {"dump_date": dump_date, "not_found": NOT_FOUND})
    )
    user = compose(
        [library.get(profile.user)], {"context": format_context(context), "question": question}
    )
    return ChatPrompt(
        messages=(Message("system", system.text), Message("user", user.text)),
        version=f"{system.version}+{user.version}",
    )


def format_context(hits: Sequence[Hit]) -> str:
    """검색 결과를 `<document id="1" title="…">` 블록으로 이어 붙인다. 본문 속 블록 태그 흉내는 지운다.

    id 는 1부터 hits 순서대로 — 모델은 출처를 이 번호로만 쓰고 서버가 제목 · 링크로 바꾼다 (ADR-026).
    `citations.sources(hits)` 와 같은 번호다.
    """
    blocks = []
    for number, hit in enumerate(hits, start=1):
        title = hit.chunk.title.replace('"', "'")
        text = _DOCUMENT_TAG.sub("", hit.chunk.text)
        blocks.append(f'<document id="{number}" title="{title}">\n{text}\n</document>')
    return "\n\n".join(blocks)


def check_profiles(profiles: Iterable[Profile], library: PromptLibrary) -> None:
    """프로필이 가리키는 프롬프트가 모두 있고, 정해진 자리표시자만 쓰는지 확인한다. 서버를 시작할 때 부른다.

    Raises:
      PromptError: 없는 프롬프트를 가리킬 때.
      ValueError: system 조각이 `SYSTEM_VARIABLES` 밖의 자리표시자를 쓰거나,
        user 조각이 `USER_VARIABLES` 를 정확히 쓰지 않을 때.
    """
    for profile in profiles:
        for ref in profile.system:
            if extra := library.get(ref).variables - SYSTEM_VARIABLES:
                raise ValueError(f"{ref}: system prompt uses unknown placeholders {sorted(extra)}")
        user = library.get(profile.user)
        if user.variables != USER_VARIABLES:
            raise ValueError(
                f"{profile.user}: user prompt must use exactly {sorted(USER_VARIABLES)}, "
                f"got {sorted(user.variables)}"
            )


def _used(parts: Sequence[PromptTemplate], available: Mapping[str, str]) -> dict[str, str]:
    needed = frozenset().union(*(part.variables for part in parts))
    return {name: value for name, value in available.items() if name in needed}
