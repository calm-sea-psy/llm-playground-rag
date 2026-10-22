"""문서 · 청크 · 검색 결과 자료형."""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    """문서 안의 섹션 하나. 섹션 헤더(`== 제목 ==`)를 경계로 나눈다."""

    level: int
    """헤더 깊이. 문서 첫머리(헤더 앞 본문)는 0, `== 제목 ==` 은 2."""
    title: str
    """섹션 제목. 문서 첫머리는 빈 문자열."""
    text: str
    """정제된 섹션 본문."""


@dataclass(frozen=True)
class Document:
    """`Source` 가 내놓는 정제된 문서 하나."""

    id: str
    """코퍼스 안에서 유일한 문서 id. 위키는 페이지 id."""
    title: str
    text: str
    """정제된 전체 본문. 섹션 본문을 순서대로 이은 것."""
    sections: tuple[Section, ...]
    categories: tuple[str, ...] = ()
    """분류 이름. 본문에서 빼서 여기에 둔다."""
    meta: Mapping[str, str] = field(default_factory=dict)
    """source · url · dump_date 등 문자열 메타데이터."""


@dataclass(frozen=True)
class Chunk:
    """임베딩 · 검색 단위가 되는 문서 조각."""

    id: str
    """`{doc_id}-{seq}` 형태의 청크 id."""
    doc_id: str
    title: str
    """원문서 제목."""
    section: str
    """원문서 섹션 제목. 문서 첫머리면 빈 문자열."""
    text: str
    """`"{제목} > {섹션}\\n{본문}"` 형태. 이 문자열을 그대로 임베딩한다."""
    seq: int
    """문서 안 청크 순서. 0부터."""


@dataclass(frozen=True)
class Hit:
    """검색 결과 한 건."""

    chunk: Chunk
    score: float
    """검색기마다 척도가 다르다 (dense 는 코사인, BM25 는 원점수, 리랭커는 0~1)."""


@dataclass(frozen=True)
class Filters:
    """검색 범위 제한. 빈 필드는 제한 없음."""

    doc_ids: frozenset[str] = frozenset()
    categories: frozenset[str] = frozenset()
    """이 중 하나라도 속한 문서만."""
