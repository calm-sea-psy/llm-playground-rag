"""위키 마크업을 섹션별 평문으로 바꾸는 정제기."""

import re
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass

import mwparserfromhell
from mwparserfromhell.nodes import Heading, Tag, Template, Wikilink
from mwparserfromhell.wikicode import Wikicode
from playground_corpus import Section

CATEGORY_PREFIXES = ("분류:", "category:")
MEDIA_PREFIXES = ("파일:", "file:", "그림:", "image:", "미디어:", "media:")

# 본문 검색에 도움이 안 되는 꼬리 섹션. 하위 섹션까지 함께 뺀다
DROP_SECTIONS = frozenset(
    {
        "각주",
        "주석",
        "내용주",
        "출처",
        "참고 문헌",
        "참고문헌",
        "참고 자료",
        "참고자료",
        "외부 링크",
        "외부링크",
        "같이 보기",
        "함께 보기",
        "관련 항목",
        "관련항목",
    }
)

# 내용째 버리는 태그. table 은 셀 단위 문장이 깨져 검색 품질을 깎으므로 통째로 뺀다
DROP_TAGS = frozenset(
    {
        "ref",
        "references",
        "gallery",
        "math",
        "chem",
        "ce",
        "timeline",
        "score",
        "syntaxhighlight",
        "source",
        "imagemap",
        "templatestyles",
        "mapframe",
        "maplink",
        "graph",
        "hiero",
        "table",
    }
)

# 마지막 위치 인자가 본문에 드러나는 글자인 틀 ({{lang|en|Seoul}} → Seoul)
KEEP_LAST_PARAM_TEMPLATES = frozenset({"lang", "llang", "국기", "국기나라", "깃발나라", "nowrap"})

# 첫 세 위치 인자가 연 · 월 · 일인 날짜 틀. KorQuAD 에 생몰일 질문이 많아 날짜를 살린다
DATE_TEMPLATES = frozenset(
    {
        "출생일",
        "출생일과 나이",
        "사망일",
        "사망일과 나이",
        "birth date",
        "birth date and age",
        "death date",
        "death date and age",
    }
)

# 함선 이름 틀. {{USS|폼프렛|SS-391}} → USS 폼프렛 (접두 + 첫 위치 인자)
SHIP_PREFIX_TEMPLATES = frozenset({"uss", "hms", "hmas", "hmcs", "sms", "rms", "ins", "jds"})

DISAMBIGUATION_MARKERS = ("동음이의", "disambig")

_INTERLANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[a-z]+)*:")
_LEFTOVER_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_LEFTOVER_LINK = re.compile(
    r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]"
)  # 파서가 놓친 [[대상|표시]] → 표시
_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")  # 닫히지 않아 파서가 놓친 태그
_BROKEN_REF = re.compile(r"<ref\b[^>\n]*$", re.MULTILINE)  # 따옴표가 안 닫혀 줄 끝까지 이어진 각주
_TABLE_LINE = re.compile(r"^[ \t]*(?:\{\||\|\}|\|-|\||!).*$", re.MULTILINE)  # 파서가 놓친 표 줄
_LIST_MARKER = re.compile(
    r"^[ \t]*[*#:;]+[ \t]*", re.MULTILINE
)  # 앞의 수식을 지우면 공백 뒤에 남는다
_MAGIC_WORD = re.compile(r"__[A-Z]+__")  # __NOTOC__ · __TOC__ 등
_FOOTNOTE_MARK = re.compile(r"\[(?:\d+|주 \d+|각주 \d+)\]")
_LEFTOVER_BRACKETS = re.compile(r"\[\[|\]\]|\{\{|\}\}|\{\||\|\}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,])")  # 지운 각주 · 틀 자리에 남은 공백
_EMPTY_PARENS = re.compile(r"[(（]\s*[,;、]?\s*[)）]")
_SPACES = re.compile(r"[ \t\u00a0\u200b]+")  # 공백 · 탭 · NBSP · zero-width space
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_MANY_NEWLINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class CleanedPage:
    """정제한 문서 한 건의 본문 구조."""

    sections: tuple[Section, ...]
    """본문이 있는 섹션만, 문서 순서대로. 첫머리는 level 0."""
    categories: tuple[str, ...]
    is_disambiguation: bool
    """동음이의 틀이나 분류가 붙은 문서."""

    @property
    def text(self) -> str:
        """섹션 본문을 빈 줄로 이은 전체 본문."""
        return "\n\n".join(section.text for section in self.sections)


class WikiCleaner:
    """위키 마크업에서 틀 · 각주 · 표 · 파일 링크를 걷어 내고 섹션 구조를 남긴다.

    예)
        page = WikiCleaner().clean(wikitext)
        page.sections[1].title  # "역사"
    """

    def clean(self, wikitext: str) -> CleanedPage:
        """위키 마크업을 정제한 섹션 목록과 분류로 바꾼다."""
        code = mwparserfromhell.parse(wikitext)
        is_disambiguation = _has_disambiguation_template(code)
        categories = _pop_links(code)
        _drop_tags(code)
        _rewrite_templates(code)
        is_disambiguation = is_disambiguation or any(
            marker in category for category in categories for marker in DISAMBIGUATION_MARKERS
        )
        return CleanedPage(
            sections=self._sections(code),
            categories=tuple(dict.fromkeys(categories)),
            is_disambiguation=is_disambiguation,
        )

    def _sections(self, code: Wikicode) -> tuple[Section, ...]:
        sections: list[Section] = []
        drop_below: int | None = None  # 빼는 섹션의 레벨. 그보다 깊은 하위 섹션도 뺀다
        for part in code.get_sections(flat=True, include_lead=True):
            nodes = part.nodes
            if nodes and isinstance(nodes[0], Heading):
                level = nodes[0].level
                title = clean_text(nodes[0].title.strip_code())
                body = Wikicode(nodes[1:])
            else:
                level, title, body = 0, "", part
            if drop_below is not None and level > drop_below:
                continue
            drop_below = level if title in DROP_SECTIONS else None
            if drop_below is not None:
                continue
            text = clean_text(body.strip_code(normalize=True, collapse=True))
            if text:
                sections.append(Section(level=level, title=title, text=text))
        return tuple(sections)


def clean_text(text: str) -> str:
    """strip_code 뒤에 남은 마크업 조각과 공백을 정리한다."""
    previous = None
    while previous != text:  # 안쪽부터 한 겹씩 남은 틀을 걷는다
        previous = text
        text = _LEFTOVER_TEMPLATE.sub("", text)
    text = _LEFTOVER_LINK.sub(r"\1", text)
    text = _BROKEN_REF.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _TABLE_LINE.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _MAGIC_WORD.sub("", text)
    text = _FOOTNOTE_MARK.sub("", text)
    text = _LEFTOVER_BRACKETS.sub("", text)
    text = _EMPTY_PARENS.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def _has_disambiguation_template(code: Wikicode) -> bool:
    for template in code.ifilter_templates(recursive=True):
        name = str(template.name).strip().lower()
        if any(marker in name for marker in DISAMBIGUATION_MARKERS):
            return True
    return False


def _pop_links(code: Wikicode) -> list[str]:
    """분류 링크를 빼서 이름을 돌려주고, 파일 · 언어 간 링크는 지운다."""
    categories: list[str] = []
    for link in code.filter_wikilinks(recursive=True):
        target = str(link.title).strip()
        lowered = target.lower()
        if lowered.startswith(CATEGORY_PREFIXES):
            categories.append(target.split(":", 1)[1].strip())
            _remove(code, link)
        elif lowered.startswith(MEDIA_PREFIXES) or _is_interlanguage(target):
            _remove(code, link)
    return categories


def _drop_tags(code: Wikicode) -> None:
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).strip().lower() in DROP_TAGS:
            _remove(code, tag)


def _rewrite_templates(code: Wikicode) -> None:
    """글자가 드러나는 틀은 그 글자로 바꾸고 나머지 틀은 지운다."""
    for template in code.filter_templates(recursive=True):
        name = str(template.name).strip().lower()
        replacement = _template_text(name, template)
        _replace(code, template, replacement)


def _template_text(name: str, template: Template) -> str:
    positional = [str(param.value).strip() for param in template.params if not param.showkey]
    if name in DATE_TEMPLATES:
        return _format_date(positional[:3])
    if name in KEEP_LAST_PARAM_TEMPLATES or name.startswith(("lang-", "llang")):
        return positional[-1] if positional else ""
    if name in SHIP_PREFIX_TEMPLATES:
        return f"{name.upper()} {positional[0]}" if positional else ""
    return ""


def _format_date(parts: Sequence[str]) -> str:
    units = ("년", "월", "일")
    pieces = [
        f"{part.lstrip('0') or '0'}{unit}"
        for part, unit in zip(parts, units, strict=False)
        if part.isdigit()
    ]
    return " ".join(pieces)


def _is_interlanguage(target: str) -> bool:
    # [[en:Seoul]] · [[zh-yue:首爾]] 처럼 언어 코드 접두가 붙은 링크
    return _INTERLANGUAGE.match(target) is not None


def _remove(code: Wikicode, node: Wikilink | Tag) -> None:
    with suppress(ValueError):  # 이미 지운 노드 안에 있던 노드
        code.remove(node)


def _replace(code: Wikicode, node: Template, value: str) -> None:
    with suppress(ValueError):  # 이미 지운 노드 안에 있던 노드
        code.replace(node, value)
