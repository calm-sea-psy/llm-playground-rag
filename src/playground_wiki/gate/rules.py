"""질문 관문: 잡담 · 실시간 질문을 검색 전에 어휘 규칙으로 골라 고정 답을 준다 (ADR-025).

위키 밖 질문을 거절하는 일은 임계값이 아니라 partial 프롬프트의 "알 수 없음"과 이 관문이 나눠 맡는다.
규칙은 좁게 둔다 — 놓친 질문은 검색 · partial 로 가서 "알 수 없음"이 받지만, 잘못 걸린 질문은 답을 잃는다.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from playground_wiki.gate.tokens import Token, Tokenizer

Rule = Literal["smalltalk", "realtime"]
Kind = Literal["greeting", "thanks", "farewell", "identity", "chitchat", "realtime"]
KINDS: tuple[str, ...] = get_args(Kind)

# 잡담 사전 — 질문 전체가 정규화한 뒤 이 중 하나와 같아야 걸린다. 섞인 문장("안녕, 세종대왕은 누구야?")은 검색으로 간다.
# 정규화가 공백 · 문장부호 · 이모지 · ㅋㅋ · ㅠㅠ · ~ 를 지우고 소문자로 바꾸니 그 변형은 따로 적지 않는다.
# 위키 문서 제목이기도 한 낱말(감사 · 대박 · 수고)은 넣지 않는다
SMALLTALK: Mapping[str, tuple[str, ...]] = {
    "greeting": (
        "안녕",
        "안녕하세요",
        "안녕하십니까",
        "하이",
        "ㅎㅇ",
        "hi",
        "hello",
        "hey",
        "헬로",
        "반가워",
        "반가워요",
        "반갑습니다",
        "처음 뵙겠습니다",
        "좋은 아침",
        "굿모닝",
    ),
    "thanks": (
        "고마워",
        "고마워요",
        "고맙습니다",
        "고맙다",
        "감사해",
        "감사해요",
        "감사합니다",
        "ㄱㅅ",
        "ㄳ",
        "땡큐",
        "thanks",
        "thank you",
        "thx",
        "덕분이야",
        "도움 됐어",
    ),
    "farewell": (
        "잘 가",
        "잘 가요",
        "안녕히 가세요",
        "안녕히 계세요",
        "바이",
        "바이바이",
        "빠이",
        "bye",
        "잘 자",
        "잘 자요",
        "굿나잇",
        "또 봐",
        "다음에 봐",
        "수고했어",
        "수고하세요",
    ),
    "identity": (
        "너 누구야",
        "넌 누구야",
        "너는 누구야",
        "누구세요",
        "너 뭐야",
        "넌 뭐야",
        "너는 뭐야",
        "너 이름이 뭐야",
        "이름이 뭐야",
        "이름 뭐야",
        "네 이름은",
        "자기소개 해 줘",
        "너 뭐 할 수 있어",
        "뭐 할 수 있어",
        "넌 뭘 할 수 있어",
    ),
    "chitchat": (
        "심심해",
        "뭐 해",
        "뭐 하고 있어",
        "기분 어때",
        "오늘 기분 어때",
        "오늘 하루 어땠어",
        "배고파",
        "졸려",
        "피곤해",
        "그래서",
        "그래",
        "응",
        "ㅇㅇ",
        "아니",
        "네",
        "넵",
        "ok",
        "오케이",
        "알겠어",
        "좋아",
    ),
}
# 웃음 · 울음 자모 (ㅋ · ㅠ · ㅜ 는 하나라도, ㅎ 는 'ㅎㅇ' 인사가 있어 두 개 이상만).
# NFKC 가 'ㅋㅠ' 같은 호환 자모를 음절('큐')로 합칠 수 있어 NFKC 전에 지운다. 첫소리 · 가운뎃소리 자모로 들어온 것도 함께
# 호환 자모 ㅋ ㅠ ㅜ ㅎ = U+314B U+3160 U+315C U+314E, 첫소리 ㅋ ㅎ = U+110F U+1112, 가운뎃소리 ㅠ ㅜ = U+1172 U+116E
_LAUGH_CRY = re.compile("[\u314b\u3160\u315c\u110f\u1172\u116e]+|[\u314e\u1112]{2,}")
# 지우는 유니코드 범주: 문장부호(P) · 기호(S, 이모지 · ~ 포함) · 공백(Z) · 제어 · 서식(C, ZWJ) · 결합 표시(Mn, 이모지 변형 선택자)
_DROP_CATEGORIES = ("P", "S", "Z", "C", "Mn")

# 실시간 규칙 — 때를 가리키는 말과 바뀌는 값을 묻는 말이 한 질문에 함께 있어야 걸린다.
# 형태소 형태의 연속으로 맞춘다. '현재' 는 KorQuAD dev 5,774 문항에 27번 나오지만 대상어와 함께 나온 적은 없다.
# 받아들인 오탐: "현재 가격 혁명 시기의 유럽 경제" (현재/MAG + 가격). '요즘' · '올해' · '최근' 은 덤프에 답이 있을 수 있어 넣지 않는다
TIME_MARKERS: tuple[tuple[str, ...], ...] = (
    ("지금",),
    ("오늘",),
    ("현재",),
    ("실시간",),
    ("방금",),
    ("어제",),
    ("내일",),
    ("이번", "주"),
    ("이번", "주말"),
)
# 때 표지 바로 뒤에 이 조사가 오면 누적 · 시작점이지 지금 값이 아니다 (지금까지 가장 많이 팔린 · 오늘부터).
# 다른 조사는 막지 않는다 — "오늘은 날씨 어때?" · "지금은 몇 시야?" · "오늘의 날씨" 는 오늘/NNG + 조사로 나온다
RANGE_PARTICLES = frozenset({"까지", "부터"})
# 때 표지 바로 뒤에 접미사가 오면 다른 낱말이다 (현재형 = 현재/NNG + 형/XSN).
# 어림 접미사는 예외 — "지금쯤 택배 어디야" 도 지금/NNG + 쯤/XSN 으로 나온다
SUFFIX_TAG = "XSN"
APPROX_SUFFIXES = frozenset({"쯤", "경", "께"})
# 위키로 답하는 질문에 걸려 넣지 않는 것: 지수(인간개발지수 · 물가 지수 · 지수함수) · 순위("현재 세계 인구 순위") ·
# 뉴스("현재 KBS 뉴스 9 앵커는 누구" — "오늘 뉴스 헤드라인" 은 헤드라인으로 잡는다).
# Kiwi 가 쪼개는 낱말은 이어진 형태로 적는다: 미세먼지 = 미세 + 먼지, 시청률 = 시청 + 률, 몇 시 = 몇 + 시 (몇 시간과 다르다)
TARGETS: tuple[tuple[str, ...], ...] = (
    ("날씨",),
    ("기온",),
    ("미세", "먼지"),
    ("황사",),
    ("환율",),
    ("주가",),
    ("코스피",),
    ("코스닥",),
    ("나스닥",),
    ("비트코인",),
    ("코인",),
    ("시세",),
    ("가격",),
    ("로또",),
    ("당첨",),
    ("헤드라인",),
    ("속보",),
    ("며칠",),
    ("요일",),
    ("날짜",),
    ("몇", "시"),
    ("출발",),
    ("도착",),
    ("택배",),
    ("배송",),
    ("경기", "결과"),
    ("스코어",),
    ("시청", "률"),
)

# 종류별 고정 답. 실시간 답의 기준일은 덤프 날짜(configs/ingest.yaml source.dump_date)를 넣는다
REPLIES: Mapping[str, str] = {
    "greeting": "안녕하세요! 한국어 위키백과에 있는 내용을 물어보세요.",
    "thanks": "도움이 됐다니 다행이에요. 더 궁금한 게 있으면 물어보세요.",
    "farewell": "안녕히 가세요. 궁금한 게 생기면 또 물어보세요.",
    "identity": (
        "저는 한국어 위키백과 문서를 찾아 답하는 도우미예요. "
        "사람 · 사건 · 개념처럼 위키백과에 있는 내용을 물어보세요."
    ),
    "chitchat": (
        "저는 위키백과 질문에 답하는 도우미라 잡담은 잘 못해요. 궁금한 사람 · 사건 · 개념을 물어보세요."
    ),
    "realtime": (
        "날씨 · 시세 · 뉴스 · 시각 같은 실시간 정보는 알 수 없어요. "
        "저는 {dump_date} 기준 한국어 위키백과 문서로만 답합니다."
    ),
}


@dataclass(frozen=True)
class GateMatch:
    """관문에 걸린 질문. 어떤 규칙 · 토큰이 걸었는지 로그 · 응답 메타데이터에 남긴다."""

    kind: Kind
    matched: tuple[str, ...]
    """건 근거 — 잡담은 정규화한 질문(정규화 뒤 빈 질문이면 비어 있다), 실시간은 때 표지와 대상어."""

    @property
    def rule(self) -> Rule:
        """걸린 규칙."""
        return "realtime" if self.kind == "realtime" else "smalltalk"


def normalize(text: str) -> str:
    """잡담 사전과 맞출 꼴. 웃음 · 울음 자모를 지우고 NFKC · 소문자로 바꾼 뒤 공백 · 문장부호 · 기호를 지운다."""
    text = unicodedata.normalize("NFKC", _LAUGH_CRY.sub("", text)).lower()
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith(_DROP_CATEGORIES))


def smalltalk_table(lexicon: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """잡담 사전을 정규화한 질문 → 종류 표로. 정규화 뒤 비거나 겹치는 말이 있으면 실수라 막는다."""
    table: dict[str, str] = {}
    for kind, phrases in lexicon.items():
        if kind not in KINDS:
            raise ValueError(f"unknown smalltalk kind {kind!r}")
        for phrase in phrases:
            key = normalize(phrase)
            if not key or key in table:
                raise ValueError(
                    f"smalltalk phrase {phrase!r} is empty or duplicated after normalize"
                )
            table[key] = kind
    return table


def realtime_match(tokens: Sequence[Token]) -> GateMatch | None:
    """때 표지와 대상어가 함께 있으면 실시간 질문. 근거는 나온 때 표지 · 대상어 전부."""
    forms = [token.form for token in tokens]
    markers = [
        pattern
        for end, pattern in _find(forms, TIME_MARKERS)
        if not _part_of_other_word(tokens, end)
    ]
    targets = [pattern for _, pattern in _find(forms, TARGETS)]
    if not markers or not targets:
        return None
    return GateMatch("realtime", tuple(" ".join(pattern) for pattern in markers + targets))


def fixed_reply(match: GateMatch, *, dump_date: str) -> str:
    """걸린 종류의 고정 답. dump_date 는 사람이 읽는 꼴('2026년 10월 1일')."""
    return REPLIES[match.kind].format(dump_date=dump_date)


def _find(
    forms: Sequence[str], patterns: Sequence[tuple[str, ...]]
) -> list[tuple[int, tuple[str, ...]]]:
    """형태 열에서 패턴이 나온 (끝 위치 — 다음 토큰 자리, 패턴). 앞에서부터.

    여러 형태소 패턴(시청 + 률)은 한 토큰(시청률)으로 나와도 맞는다 — 사용자 사전에 낱말이 등록되면 경계가 합쳐진다.
    """
    found = []
    for start in range(len(forms)):
        for pattern in patterns:
            if tuple(forms[start : start + len(pattern)]) == pattern:
                found.append((start + len(pattern), pattern))
            elif len(pattern) > 1 and forms[start] == "".join(pattern):
                found.append((start + 1, pattern))
    return found


def _part_of_other_word(tokens: Sequence[Token], end: int) -> bool:
    """때 표지 바로 뒤 토큰이 누적 · 시작점 조사(까지 · 부터)나 접미사(어림 접미사 말고)면 지금을 가리키는 말이 아니다."""
    if end >= len(tokens):
        return False
    after = tokens[end]
    if after.tag == SUFFIX_TAG:
        return after.form not in APPROX_SUFFIXES
    return after.form in RANGE_PARTICLES and after.tag.startswith("J")


class QueryGate:
    """질문을 검색 전에 본다. 잡담이면 종류별, 실시간이면 덤프 날짜를 넣은 고정 답으로 끝낸다.

    예)
        gate = QueryGate(KiwiTokenizer())
        match = gate.check(question)
        if match is not None:
            logger.info("gate %s %s", match.kind, match.matched)
            return fixed_reply(match, dump_date="2026년 10월 1일")
    """

    def __init__(
        self, tokenizer: Tokenizer, lexicon: Mapping[str, Sequence[str]] = SMALLTALK
    ) -> None:
        self._tokenizer = tokenizer
        self._smalltalk = smalltalk_table(lexicon)

    def check(self, question: str) -> GateMatch | None:
        """걸리면 근거와 함께, 아니면 None (검색 → 답변 정책으로). 정규화 뒤 빈 질문은 잡담으로 본다."""
        key = normalize(question)
        if not key:
            return GateMatch("chitchat", ())
        if key in self._smalltalk:
            return GateMatch(self._smalltalk[key], (key,))
        return realtime_match(self._tokenizer.tokenize(question))
