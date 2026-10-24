from playground_wiki.evaluation.wikidata import (
    GREGORIAN,
    Fact,
    fact_question,
    facts,
    format_time,
    josa,
    kowiki_titles,
    label_fact,
    render,
    value_ids,
    value_names,
)


def _item(pid: str, qid: str, *, rank: str = "normal") -> dict:
    return {
        "rank": rank,
        "mainsnak": {
            "snaktype": "value",
            "property": pid,
            "datavalue": {"type": "wikibase-entityid", "value": {"id": qid}},
        },
    }


def _time(pid: str, time: str, precision: int, calendar: str = GREGORIAN) -> dict:
    return {
        "rank": "normal",
        "mainsnak": {
            "snaktype": "value",
            "property": pid,
            "datavalue": {
                "type": "time",
                "value": {"time": time, "precision": precision, "calendarmodel": calendar},
            },
        },
    }


PERSON = {
    "sitelinks": {"kowiki": {"title": "홍길동 (축구 선수)"}},
    "claims": {
        "P31": [_item("P31", "Q5")],
        "P569": [_time("P569", "+1950-03-02T00:00:00Z", 11)],
        "P19": [_item("P19", "Q8684"), _item("P19", "Q999", rank="deprecated")],
        "P26": [{"rank": "normal", "mainsnak": {"snaktype": "somevalue", "property": "P26"}}],
        "P571": [_time("P571", "+1999-00-00T00:00:00Z", 9)],  # 사람에게는 묻지 않는 속성
    },
}


def test_josa_follows_the_last_sound_of_the_word():
    assert josa("서울", "은") == "은"
    assert josa("부산", "이") == "이"
    assert josa("대구", "은") == "는"
    assert josa("홍길동 (축구 선수)", "이") == "가"  # 괄호 안 마지막 글자 "수"
    assert josa("KBS", "을") == "를"
    assert josa("2010", "과") == "과"  # 이천십 — 마지막 숫자 0 은 받침 있음
    assert render("{주제}{은} 언제 태어났나?", "대구") == "대구는 언제 태어났나?"


def test_format_time_keeps_precision_and_skips_julian():
    assert format_time(
        _time("P", "+1950-03-02T00:00:00Z", 11)["mainsnak"]["datavalue"]["value"]
    ) == ("1950년 3월 2일", False)
    assert format_time(
        _time("P", "+1950-03-00T00:00:00Z", 10)["mainsnak"]["datavalue"]["value"]
    ) == ("1950년 3월", False)
    assert format_time(
        _time("P", "+1950-00-00T00:00:00Z", 9)["mainsnak"]["datavalue"]["value"]
    ) == ("1950년", True)
    julian = _time("P", "+1450-01-01T00:00:00Z", 11, "http://www.wikidata.org/entity/Q1985786")
    assert format_time(julian["mainsnak"]["datavalue"]["value"]) is None
    assert (
        format_time(_time("P", "+1900-00-00T00:00:00Z", 7)["mainsnak"]["datavalue"]["value"])
        is None
    )


def test_value_names_prefer_kowiki_title_then_label_and_aliases():
    seoul = {
        "sitelinks": {"kowiki": {"title": "서울특별시"}},
        "labels": {"ko": {"value": "서울"}},
        "aliases": {"ko": [{"value": "한성"}, {"value": "서울"}]},
    }
    town = {"sitelinks": {"kowiki": {"title": "광산 (마을)"}}}

    assert value_names(seoul) == ("서울특별시", "서울", "한성")
    assert value_names(town) == ("광산 (마을)", "광산")
    assert (
        value_names({"labels": {"en": {"value": "Nowhere"}}}) == ()
    )  # 한국어 이름이 없으면 그 속성은 건너뛴다


def test_facts_take_template_properties_by_subject_kind():
    names = {"Q8684": ("서울특별시", "서울")}

    found = {fact.pid: fact for fact in facts("Q1", "홍길동 (축구 선수)", PERSON, names)}

    assert set(found) == {
        "P569",
        "P19",
    }  # 배우자는 값 없음(somevalue), P571 은 사람에게 묻지 않는다
    assert found["P569"].answers == ("1950년 3월 2일",)
    assert found["P19"].answers == ("서울특별시", "서울")  # deprecated 값은 뺀다
    assert found["P19"].strict  # 지명 — 같은 문장 조건
    assert not found["P569"].strict
    assert value_ids(PERSON) == {"Q8684"}  # 템플릿 속성의 값만, deprecated 는 빼고


def test_facts_skip_property_whose_answer_is_in_the_title():
    village = {"claims": {"P131": [_item("P131", "Q10")]}}
    names = {"Q10": ("도마타군",)}

    assert facts("Q2", "가미카모촌 (오카야마현 도마타군)", village, names) == []
    assert facts("Q3", "가미카모촌", village, names)[0].answers == ("도마타군",)


def test_kowiki_titles_skip_missing_pages():
    response = {
        "entities": {"Q1": PERSON, "-1": {"site": "kowiki", "title": "없는 문서", "missing": ""}}
    }

    assert kowiki_titles(response) == {"Q1": "홍길동 (축구 선수)"}


def _question(pid: str, answers: tuple[str, ...], *, strict: bool):
    return fact_question(Fact("Q1", "홍길동", pid, answers, strict))


def test_label_answerable_needs_answer_in_subject_chunk():
    question = _question("P569", ("1950년 3월 2일",), strict=False)

    hits = [
        ("다른 문서", "1950년 3월 2일에 열린 대회"),
        ("홍길동", "홍길동은 1950년 3월 2일 서울에서 태어났다."),
    ]
    elsewhere = [("다른 문서", "1950년 3월 2일에 열린 대회"), ("홍길동", "홍길동은 축구 선수이다.")]

    assert label_fact(question, hits).label == "answerable"
    assert label_fact(question, hits).rank == 2
    assert label_fact(question, elsewhere).reason == "answer_in_other_document"


def test_label_strict_needs_property_word_in_the_same_sentence():
    question = _question("P19", ("서울특별시", "서울"), strict=True)

    same = [("홍길동", "홍길동은 서울에서 태어났다. 이후 부산으로 이사했다.")]
    apart = [("홍길동", "홍길동은 1950년에 태어났다. 서울 FC 에서 뛰었다.")]

    assert label_fact(question, same).label == "answerable"
    assert label_fact(question, apart).reason == "answer_not_with_property"


def test_label_no_fact_only_when_subject_retrieved_without_answer_or_property_words():
    question = _question("P26", ("김영희",), strict=False)

    absent = [("홍길동", "홍길동은 축구 선수이다."), ("축구", "공을 차는 운동")]
    mentioned = [("홍길동", "홍길동은 1980년에 결혼했다.")]  # 다른 표기로 적힌 배우자일 수 있다
    not_retrieved = [("축구", "공을 차는 운동")]

    assert label_fact(question, absent).label == "no_fact"
    assert label_fact(question, absent).rank == 1
    assert label_fact(question, mentioned).reason == "property_mentioned"
    assert label_fact(question, not_retrieved).reason == "subject_not_retrieved"
