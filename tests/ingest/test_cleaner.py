from playground_rag.ingest import WikiCleaner
from playground_rag.ingest.cleaner import clean_text


def _clean(wikitext: str):
    return WikiCleaner().clean(wikitext)


def test_clean_splits_sections_with_lead_at_level_zero():
    page = _clean("'''서울'''은 수도이다.\n== 역사 ==\n조선의 수도였다.\n=== 근대 ===\n개항했다.")

    assert [(s.level, s.title, s.text) for s in page.sections] == [
        (0, "", "서울은 수도이다."),
        (2, "역사", "조선의 수도였다."),
        (3, "근대", "개항했다."),
    ]
    assert page.text == "서울은 수도이다.\n\n조선의 수도였다.\n\n개항했다."


def test_clean_drops_tail_sections_with_their_subsections():
    page = _clean(
        "본문.\n== 각주 ==\n각주 내용\n=== 내용주 ===\n하위 내용\n== 같이 보기 ==\n* 링크\n"
    )

    assert [s.title for s in page.sections] == [""]


def test_clean_keeps_section_after_dropped_section():
    page = _clean("본문.\n== 참고 문헌 ==\n책\n== 생애 ==\n태어났다.")

    assert [s.title for s in page.sections] == ["", "생애"]


def test_clean_moves_categories_out_of_text():
    page = _clean("본문이다.\n[[분류:대한민국의 도시|서울]]\n[[Category:Capitals]]")

    assert page.categories == ("대한민국의 도시", "Capitals")
    assert page.text == "본문이다."


def test_clean_removes_media_and_interlanguage_links_but_keeps_link_text():
    page = _clean(
        "[[파일:Seoul.jpg|섬네일|250px|서울 전경]][[서울특별시|서울]]은 [[수도]]이다.[[en:Seoul]]"
    )

    assert page.text == "서울은 수도이다."


def test_clean_removes_refs_and_tables():
    page = _clean('수도이다.<ref name="a">출처 문장</ref>\n{| class="wikitable"\n|-\n| 셀\n|}\n끝.')

    assert page.text == "수도이다.\n\n끝."  # 표가 있던 자리는 문단 경계로 남는다


def test_clean_rewrites_text_bearing_templates():
    page = _clean(
        "카터({{lang|en|Jimmy Carter}}, {{출생일|1924|10|01}})는 {{USS|폼프렛|SS-391}}에서 "
        "복무했다.{{Sfn|Thomas|1978}}"
    )

    assert page.text == "카터(Jimmy Carter, 1924년 10월 1일)는 USS 폼프렛에서 복무했다."


def test_clean_detects_disambiguation_by_template():
    assert _clean("'''소수'''는 다음을 가리킨다.\n{{동음이의}}").is_disambiguation


def test_clean_detects_disambiguation_by_category():
    assert _clean("목록.\n[[분류:동음이의 문서]]").is_disambiguation


def test_clean_text_keeps_letters_and_digits_when_collapsing_spaces():
    # 공백 정규식이 깨져 a · 0 · 2 · b 가 지워졌던 회귀
    nbsp, zero_width_space = chr(0xA0), chr(0x200B)
    text = f"James Earl Carter 1924년 10월 2일 ab{nbsp}b{zero_width_space}c"

    assert clean_text(text) == "James Earl Carter 1924년 10월 2일 ab b c"


def test_clean_text_removes_markup_the_parser_missed():
    text = (
        "__NOTOC__\n"
        "* 목록 항목\n"
        ": 정의 설명\n"
        "! 표 머리\n"
        "|- \n"
        "[[:en:Listen|들어라]] <sub>아래<sub> 문장 [1] .\n"
        '각주<ref name="깨진|저자 = 누구'
    )

    assert clean_text(text) == "목록 항목\n정의 설명\n\n들어라 아래 문장.\n각주"


def test_clean_text_keeps_angle_bracket_titles():
    assert clean_text("소설 <afternoon, a story>(1987)") == "소설 <afternoon, a story>(1987)"
