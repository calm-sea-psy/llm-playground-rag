import bz2
from pathlib import Path

import pytest

from playground_wiki.sources import Redirect, Skipped, WikiSource

_NS = "http://www.mediawiki.org/xml/export-0.11/"
_LONG_TEXT = "서울특별시는 대한민국의 수도이다. " * 20


def _page(page_id: str, title: str, ns: str, text: str, *, redirect: str | None = None) -> str:
    redirect_tag = f'<redirect title="{redirect}" />' if redirect else ""
    return (
        f"<page><title>{title}</title><ns>{ns}</ns><id>{page_id}</id>{redirect_tag}"
        f'<revision><id>1</id><text xml:space="preserve">{text}</text></revision></page>'
    )


@pytest.fixture
def dump(tmp_path: Path) -> Path:
    pages = [
        _page("1", "서울특별시", "0", f"{_LONG_TEXT}\n== 역사 ==\n조선의 수도.\n[[분류:도시]]"),
        _page("2", "서울시", "0", "#넘겨주기 [[서울특별시]]", redirect="서울특별시"),
        _page("3", "토론:서울특별시", "1", _LONG_TEXT),
        _page("4", "짧은 문서", "0", "한 줄."),
        _page("5", "소수", "0", f"{_LONG_TEXT}\n{{{{동음이의}}}}"),
    ]
    xml = (
        f'<mediawiki xmlns="{_NS}"><siteinfo><sitename>위키백과</sitename></siteinfo>'
        + "".join(pages)
        + "</mediawiki>"
    )
    path = tmp_path / "kowiki-test-pages-articles.xml.bz2"
    path.write_bytes(bz2.compress(xml.encode("utf-8")))
    return path


def test_raw_pages_keeps_only_article_namespace(dump: Path):
    source = WikiSource(dump, dump_date="20261001")

    titles = [page.title for page in source.raw_pages()]

    assert titles == ["서울특별시", "서울시", "짧은 문서", "소수"]


def test_parsed_classifies_pages_in_dump_order(dump: Path):
    source = WikiSource(dump, dump_date="20261001", workers=2)

    parsed = list(source.parsed())

    assert [type(page).__name__ for page in parsed] == [
        "Document",
        "Redirect",
        "Skipped",
        "Skipped",
    ]
    assert parsed[1] == Redirect(title="서울시", target="서울특별시")
    assert parsed[2] == Skipped(title="짧은 문서", reason="stub")
    assert parsed[3] == Skipped(title="소수", reason="disambiguation")


def test_documents_fill_meta_and_categories(dump: Path):
    source = WikiSource(dump, dump_date="20261001", workers=1)

    [document] = list(source.documents())

    assert document.id == "1"
    assert document.categories == ("도시",)
    assert [section.title for section in document.sections] == ["", "역사"]
    assert document.meta == {
        "source": "kowiki",
        "url": "https://ko.wikipedia.org/wiki/%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C",
        "dump_date": "20261001",
    }
