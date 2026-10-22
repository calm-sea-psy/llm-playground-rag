"""정제 규칙 점검: 표본(scripts/sample_pages.py)을 지금 규칙으로 다시 정제해 남은 마크업 · 구멍을 센다.

첫 줄 구멍(`하지원(, 1978년 …)` 처럼 괄호 안 틀이 지워진 자리)은 원문 첫 줄 괄호에 있던 틀 이름을 함께 세어
어느 틀을 살려야 할지 보여 준다. 예시는 data/sample/ 아래 텍스트로 남긴다 (원문이라 커밋하지 않는다).

사용법:
  uv run python scripts/scan_cleaning.py
  uv run python scripts/scan_cleaning.py --examples 50
"""

import argparse
import re
import time
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import mwparserfromhell
import pyarrow.parquet as pq
from playground_corpus import Document

from playground_rag.ingest import WikiCleaner
from playground_rag.sources import RawPage, parse_page

SAMPLE_DIR = Path("data/sample")
MIN_CHARS = 200  # WikiSource 기본값과 같게
LEAD_CHARS = 300  # 첫 줄 구멍을 찾는 범위 (첫 문단 앞부분)
RAW_LEAD_CHARS = 800  # 원문에서 굵은 제목 뒤로 틀 이름을 모으는 범위
_LEAD_HOLE = re.compile(r"[(（]\s*[,;、]")
# 정제 뒤 남으면 안 되는 흔적. 이름 → 정규식
PATTERNS = {
    "(, 구멍": _LEAD_HOLE,
    "틀 {{ }}": re.compile(r"\{\{|\}\}"),
    "링크 [[ ]]": re.compile(r"\[\[|\]\]"),
    "세로줄 |": re.compile(r"\|"),
    "따옴표 ''": re.compile(r"''"),
    "URL": re.compile(r"https?://"),
    "파일 · 섬네일": re.compile(r"섬네일|thumb|파일:|file:", re.IGNORECASE),
    "HTML 태그": re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[\s>/]"),
}


def main() -> None:
    args = _parse_args()
    sample = args.sample or _latest_sample()
    table = pq.read_table(sample).to_pylist()
    pages = [RawPage(id=row["id"], title=row["title"], wikitext=row["wikitext"]) for row in table]
    raw_by_id = {page.id: page.wikitext for page in pages}
    start = time.perf_counter()
    process = partial(parse_page, cleaner=WikiCleaner(), dump_date="", min_chars=MIN_CHARS)
    with Pool() as pool:
        parsed = pool.map(process, pages, chunksize=32)
    documents = [page for page in parsed if isinstance(page, Document)]
    print(f"pages={len(pages):,} documents={len(documents):,} ({time.perf_counter() - start:.0f}s)")

    for name, pattern in PATTERNS.items():
        hits = sum(1 for doc in documents if pattern.search(doc.text))
        print(f"  {name}: {hits:,} ({hits / len(documents):.2%})")

    holes = [doc for doc in documents if _LEAD_HOLE.search(doc.text[:LEAD_CHARS])]
    print(f"lead (, holes: {len(holes):,} ({len(holes) / len(documents):.2%})")
    names: Counter[str] = Counter()
    lines = []
    for doc in holes:
        lead_names = _lead_template_names(raw_by_id[doc.id])
        names.update(set(lead_names))
        if len(lines) < args.examples:
            match = _LEAD_HOLE.search(doc.text)
            at = match.start() if match else 0
            raw = _raw_lead(raw_by_id[doc.id])
            lines.append(
                f"## {doc.title}\n정제: {doc.text[max(0, at - 40) : at + 60]!r}\n"
                f"틀: {lead_names}\n원문: {raw[:300]!r}\n"
            )
    print("lead templates in holes (docs):")
    for name, count in names.most_common(args.top):
        print(f"  {count:5d} {name}")
    out = sample.with_name("scan-lead-holes.txt")
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"examples: {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=Path, help="기본: data/sample/ 의 가장 최근 표본")
    parser.add_argument("--examples", type=int, default=30)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def _latest_sample() -> Path:
    samples = sorted(SAMPLE_DIR.glob("*-sample-*.parquet"), key=lambda path: path.stat().st_mtime)
    if not samples:
        raise SystemExit("표본이 없다: uv run python scripts/sample_pages.py")
    return samples[-1]


def _raw_lead(wikitext: str) -> str:
    """원문에서 첫 굵은 제목(''') 부터 RAW_LEAD_CHARS 글자. 없으면 앞부분."""
    at = wikitext.find("'''")
    return wikitext[max(at, 0) : max(at, 0) + RAW_LEAD_CHARS]


def _lead_template_names(wikitext: str) -> list[str]:
    """첫 줄 범위에 있는 틀 이름 (안쪽 틀 포함, 소문자)."""
    code = mwparserfromhell.parse(_raw_lead(wikitext))
    return [str(t.name).strip().lower() for t in code.filter_templates(recursive=True)]


if __name__ == "__main__":
    main()
