"""덤프 전체에서 고르게 뽑은 정제 전 문서 표본을 parquet 로 저장한다 (정제 규칙 점검용, 한 번만).

넘겨주기를 뺀 본문 이름공간 문서를 덤프 순서로 세어 --stride 번째마다 하나씩 담는다.
덤프 앞쪽에 몰리지 않아 문서 종류 분포가 전체와 비슷하다.

출력:
  data/sample/{source}-sample-{n}.parquet   id · title · wikitext

사용법:
  uv run python scripts/sample_pages.py              # 약 2만 페이지 (본문 약 76만 / 38)
  uv run python scripts/scan_cleaning.py             # 표본을 다시 정제해 패턴을 센다
"""

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from playground_rag.sources import WikiSource
from playground_rag.sources.wiki import REDIRECT_MARKERS

SAMPLE_DIR = Path("data/sample")
STRIDE = 38  # 넘겨주기 뺀 본문 문서 약 76만 / 38 ≈ 2만
PROGRESS_EVERY = 200_000


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["source"]
    source = WikiSource(Path(config["dump_path"]), dump_date=str(config["dump_date"]))
    rows: dict[str, list[str]] = {"id": [], "title": [], "wikitext": []}
    seen = 0
    start = time.perf_counter()
    for page in source.raw_pages():
        if page.wikitext.lstrip().lower().startswith(REDIRECT_MARKERS):
            continue
        if seen % args.stride == 0:
            rows["id"].append(page.id)
            rows["title"].append(page.title)
            rows["wikitext"].append(page.wikitext)
        seen += 1
        if seen % PROGRESS_EVERY == 0:
            print(
                f"{seen:,} pages, {len(rows['id']):,} sampled ({time.perf_counter() - start:.0f}s)"
            )
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out = SAMPLE_DIR / f"{config['name']}-sample-{len(rows['id'])}.parquet"
    pq.write_table(pa.table(rows), out)
    print(
        f"done: {seen:,} pages → {len(rows['id']):,} sampled → {out} ({time.perf_counter() - start:.0f}s)"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/ingest.yaml"))
    parser.add_argument("--stride", type=int, default=STRIDE)
    return parser.parse_args()


if __name__ == "__main__":
    main()
