"""F4 롱테일 · 답 없음 B 문항을 Wikidata 로 만든다 — 덜 알려진 문서의 (속성 · 참값)을 템플릿 질문으로.

KorQuAD dev 는 2018년부터 공개돼 후보 모델이 문항을 봤을 수 있다. 그래서 역링크가 적은 문서(롱테일)에서 문항을 따로 만든다.
단계별로 돌린다 — 단계마다 결과를 data/bench/ 에 남기고, Wikidata 응답은 항목마다 캐시해 다시 묻지 않는다.
  candidates  코퍼스 문서 중 역링크 LONGTAIL_MAX 이하 · KorQuAD 문서가 아닌 것에서 seed 로 CANDIDATES 개
              → data/bench/longtail-candidates-v1.jsonl (역링크: scripts/count_backlinks.py)
  fetch       후보 제목으로 wbgetentities(sites=kowiki) → 주제 항목, 템플릿 속성이 가리키는 값 항목의 한국어 표기 →
              data/bench/wikidata/entities.jsonl · values.jsonl (네트워크만 쓴다 — GPU 를 쓰는 벤치와 같이 돌려도 된다)
  questions   속성 템플릿(playground_wiki.evaluation.wikidata)으로 질문과 참값 표기 → data/bench/longtail-questions-v1.jsonl
              사람 항목 중 사망일이 없는 이(생존 인물로 본다)에게는 민감 질문 하나 (참값 없음 — null 만 정답)
  search      /ask 와 같은 검색 상위 5 → data/bench/longtail-search-v1.jsonl (GPU — 벤치가 끝난 뒤에)
  label       답 있음 · 답 없음 B · 버림(이유)을 붙여 → data/bench/longtail-labeled-v1.jsonl. 버린 것도 검색 결과와 함께
              남긴다 (선택 편향 없는 표본이 필요할 때)
  pilot       라벨마다 seed 로 몇 개씩 보여 준다 — 어색한 질문 · 너무 빡빡한 조건 · 빠진 표기를 눈으로 거른다

Wikidata 는 CC0. 요청 규칙: 연락처가 담긴 User-Agent, 요청 하나씩 REQUEST_INTERVAL 초 간격, 429 면 물러난다.
data/ 는 git 제외.

사용법:
  uv run python scripts/build_longtail_set.py candidates
  uv run python scripts/build_longtail_set.py fetch
  uv run python scripts/build_longtail_set.py questions
  uv run python scripts/build_longtail_set.py search
  uv run python scripts/build_longtail_set.py label
  uv run python scripts/build_longtail_set.py pilot
"""

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from playground_evalkit import load_korquad, normalize_title
from playground_retrieval.search import SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.config import ApiConfig
from playground_wiki.evaluation.wikidata import (
    PROPERTIES,
    Fact,
    FactQuestion,
    fact_question,
    facts,
    is_human,
    kowiki_titles,
    label_fact,
    sensitive_question,
    value_ids,
    value_names,
)

SUBSET = Path("data/parsed/kowiki-subset-10000")  # subset.json: KorQuAD dev 경로 · 제목 보정표 (D4)
PARSED = Path("data/parsed/kowiki")
BACKLINKS = Path("data/bench/backlinks-20261001.parquet")
CANDIDATES_PATH = Path("data/bench/longtail-candidates-v1.jsonl")
WIKIDATA = Path("data/bench/wikidata")
ENTITIES_PATH = WIKIDATA / "entities.jsonl"
VALUES_PATH = WIKIDATA / "values.jsonl"
QUESTIONS_PATH = Path("data/bench/longtail-questions-v1.jsonl")
SEARCH_PATH = Path("data/bench/longtail-search-v1.jsonl")
LABELED_PATH = Path("data/bench/longtail-labeled-v1.jsonl")
SEED = 0
LONGTAIL_MAX = 2  # 코퍼스 문서 역링크 하위 약 25% (p25 = 2, 2026-10-24 집계)
CANDIDATES = 6000  # 1단계 롱테일 100 · B 100 + 2단계 +300 · +200 이 라벨 뒤에 남을 만큼 (수율은 파일럿으로 확인)
TOP_K = 5
API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "llm-playground-wiki-bench/0.1 (https://github.com/calm-sea-psy; LLM benchmark question set) httpx"
BATCH = 50  # wbgetentities 가 한 번에 받는 제목 · id 수
REQUEST_INTERVAL = 1.0  # Wikimedia API 규칙 — 요청은 하나씩, 간격을 둔다
MAX_RETRIES = 5
PILOT_PER_LABEL = 10


def main() -> None:
    args = _parse_args()
    steps = {
        "candidates": candidates,
        "fetch": fetch,
        "questions": questions,
        "search": search,
        "label": label,
        "pilot": pilot,
    }
    steps[args.step]()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "step", choices=["candidates", "fetch", "questions", "search", "label", "pilot"]
    )
    return parser.parse_args()


def candidates() -> None:
    backlinks = pq.read_table(BACKLINKS).to_pydict()
    corpus = set()
    for path in sorted(PARSED.glob("docs-*.parquet")):
        corpus.update(pq.read_table(path, columns=["id"])["id"].to_pylist())
    summary = json.loads((SUBSET / "subset.json").read_text(encoding="utf-8"))
    korquad = {
        item.title
        for item in load_korquad(
            Path(summary["korquad_dev"]), title_aliases=summary["korquad_title_map"]
        )
    }
    pool = [
        {"page_id": page_id, "title": title, "backlinks": count}
        for page_id, title, count in zip(
            backlinks["page_id"], backlinks["title"], backlinks["backlinks"], strict=True
        )
        if page_id in corpus and count <= LONGTAIL_MAX and normalize_title(title) not in korquad
    ]
    picked = random.Random(SEED).sample(pool, min(CANDIDATES, len(pool)))
    _write_jsonl(CANDIDATES_PATH, sorted(picked, key=lambda row: int(row["page_id"])))
    print(
        f"코퍼스 문서 {len(corpus):,} 중 역링크 {LONGTAIL_MAX} 이하 · KorQuAD 아님 {len(pool):,} → {len(picked):,}\n"
        f"저장: {CANDIDATES_PATH}"
    )


def fetch() -> None:
    titles = [row["title"] for row in _read_jsonl(CANDIDATES_PATH)]
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as client:
        done = {row["title"] for row in _read_jsonl_if(ENTITIES_PATH)}
        todo = [title for title in titles if title not in done]
        print(f"주제 항목: 후보 {len(titles):,} · 남은 {len(todo):,}", flush=True)
        for index, batch in enumerate(_batches(todo), start=1):
            response = _get(
                client, sites="kowiki", titles="|".join(batch), props="claims|sitelinks"
            )
            found = {title: qid for qid, title in kowiki_titles(response).items()}
            entities = response.get("entities", {})
            for title in batch:
                qid = found.get(title)
                _append_jsonl(
                    ENTITIES_PATH,
                    {"title": title, "qid": qid, "entity": entities.get(qid) if qid else None},
                )
            if index % 10 == 0:
                print(f"  {index * BATCH:,}/{len(todo):,}", flush=True)

        needed = set()
        for row in _read_jsonl_if(ENTITIES_PATH):
            if row["entity"]:
                needed |= value_ids(row["entity"])
        have = {row["qid"] for row in _read_jsonl_if(VALUES_PATH)}
        todo_ids = sorted(needed - have)
        print(f"값 항목: {len(needed):,} · 남은 {len(todo_ids):,}", flush=True)
        for index, batch in enumerate(_batches(todo_ids), start=1):
            response = _get(
                client, ids="|".join(batch), props="labels|aliases|sitelinks", languages="ko"
            )
            for qid in batch:
                _append_jsonl(
                    VALUES_PATH, {"qid": qid, "entity": response.get("entities", {}).get(qid)}
                )
            if index % 20 == 0:
                print(f"  {index * BATCH:,}/{len(todo_ids):,}", flush=True)
    print(f"저장: {ENTITIES_PATH} · {VALUES_PATH}")


def questions() -> None:
    backlinks = {row["title"]: row["backlinks"] for row in _read_jsonl(CANDIDATES_PATH)}
    names = {
        row["qid"]: value_names(row["entity"]) for row in _read_jsonl(VALUES_PATH) if row["entity"]
    }
    rows, per_property, humans = [], Counter(), 0
    entities = [row for row in _read_jsonl(ENTITIES_PATH) if row["entity"]]
    for row in entities:
        qid, subject, entity = row["qid"], row["title"], row["entity"]
        human = is_human(entity)
        humans += human
        for fact in facts(qid, subject, entity, names):
            question = fact_question(fact)
            rows.append(_question_row(question, backlinks[subject]))
            per_property[question.template.name] += 1
        if human and "P570" not in entity.get("claims", {}):
            question_id, text = sensitive_question(qid, subject)
            rows.append(
                {
                    "id": question_id,
                    "kind": "sensitive",
                    "qid": qid,
                    "subject": subject,
                    "question": text,
                    "answers": [],
                    "backlinks": backlinks[subject],
                }
            )
    _write_jsonl(QUESTIONS_PATH, rows)
    print(
        f"후보 {len(backlinks):,} 중 Wikidata 항목 {len(entities):,} (사람 {humans:,}) → 질문 {len(rows):,} "
        f"(민감 {sum(row['kind'] == 'sensitive' for row in rows):,})"
    )
    print("속성별:", ", ".join(f"{name} {count}" for name, count in per_property.most_common()))
    print(f"저장: {QUESTIONS_PATH}")


def search() -> None:
    rows = _read_jsonl(QUESTIONS_PATH)
    done = {row["id"] for row in _read_jsonl_if(SEARCH_PATH)}
    todo = [row for row in rows if row["id"] not in done]
    config = ApiConfig.from_env()
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg:
        service = SearchService.build(config.search, pg, QdrantStore.from_env())
        print(f"검색 준비 {time.perf_counter() - start:.0f}s · 남은 {len(todo):,}", flush=True)
        asyncio.run(_search_all(service, todo))
    print(f"저장: {SEARCH_PATH} ({time.perf_counter() - start:.0f}s)")


async def _search_all(service: SearchService, rows: Sequence[dict]) -> None:
    # 서비스가 질의를 한 번에 하나씩 돌리므로 순서대로 기다린다. 한 줄씩 이어 써서 멈춰도 이어 간다
    for index, row in enumerate(rows, start=1):
        result = await service.search(row["question"], k=TOP_K)
        hits = [
            {
                "chunk_id": hit.chunk.id,
                "title": hit.chunk.title,
                "section": hit.chunk.section,
                "text": hit.chunk.text,
                "score": round(hit.score, 6),
            }
            for hit in result.hits
        ]
        _append_jsonl(SEARCH_PATH, {"id": row["id"], "hits": hits})
        if index % 200 == 0:
            print(f"  {index:,}/{len(rows):,}", flush=True)


def label() -> None:
    rows = {row["id"]: row for row in _read_jsonl(QUESTIONS_PATH)}
    templates = {template.pid: template for template in PROPERTIES}
    labeled, counts = [], Counter()
    for searched in _read_jsonl(SEARCH_PATH):
        row = rows[searched["id"]]
        hits = [(hit["title"], hit["text"]) for hit in searched["hits"]]
        if row["kind"] == "sensitive":
            subject = normalize_title(row["subject"])
            ranks = [
                rank for rank, (title, _) in enumerate(hits, 1) if normalize_title(title) == subject
            ]
            result = (
                ("sensitive", "subject_retrieved", ranks[0])
                if ranks
                else ("dropped", "subject_not_retrieved", None)
            )
        else:
            fact = Fact(
                row["qid"], row["subject"], row["pid"], tuple(row["answers"]), row["strict"]
            )
            question = FactQuestion(row["id"], row["question"], fact, templates[row["pid"]])
            found = label_fact(question, hits)
            result = (found.label, found.reason, found.rank)
        labeled.append(
            {
                **row,
                "label": result[0],
                "reason": result[1],
                "rank": result[2],
                "hits": searched["hits"],
            }
        )
        counts[(row.get("property", "민감"), result[0], result[1])] += 1
    _write_jsonl(LABELED_PATH, labeled)
    totals = Counter(row["label"] for row in labeled)
    print(
        f"라벨 {len(labeled):,}: "
        + ", ".join(f"{name} {count}" for name, count in totals.most_common())
    )
    reasons = Counter((row["label"], row["reason"]) for row in labeled)
    print(
        "이유:",
        ", ".join(f"{name}/{reason} {count}" for (name, reason), count in reasons.most_common()),
    )
    by_property = Counter()
    for (prop, name, _), count in counts.items():
        by_property[(prop, name)] += count
    for prop in dict.fromkeys(prop for prop, _ in by_property):
        cells = ", ".join(
            f"{name} {by_property[(prop, name)]}" for p, name in by_property if p == prop
        )
        print(f"  {prop}: {cells}")
    print(f"저장: {LABELED_PATH}")


def pilot() -> None:
    rows = _read_jsonl(LABELED_PATH)
    rng = random.Random(SEED)
    for name in ("answerable", "no_fact", "sensitive", "dropped"):
        group = [row for row in rows if row["label"] == name]
        print(f"\n===== {name} ({len(group)})")
        for row in rng.sample(group, min(PILOT_PER_LABEL, len(group))):
            print(
                f"- [{row.get('property', '민감')}] {row['question']}  → {row['answers'][:3]}  ({row['reason']}, rank {row['rank']})"
            )
            if row["rank"]:
                text = row["hits"][row["rank"] - 1]["text"].replace("\n", " ")
                print(f"    {text[:160]}")


def _question_row(question: FactQuestion, backlinks: int) -> dict:
    fact, template = question.fact, question.template
    return {
        "id": question.id,
        "kind": "fact",
        "qid": fact.qid,
        "subject": fact.subject,
        "pid": fact.pid,
        "property": template.name,
        "question": question.question,
        "answers": list(fact.answers),
        "strict": fact.strict,
        "shape": template.shape,
        "backlinks": backlinks,
    }


def _get(client: httpx.Client, **params: str) -> dict:
    """wbgetentities 한 번. 429 · 5xx 는 기다렸다 다시 (MAX_RETRIES 번).

    maxlag 는 걸지 않는다 — 편집 봇이 서버가 밀릴 때 물러나라는 장치인데, 읽기만 하는 이 질의에서도
    검색 서비스(WDQS) 지연(몇 초 ~ 몇 시간)에 걸려 전부 거절됐다 (2026-10-24). 요청은 하나씩 간격을 둔다.
    """
    # sitefilter: 다른 언어 위키 sitelink 는 받지 않는다 (응답이 작아진다)
    query = {"action": "wbgetentities", "format": "json", "sitefilter": "kowiki", **params}
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(REQUEST_INTERVAL)
        response = client.get(API, params=query)
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS or response.is_server_error:
            time.sleep(float(response.headers.get("Retry-After", 10 * attempt)))
            continue
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(f"wbgetentities error: {body['error']}")
        return body
    raise RuntimeError(f"wbgetentities failed after {MAX_RETRIES} attempts")


def _batches(values: Sequence[str]) -> Iterator[list[str]]:
    for start in range(0, len(values), BATCH):
        yield list(values[start : start + BATCH])


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_jsonl_if(path: Path) -> list[dict]:
    return _read_jsonl(path) if path.exists() else []


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
