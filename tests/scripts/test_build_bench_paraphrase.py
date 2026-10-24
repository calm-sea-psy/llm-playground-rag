"""scripts/build_bench_paraphrase.py 의 짝짓기 · 줄 바꾸기 · 파일 쓰기. KorQuAD 없이 합성 입력으로 돈다."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts" / "build_bench_paraphrase.py"
_spec = importlib.util.spec_from_file_location("build_bench_paraphrase", _SCRIPT)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _item(item_id, question, kind="answerable"):
    return {
        "id": f"korquad:{item_id}",
        "kind": kind,
        "question": question,
        "answers": ["정답"],
        "gold_title": "문서",
        "context": [
            {
                "rank": rank,
                "chunk_id": f"{item_id}-{rank}",
                "title": "문서",
                "section": "",
                "text": f"본문 {rank}",
                "score": 1.0 - rank / 10,
            }
            for rank in range(1, 6)
        ],
        "strata": {"backlinks": 1, "answer_shape": "other", "answer_rank": 1},
    }


def _line(item_id, question, paraphrase, *, sample="f4-answerable", problems=()):
    return {
        "sample": sample,
        "id": f"korquad:{item_id}" if sample == "f4-answerable" else item_id,
        "title": "문서",
        "question": question,
        "answers": ["정답"],
        "paraphrase": paraphrase,
        "model": "claude-opus-5-5",
        "prompt_version": "paraphrase-v1",
        "attempts": 1,
        "problems": list(problems),
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_paraphrase_row_swaps_question_and_keeps_context():
    source = _item("1-0-0", "원문 질문은?")
    line = _line("1-0-0", "원문 질문은?", "고쳐 쓴 질문은?")

    row = bench.paraphrase_row(source, bench._parse_paraphrase(line))

    assert row["id"] == "paraphrase:1-0-0"
    assert row["kind"] == "paraphrase"
    assert row["question"] == "고쳐 쓴 질문은?"
    assert row["source_id"] == "korquad:1-0-0"
    assert row["original_question"] == "원문 질문은?"
    assert row["context"] == source["context"]  # 다시 검색하지 않고 원문 문항의 5 청크 그대로
    for key in ("answers", "gold_title", "strata"):
        assert row[key] == source[key]
    assert set(row) == set(source) | {"source_id", "original_question"}


def test_paraphrase_row_does_not_share_context_with_source():
    source = _item("1-0-0", "원문 질문은?")
    row = bench.paraphrase_row(
        source, bench._parse_paraphrase(_line("1-0-0", "원문 질문은?", "새"))
    )

    row["context"][0]["text"] = "바뀜"

    assert source["context"][0]["text"] == "본문 1"


def test_pair_excludes_each_reason():
    items = [
        _item("1-0-0", "질문 1"),
        _item("2-0-0", "질문 2"),
        _item("3-0-0", "질문 3", kind="no_doc"),
        _item("4-0-0", "세트를 다시 만들어 바뀐 질문"),
    ]
    lines = [
        _line("1-0-0", "질문 1", "새 질문 1"),
        _line("2-0-0", "질문 2", "새 질문 2", problems=["answer_leak"]),
        _line("9-0-0", "질문 9", "새 질문 9"),
        _line("3-0-0", "질문 3", "새 질문 3"),
        _line("4-0-0", "질문 4", "새 질문 4"),
    ]

    result = bench.pair([bench._parse_paraphrase(line) for line in lines], items)

    assert [row["id"] for row in result.rows] == ["paraphrase:1-0-0"]
    assert [paraphrase.id for paraphrase in result.used] == ["korquad:1-0-0"]
    assert [(exclusion.id, exclusion.reason) for exclusion in result.excluded] == [
        ("korquad:2-0-0", "problems"),
        ("korquad:9-0-0", "missing"),
        ("korquad:3-0-0", "not_answerable"),
        ("korquad:4-0-0", "question_mismatch"),
    ]


def test_pair_rejects_duplicate_paraphrase_ids():
    lines = [_line("1-0-0", "질문 1", "새 질문 1"), _line("1-0-0", "질문 1", "또 새 질문 1")]

    with pytest.raises(ValueError, match="duplicate paraphrase ids"):
        bench.pair([bench._parse_paraphrase(line) for line in lines], [_item("1-0-0", "질문 1")])


def test_build_writes_items_and_meta(tmp_path):
    paraphrase_path = tmp_path / "paraphrase.jsonl"
    _write_jsonl(
        paraphrase_path,
        [
            _line("1-0-0", "질문 1", "새 질문 1"),
            _line("2-0-0", "질문 2", "새 질문 2", problems=["same_as_original"]),
            _line("7-0-0", "F9 질문", "새 F9 질문", sample="f9-random"),
        ],
    )
    repo = tmp_path / "bench-data"
    _write_jsonl(
        repo / "source-set" / "items.jsonl", [_item("1-0-0", "질문 1"), _item("2-0-0", "질문 2")]
    )

    bench.build(paraphrase_path, repo, "source-set", "out-set")

    out_lines = (repo / "out-set" / "items.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in out_lines] == ["paraphrase:1-0-0"]
    meta = json.loads((repo / "out-set" / "meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == "out-set"
    assert meta["source_set"] == "source-set"
    assert meta["paraphrase_models"] == {"claude-opus-5-5": 1}
    assert meta["prompt_versions"] == {"paraphrase-v1": 1}
    assert meta["counts"] == {
        "paraphrase": 1,
        "f4_paraphrases": 2,
        "excluded": {"problems": 1, "missing": 0, "not_answerable": 0, "question_mismatch": 0},
        "other_samples": {"f9-random": 1},
    }
    assert list(meta["counts"]["excluded"]) == list(bench.EXCLUDE_REASONS)
    assert meta["excluded"] == [{"id": "korquad:2-0-0", "reason": "problems"}]
