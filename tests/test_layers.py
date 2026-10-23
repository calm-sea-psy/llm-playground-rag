"""하위 패키지 사이 의존 방향 검사. 나중에 라이브러리(core · store · retrieval · eval)를 모듈 레포로 뗄 때
폴더만 옮기면 되도록, 라이브러리 쪽이 제품 쪽(api · sources · ingest)과 설정 · 웹 의존을 가져오지 못하게 막는다.

    core ← store · retrieval · eval   (라이브러리)
    core ← ingest ← sources            (위키 전용)
    전부 ← api                          (제품)
"""

import ast
from pathlib import Path

import pytest

PACKAGE = "playground_rag"
SRC = Path(__file__).parents[1] / "src" / PACKAGE

# 하위 패키지 → 가져와도 되는 다른 하위 패키지
ALLOWED_INTERNAL = {
    "core": set(),
    "store": {"core"},
    "retrieval": {"core"},  # 저장소는 Protocol 로 받는다 — store 구현을 직접 가져오지 않는다
    "eval": {"core"},
    "ingest": {"core"},
    "sources": {"core", "ingest"},
    "api": {"core", "store", "retrieval", "eval", "ingest", "sources"},
}
LIBRARY = ("core", "store", "retrieval", "eval")
# 라이브러리 쪽에서 쓰면 안 되는 외부 패키지: 웹 · 설정 파일은 제품 쪽 일, DB 클라이언트는 store 만
# 어디서도 쓰면 안 되는 외부 패키지. api/** 는 pydantic 때문에 ruff TID251 을 꺼 두어 ruff 로는 못 잡는다
BANNED_EVERYWHERE = frozenset({"ollama", "openai"})  # LLM 은 playground_llm 으로만 부른다
BANNED_EXTERNAL = {
    "core": {"fastapi", "pydantic", "yaml", "psycopg", "qdrant_client", "faiss", "torch"},
    "store": {"fastapi", "pydantic", "yaml"},
    "retrieval": {"fastapi", "pydantic", "yaml", "psycopg", "qdrant_client"},
    "eval": {"fastapi", "pydantic", "yaml", "psycopg", "qdrant_client"},
}


def _imports(path: Path) -> list[tuple[int, str]]:
    """파일의 모든 import (함수 안 포함) 를 (줄, 모듈 이름) 으로."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
    return found


def _violations(package: str, src: Path = SRC) -> list[tuple[str, int, str]]:
    """규칙을 어긴 import 를 (패키지 안 파일 경로, 줄, 모듈 이름) 으로."""
    allowed = ALLOWED_INTERNAL[package] | {package}
    banned = BANNED_EXTERNAL.get(package, set()) | BANNED_EVERYWHERE
    problems = []
    for path in sorted((src / package).rglob("*.py")):
        for line, module in _imports(path):
            parts = module.split(".")
            internal_breach = parts[0] == PACKAGE and len(parts) > 1 and parts[1] not in allowed
            if internal_breach or parts[0] in banned:
                problems.append((path.relative_to(src).as_posix(), line, module))
    return problems


def test_every_subpackage_has_a_rule():
    subpackages = {path.parent.name for path in SRC.glob("*/__init__.py")}

    assert subpackages == set(ALLOWED_INTERNAL)


@pytest.mark.parametrize("package", sorted(ALLOWED_INTERNAL))
def test_subpackage_imports_follow_layers(package: str):
    assert _violations(package) == []


def test_library_does_not_import_product_side():
    product = set(ALLOWED_INTERNAL) - set(LIBRARY)

    for package in LIBRARY:
        assert not ALLOWED_INTERNAL[package] & product, package


def test_checker_catches_imports_inside_functions(tmp_path: Path):
    (tmp_path / "retrieval").mkdir()
    (tmp_path / "retrieval" / "bad.py").write_text(
        "from playground_rag.core import Hit\n"
        "def f():\n"
        "    from playground_rag.api import app\n"
        "    import yaml\n",
        encoding="utf-8",
    )

    assert _violations("retrieval", tmp_path) == [
        ("retrieval/bad.py", 3, "playground_rag.api"),
        ("retrieval/bad.py", 4, "yaml"),
    ]


def test_checker_bans_llm_clients_even_in_api(tmp_path: Path):
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "bad.py").write_text(
        "import pydantic\nfrom ollama import Client\n", encoding="utf-8"
    )

    assert _violations("api", tmp_path) == [("api/bad.py", 2, "ollama")]
