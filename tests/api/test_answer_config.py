from pathlib import Path

import pytest
import yaml
from playground_llm import Think

from playground_wiki.api.config import ANSWER_CONFIG, load_answer_policy, parse_profiles
from playground_wiki.generation import AnswerPolicy, Thresholds

CONFIG_DIR = Path(__file__).parents[2] / "configs"
ROW = {
    "provider": "ollama",
    "model": "gemma4:12b",
    "params": {"think": "off", "stop": ["\n\n"]},
    "prompts": ["base@v1", "grounded@v1"],
    "timeout_sec": 60,
}


def _write(directory: Path, thresholds: dict, profiles: dict) -> Path:
    text = yaml.safe_dump({"thresholds": thresholds, "profiles": profiles}, allow_unicode=True)
    (directory / ANSWER_CONFIG).write_text(text, encoding="utf-8")
    return directory


def test_repo_answer_yaml_profiles_build_a_policy():
    raw = yaml.safe_load((CONFIG_DIR / ANSWER_CONFIG).read_text(encoding="utf-8"))

    profiles = parse_profiles(raw["profiles"])

    assert profiles["grounded"].spec.think is Think.OFF
    assert profiles["partial"].spec.think is Think.ON
    assert all(profile.prompts[0].startswith("base@") for profile in profiles.values())
    AnswerPolicy(Thresholds(high=1.0, low=0.0), profiles)  # 단계 구성이 맞는지


def test_load_builds_policy_and_turns_yaml_lists_into_tuples(tmp_path: Path):
    config_dir = _write(tmp_path, {"high": 0.99, "low": 0.5}, {"grounded": ROW, "partial": ROW})

    policy = load_answer_policy(config_dir)

    assert policy.thresholds == Thresholds(high=0.99, low=0.5)
    grounded = policy.profiles["grounded"]
    assert grounded.spec.params["stop"] == ("\n\n",)
    assert grounded.prompts == ("base@v1", "grounded@v1")


def test_load_refuses_uncalibrated_thresholds(tmp_path: Path):
    config_dir = _write(tmp_path, {"high": None, "low": None}, {"grounded": ROW, "partial": ROW})

    with pytest.raises(ValueError, match="not calibrated"):
        load_answer_policy(config_dir)


def test_unquoted_off_in_yaml_is_rejected(tmp_path: Path):
    # YAML 은 따옴표 없는 off 를 false 로 읽는다 — ModelSpec 이 bool 을 거부해 바로 드러난다
    row = "{provider: ollama, model: m, params: {think: off}, prompts: [base@v1], timeout_sec: 60}"
    text = f"thresholds: {{high: 0.9, low: 0.5}}\nprofiles:\n  grounded: {row}\n  partial: {row}\n"
    (tmp_path / ANSWER_CONFIG).write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="think must be one of"):
        load_answer_policy(tmp_path)


def test_misspelled_profile_key_is_rejected(tmp_path: Path):
    row = {key: value for key, value in ROW.items() if key != "timeout_sec"} | {"timeout": 60}
    config_dir = _write(tmp_path, {"high": 0.9, "low": 0.5}, {"grounded": row, "partial": ROW})

    with pytest.raises(ValueError, match="profile 'grounded' keys must be"):
        load_answer_policy(config_dir)
