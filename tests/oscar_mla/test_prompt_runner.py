import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[2] / "tools" / "oscar_mla" / "run_calibration_prompts.py"
)
_SPEC = importlib.util.spec_from_file_location("run_calibration_prompts", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_calibration_split = _MODULE.run_calibration_split


def _manifest(path) -> None:
    rows = [
        {"id": "train-0", "split": "train", "tokens": 3, "text": "abc"},
        {"id": "holdout-0", "split": "holdout", "tokens": 2, "text": "de"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prompt_runner_checks_usage_and_writes_reproducible_evidence(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)

    def post_completion(**kwargs):
        return {
            "id": "request-1",
            "usage": {
                "prompt_tokens": len(kwargs["prompt"]),
                "completion_tokens": 1,
            },
            "choices": [{"text": "x"}],
        }

    summary = run_calibration_split(
        manifest_path=manifest,
        split="train",
        expected_tokens=3,
        endpoint="http://127.0.0.1:18080/v1/completions",
        model="test",
        output_dir=tmp_path / "output",
        timeout_seconds=1,
        progress_interval_seconds=600,
        post_completion=post_completion,
    )

    assert summary["rows"] == 1
    assert summary["prompt_tokens"] == 3
    assert summary["completion_tokens"] == 1
    response = json.loads(
        (tmp_path / "output" / "responses.jsonl").read_text(encoding="utf-8")
    )
    assert response["id"] == "train-0"
    assert len(response["response_sha256"]) == 64


def test_prompt_runner_rejects_manifest_and_server_token_mismatches(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)

    with pytest.raises(ValueError, match="manifest token count"):
        run_calibration_split(
            manifest_path=manifest,
            split="train",
            expected_tokens=4,
            endpoint="unused",
            model="test",
            output_dir=tmp_path / "not-created",
            timeout_seconds=1,
        )

    def wrong_usage(**kwargs):
        del kwargs
        return {
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
            }
        }

    with pytest.raises(ValueError, match="prompt token mismatch"):
        run_calibration_split(
            manifest_path=manifest,
            split="train",
            expected_tokens=3,
            endpoint="unused",
            model="test",
            output_dir=tmp_path / "output",
            timeout_seconds=1,
            post_completion=wrong_usage,
        )
