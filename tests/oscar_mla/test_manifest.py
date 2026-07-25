import hashlib
import json

import pytest

from vllm.model_executor.layers.quantization.oscar_mla.manifest import (
    CalibrationSource,
    build_calibration_manifest,
)


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


def _write_jsonl(path, rows) -> str:
    data = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode()).hexdigest()


def _source(tmp_path, name, category, rows) -> CalibrationSource:
    path = tmp_path / f"{name}.jsonl"
    sha256 = _write_jsonl(path, rows)
    return CalibrationSource(
        name=name,
        category=category,
        path=str(path),
        uri=f"test://{name}",
        revision="a" * 40,
        file_sha256=sha256,
        text_fields=("text",),
        id_field="id",
    )


def _rows(prefix, count=40):
    return [
        {"id": f"{prefix}-{index}", "text": f"{prefix} text {index} " + "x" * 30}
        for index in range(count)
    ]


def test_manifest_is_deterministic_disjoint_and_meets_quotas(tmp_path) -> None:
    official_path = tmp_path / "official.jsonl"
    official_text = "general text 0 " + "x" * 30
    _write_jsonl(official_path, [{"prompt": official_text}])
    sources = [
        _source(tmp_path, "general", "general", _rows("general")),
        _source(tmp_path, "math", "math", _rows("math")),
        _source(tmp_path, "code", "code", _rows("code")),
    ]
    quotas = {
        "train": {"general": 120, "math": 120, "code": 120},
        "holdout": {"general": 40, "math": 40, "code": 40},
    }
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    kwargs = {
        "sources": sources,
        "official_manifest_path": official_path,
        "quotas": quotas,
        "tokenizer": CharacterTokenizer(),
        "tokenizer_sha256": "b" * 64,
        "seed": 137,
        "holdout_fraction": 0.25,
        "max_chunk_tokens": 64,
    }

    first = build_calibration_manifest(output_path=first_path, **kwargs)
    second = build_calibration_manifest(output_path=second_path, **kwargs)

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.token_counts == second.token_counts == quotas
    assert first.excluded_official_exact == 1
    entries = [
        json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()
    ]
    source_splits = {}
    for entry in entries:
        source_splits.setdefault(
            (entry["source_name"], entry["source_id"]),
            set(),
        ).add(entry["split"])
        assert entry["tokens"] <= 64
        assert (
            hashlib.sha256(" ".join(entry["text"].split()).encode()).hexdigest()
            == entry["text_sha256"]
        )
    assert all(len(splits) == 1 for splits in source_splits.values())


def test_manifest_rejects_source_hash_mismatch(tmp_path) -> None:
    official_path = tmp_path / "official.jsonl"
    _write_jsonl(official_path, [{"prompt": "official"}])
    source = _source(tmp_path, "general", "general", _rows("general"))
    wrong_source = CalibrationSource(**{**source.__dict__, "file_sha256": "f" * 64})

    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        build_calibration_manifest(
            sources=[wrong_source],
            official_manifest_path=official_path,
            quotas={
                "train": {"general": 20},
                "holdout": {"general": 20},
            },
            tokenizer=CharacterTokenizer(),
            tokenizer_sha256="b" * 64,
            seed=139,
            holdout_fraction=0.25,
            max_chunk_tokens=64,
            output_path=tmp_path / "manifest.jsonl",
        )


def test_manifest_rejects_unmet_quota(tmp_path) -> None:
    official_path = tmp_path / "official.jsonl"
    _write_jsonl(official_path, [{"prompt": "official"}])
    source = _source(
        tmp_path,
        "general",
        "general",
        [{"id": "only", "text": "short"}],
    )

    with pytest.raises(ValueError, match="quota unmet"):
        build_calibration_manifest(
            sources=[source],
            official_manifest_path=official_path,
            quotas={
                "train": {"general": 1000},
                "holdout": {"general": 1000},
            },
            tokenizer=CharacterTokenizer(),
            tokenizer_sha256="b" * 64,
            seed=149,
            holdout_fraction=0.5,
            max_chunk_tokens=64,
            output_path=tmp_path / "manifest.jsonl",
        )
