#!/usr/bin/env python3
"""Send one deterministic calibration split through an OpenAI-compatible server."""

import argparse
import hashlib
import json
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _load_split(
    manifest_path: Path,
    split: str,
    expected_tokens: int,
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError(f"manifest contains no rows for split {split!r}")
    actual_tokens = sum(int(row["tokens"]) for row in selected)
    if actual_tokens != expected_tokens:
        raise ValueError(
            f"manifest token count for {split} is {actual_tokens}, "
            f"expected {expected_tokens}"
        )
    ids = [row["id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest split {split!r} contains duplicate IDs")
    return selected


def _post_completion(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "seed": 20260725,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def run_calibration_split(
    *,
    manifest_path: Path,
    split: str,
    expected_tokens: int,
    endpoint: str,
    model: str,
    output_dir: Path,
    timeout_seconds: float,
    progress_interval_seconds: float = 600,
    post_completion: Callable[..., dict[str, Any]] = _post_completion,
) -> dict[str, Any]:
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")
    rows = _load_split(manifest_path, split, expected_tokens)
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    responses_path = output_dir / "responses.jsonl"
    started = time.monotonic()
    next_progress = progress_interval_seconds
    observed_prompt_tokens = 0
    observed_completion_tokens = 0

    with responses_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, start=1):
            request_started = time.monotonic()
            response = post_completion(
                endpoint=endpoint,
                model=model,
                prompt=row["text"],
                timeout_seconds=timeout_seconds,
            )
            usage = response.get("usage")
            if not isinstance(usage, dict):
                raise ValueError(f"response for {row['id']} has no usage object")
            prompt_tokens = int(usage.get("prompt_tokens", -1))
            completion_tokens = int(usage.get("completion_tokens", -1))
            if prompt_tokens != row["tokens"]:
                raise ValueError(
                    f"prompt token mismatch for {row['id']}: "
                    f"{prompt_tokens} != {row['tokens']}"
                )
            if completion_tokens != 1:
                raise ValueError(
                    f"completion token mismatch for {row['id']}: "
                    f"{completion_tokens} != 1"
                )
            observed_prompt_tokens += prompt_tokens
            observed_completion_tokens += completion_tokens
            response_sha256 = hashlib.sha256(
                json.dumps(response, sort_keys=True).encode()
            ).hexdigest()
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "request_id": response.get("id"),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "duration_seconds": time.monotonic() - request_started,
                        "response_sha256": response_sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            elapsed = time.monotonic() - started
            if elapsed >= next_progress:
                print(
                    json.dumps(
                        {
                            "elapsed_seconds": elapsed,
                            "completed_rows": index,
                            "total_rows": len(rows),
                            "observed_prompt_tokens": observed_prompt_tokens,
                            "expected_prompt_tokens": expected_tokens,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                while elapsed >= next_progress:
                    next_progress += progress_interval_seconds

    responses_sha256 = hashlib.sha256(responses_path.read_bytes()).hexdigest()
    summary = {
        "format_version": 1,
        "split": split,
        "rows": len(rows),
        "prompt_tokens": observed_prompt_tokens,
        "completion_tokens": observed_completion_tokens,
        "duration_seconds": time.monotonic() - started,
        "responses_sha256": responses_sha256,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "holdout"), required=True)
    parser.add_argument("--expected-tokens", type=int, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()

    summary = run_calibration_split(
        manifest_path=args.manifest,
        split=args.split,
        expected_tokens=args.expected_tokens,
        endpoint=args.base_url.rstrip("/") + "/completions",
        model=args.model,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
