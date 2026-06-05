#!/usr/bin/env python3
"""Validate contributed training-example JSONL files for archon-models.

Every contributed example is one JSON object per line (JSONL). This checks each
file against the v0.1 schema (see docs/DATA_SCHEMA.md), runs a basic PII scan,
and flags in-file duplicates. Exit code is non-zero if any file fails, so it
doubles as the CI gate on pull requests.

Usage:
    python scripts/validate_examples.py data/contrib/**/*.jsonl
    python scripts/validate_examples.py data/contrib            # walks the dir
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

VALID_BUCKETS = {"A", "B", "C", "D", "E", "F", "G", "H"}
VALID_ROLES = {"system", "user", "assistant", "tool"}
VALID_SOURCES = {"synthetic", "distilled", "real-run", "hand-written"}

# Conservative PII / leak patterns — flag, don't auto-fix.
PII_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"), "anthropic key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "github token"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/"), "absolute home path"),
    (re.compile(r"[A-Za-z0-9._%+-]+@(?:gmail|outlook|yahoo|hotmail)\.[A-Za-z]{2,}"), "personal email"),
]


def _err(file: Path, line_no: int | None, msg: str) -> str:
    loc = f"{file}" + (f":{line_no}" if line_no else "")
    return f"  ✗ {loc}: {msg}"


def validate_record(rec: object) -> list[str]:
    """Return a list of problems with one record (empty list = valid)."""
    problems: list[str] = []
    if not isinstance(rec, dict):
        return ["record is not a JSON object"]

    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        problems.append("'messages' must be a list of >= 2 turns")
        msgs = []

    roles_seen = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            problems.append(f"messages[{i}] is not an object")
            continue
        role = m.get("role")
        if role not in VALID_ROLES:
            problems.append(f"messages[{i}].role '{role}' not in {sorted(VALID_ROLES)}")
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            problems.append(f"messages[{i}].content must be a non-empty string")
        roles_seen.append(role)

    if msgs:
        if "user" not in roles_seen:
            problems.append("needs at least one 'user' turn")
        if "assistant" not in roles_seen:
            problems.append("needs at least one 'assistant' turn")
        if roles_seen and roles_seen[-1] != "assistant":
            problems.append("final turn must be 'assistant'")

    meta = rec.get("meta")
    if not isinstance(meta, dict):
        problems.append("'meta' object is required")
    else:
        if meta.get("bucket") not in VALID_BUCKETS:
            problems.append(f"meta.bucket '{meta.get('bucket')}' not in {sorted(VALID_BUCKETS)}")
        if not isinstance(meta.get("author"), str) or not meta.get("author", "").strip():
            problems.append("meta.author (your GitHub handle) is required")
        src = meta.get("source")
        if src is not None and src not in VALID_SOURCES:
            problems.append(f"meta.source '{src}' not in {sorted(VALID_SOURCES)}")
    return problems


def scan_pii(text: str) -> list[str]:
    return [label for pat, label in PII_PATTERNS if pat.search(text)]


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:  # noqa: BLE001
        return [_err(path, None, f"could not read: {e}")]

    n = 0
    for line_no, raw in enumerate(lines, 1):
        raw = raw.strip()
        if not raw:
            continue
        n += 1
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(_err(path, line_no, f"invalid JSON ({e.msg})"))
            continue
        for p in validate_record(rec):
            errors.append(_err(path, line_no, p))
        for leak in scan_pii(raw):
            errors.append(_err(path, line_no, f"possible {leak} — remove before contributing"))
        if raw in seen:
            errors.append(_err(path, line_no, "duplicate record (identical to an earlier line)"))
        seen.add(raw)

    if n == 0:
        errors.append(_err(path, None, "no records found"))
    return errors


def collect_files(args: list[str]) -> list[Path]:
    files: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
        elif p.suffix == ".jsonl":
            files.append(p)
    return files


def main() -> int:
    targets = sys.argv[1:] or ["data/contrib"]
    files = collect_files(targets)
    if not files:
        print("No .jsonl files found to validate.")
        return 0

    all_errors: list[str] = []
    for f in files:
        errs = validate_file(f)
        if errs:
            all_errors.extend(errs)
        else:
            print(f"  ✓ {f}")

    if all_errors:
        print("\nValidation FAILED:\n" + "\n".join(all_errors))
        print(f"\n{len(all_errors)} problem(s) across {len(files)} file(s).")
        return 1
    print(f"\nAll {len(files)} file(s) valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
