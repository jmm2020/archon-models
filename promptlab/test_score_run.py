"""Unit tests for score_run (judgment loading + row scoring with injected judge)."""
import json

from score_run import load_judgments, score_rows

MANIFEST = {"create-plan": {"output_type": "markdown"},
            "routing": {"output_type": "json",
                        "schema": {"required_keys": ["workflow", "confidence", "escalate"],
                                   "enums": {"confidence": ["high", "med", "low"]}}}}
EVAL = {"create-plan": [{"role": "create-plan", "system": "sys", "user": "u",
                         "gold": "# Plan\n## Tasks\nstuff\n## Risks\nr"}],
        "routing": [{"role": "routing", "system": "sys", "user": "u",
                     "gold": '{"workflow":"archon-ci-fix","confidence":"high","escalate":false}'}]}


def _write_run(tmp_path, rows):
    run_dir = tmp_path / "promptlab_t"
    for m, text in rows:
        f = run_dir / m["output_file"]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return run_dir, [m for m, _ in rows]


def test_load_judgments_keying(tmp_path):
    f = tmp_path / "judgments.jsonl"
    f.write_text(json.dumps({"role": "create-plan", "variant": "v1", "idx": 0,
                             "faithfulness": 5, "completeness": 4, "grounding": 4, "format": 5}) + "\n")
    j = load_judgments(f)
    assert j[("create-plan", "v1", 0)]["completeness"] == 4
    assert load_judgments(None) == {}


def test_markdown_row_with_judgment_gets_composite(tmp_path):
    md = "# Plan\n## Tasks\nstuff\n## Risks\nr"
    run_dir, manifest_rows = _write_run(tmp_path, [
        ({"role": "create-plan", "variant": "v1", "idx": 0, "out_type": "markdown",
          "output_file": "outputs/create-plan/v1__0.txt"}, md)])
    judgments = {("create-plan", "v1", 0): {"faithfulness": 5, "completeness": 5,
                                            "grounding": 5, "format": 5}}
    rows = score_rows(manifest_rows, run_dir, EVAL, MANIFEST, judgments)
    assert rows[0]["judge"] == 1.0
    assert rows[0]["score"] == 1.0  # gate 1.0 * 0.4 + judge 1.0 * 0.6
    assert rows[0]["variant"] == "v1" and rows[0]["idx"] == 0


def test_markdown_row_without_judgment_is_flagged_deterministic(tmp_path):
    md = "# Plan\n## Tasks\nstuff\n## Risks\nr"
    run_dir, manifest_rows = _write_run(tmp_path, [
        ({"role": "create-plan", "variant": "baseline", "idx": 0, "out_type": "markdown",
          "output_file": "outputs/create-plan/baseline__0.txt"}, md)])
    rows = score_rows(manifest_rows, run_dir, EVAL, MANIFEST, {})
    assert rows[0]["judge"] is None
    assert rows[0]["score"] == rows[0]["gate"]
    assert any("no judgment available" in n for n in rows[0]["notes"])


def test_json_row_scored_deterministically(tmp_path):
    out = '{"workflow":"archon-ci-fix","confidence":"high","escalate":false}'
    run_dir, manifest_rows = _write_run(tmp_path, [
        ({"role": "routing", "variant": "v2", "idx": 0, "out_type": "json",
          "output_file": "outputs/routing/v2__0.txt"}, out)])
    rows = score_rows(manifest_rows, run_dir, EVAL, MANIFEST, {})
    assert rows[0]["score"] == 1.0
    assert rows[0]["parses"] is True
