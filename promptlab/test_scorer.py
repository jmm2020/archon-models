#!/usr/bin/env python3
"""Unit tests for the deterministic tier of promptlab.scorer.

Run: python -m pytest promptlab/test_scorer.py -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorer  # noqa: E402

ROUTING_SCHEMA = {
    "required_keys": ["confidence", "escalate", "workflow"],
    "enums": {"confidence": ["high", "med", "low"]},
}
MANIFEST = {
    "routing": {"output_type": "json", "schema": ROUTING_SCHEMA},
    "classify-issue": {"output_type": "json",
                       "schema": {"required_keys": ["issue_type", "title", "reasoning"],
                                  "enums": {"issue_type": ["bug", "feature", "enhancement",
                                                           "refactor", "chore", "documentation"]}}},
    "structured-output": {"output_type": "json", "schema": {"required_keys": [], "enums": {}}},
    "create-plan": {"output_type": "markdown", "schema": None},
}


# ---- JSON: parse failures -------------------------------------------------

def test_unparseable_json_scores_zero():
    r = scorer.score_json("routing", "not json at all", {"workflow": "x", "escalate": False}, ROUTING_SCHEMA)
    assert r.parses is False
    assert r.score == 0.0


def test_json_fences_are_tolerated():
    out = "```json\n{\"workflow\":\"archon-debug-regression\",\"confidence\":\"high\",\"escalate\":false}\n```"
    assert scorer.parse_json_loose(out)["workflow"] == "archon-debug-regression"


# ---- JSON: routing field match -------------------------------------------

def test_routing_exact_match_scores_one():
    gold = {"workflow": "archon-debug-regression", "confidence": "high", "escalate": False}
    out = json.dumps(gold)
    r = scorer.score_json("routing", out, gold, ROUTING_SCHEMA)
    assert r.keys_ok and r.enums_ok and r.match == 1.0
    assert r.score == 1.0


def test_routing_wrong_workflow_loses_match():
    gold = {"workflow": "archon-debug-regression", "confidence": "high", "escalate": False}
    out = json.dumps({"workflow": "archon-write-tests", "confidence": "high", "escalate": False})
    r = scorer.score_json("routing", out, gold, ROUTING_SCHEMA)
    # wrong workflow (0.7 lost), escalate matches (0.3 kept)
    assert r.match == 0.3
    assert r.score < 1.0


def test_routing_missing_key_flagged():
    gold = {"workflow": "x", "confidence": "high", "escalate": False}
    out = json.dumps({"workflow": "x", "confidence": "high"})  # no escalate
    r = scorer.score_json("routing", out, gold, ROUTING_SCHEMA)
    assert r.keys_ok is False


def test_routing_bad_enum_flagged():
    gold = {"workflow": "x", "confidence": "high", "escalate": False}
    out = json.dumps({"workflow": "x", "confidence": "SUPER", "escalate": False})
    r = scorer.score_json("routing", out, gold, ROUTING_SCHEMA)
    assert r.enums_ok is False


# ---- JSON: classify-issue -------------------------------------------------

def test_classify_issue_type_match():
    gold = {"issue_type": "bug", "title": "x", "reasoning": "y"}
    out = json.dumps({"issue_type": "bug", "title": "z", "reasoning": "w"})
    r = scorer.score_json("classify-issue", out, gold, MANIFEST["classify-issue"]["schema"])
    assert r.match == 1.0


def test_classify_issue_type_mismatch():
    gold = {"issue_type": "bug", "title": "x", "reasoning": "y"}
    out = json.dumps({"issue_type": "feature", "title": "z", "reasoning": "w"})
    r = scorer.score_json("classify-issue", out, gold, MANIFEST["classify-issue"]["schema"])
    assert r.match == 0.0


# ---- JSON: structured-output (per-record schema) --------------------------

def test_structured_output_validates_against_gold_keys():
    gold = {"decision": "approve", "reason": "ok", "required_actions": []}
    out = json.dumps({"decision": "approve", "reason": "ok", "required_actions": []})
    r = scorer.score_json("structured-output", out, gold, {"required_keys": [], "enums": {}})
    assert r.match == 1.0 and r.keys_ok is True


def test_structured_output_missing_keys_partial():
    gold = {"decision": "approve", "reason": "ok", "required_actions": []}
    out = json.dumps({"decision": "approve"})  # missing two gold keys
    r = scorer.score_json("structured-output", out, gold, {"required_keys": [], "enums": {}})
    assert r.keys_ok is False
    assert 0.0 < r.match < 1.0


# ---- Markdown gate --------------------------------------------------------

GOLD_MD = "# Feature: X\n\n**Summary**: ...\n\n## Tasks\n- a\n- b\n\n## Risks\n- r\n"


def test_markdown_full_sections_pass():
    out = "# Feature: X\n\nbody\n\n## Tasks\n- a\n\n## Risks\n- r\n" + ("x" * 50)
    r = scorer.score_markdown("create-plan", out, GOLD_MD)
    assert r.gate == 1.0 and r.score == 1.0


def test_markdown_missing_sections_partial():
    out = "# Feature: X\n\nbody only, no other headers\n" + ("x" * 50)
    r = scorer.score_markdown("create-plan", out, GOLD_MD)
    assert 0.0 < r.gate < 1.0


def test_markdown_empty_fails():
    r = scorer.score_markdown("create-plan", "   ", GOLD_MD)
    assert r.gate == 0.0 and r.score == 0.0


def test_markdown_leakage_penalized():
    sysp = "You are an implementation planner for the mealwise codebase (TypeScript)."
    leaky = sysp + "\n\n# Feature: X\n## Tasks\n- a\n## Risks\n- r\n" + ("x" * 80)
    clean = "# Feature: X\n## Tasks\n- a\n## Risks\n- r\n" + ("x" * 80)
    r_leak = scorer.score_markdown("create-plan", leaky, GOLD_MD, system_prompt=sysp)
    r_clean = scorer.score_markdown("create-plan", clean, GOLD_MD, system_prompt=sysp)
    assert r_leak.gate < r_clean.gate


def test_judge_injection_blends_with_gate():
    out = "# Feature: X\n## Tasks\n- a\n## Risks\n- r\n" + ("x" * 80)
    def fake_judge(role, o, g):
        return {"faithfulness": 5, "completeness": 5, "grounding": 5, "format": 5}
    r = scorer.score_markdown("create-plan", out, GOLD_MD, judge=fake_judge)
    assert r.judge == 1.0
    assert r.score == round(0.4 * r.gate + 0.6 * 1.0, 3)


# ---- dispatch -------------------------------------------------------------

def test_tiered_dispatch_json_vs_markdown():
    rec_json = {"role": "routing", "gold": json.dumps({"workflow": "x", "confidence": "high", "escalate": False})}
    r1 = scorer.tiered_score(rec_json, json.dumps({"workflow": "x", "confidence": "high", "escalate": False}), MANIFEST)
    assert r1.output_type == "json" and r1.score == 1.0

    rec_md = {"role": "create-plan", "gold": GOLD_MD, "system": ""}
    out = "# Feature: X\n## Tasks\n- a\n## Risks\n- r\n" + ("x" * 80)
    r2 = scorer.tiered_score(rec_md, out, MANIFEST)
    assert r2.output_type == "markdown"


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
