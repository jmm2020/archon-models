#!/usr/bin/env python3
"""Unit tests for promptlab.report aggregation. Run: python -m pytest promptlab/test_report.py -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402


def _rows():
    rows = []
    for idx in range(4):
        rows.append({"variant": "baseline", "role": "create-plan", "idx": idx, "score": 0.60})
        rows.append({"variant": "cand_contract", "role": "create-plan", "idx": idx, "score": 0.80})
        rows.append({"variant": "cand_terse", "role": "create-plan", "idx": idx, "score": 0.50})
    for idx in range(3):
        rows.append({"variant": "baseline", "role": "routing", "idx": idx, "score": 0.67})
        rows.append({"variant": "cand_contract", "role": "routing", "idx": idx, "score": 1.00})
    return rows


def test_winner_is_highest_mean():
    agg = report.aggregate(_rows())
    assert agg["create-plan"]["winner"] == "cand_contract"
    assert agg["routing"]["winner"] == "cand_contract"


def test_leaderboard_sorted_desc():
    agg = report.aggregate(_rows())
    means = [row["mean"] for row in agg["create-plan"]["leaderboard"]]
    assert means == sorted(means, reverse=True)


def test_delta_vs_baseline():
    agg = report.aggregate(_rows())
    board = {r["variant"]: r for r in agg["create-plan"]["leaderboard"]}
    assert board["cand_contract"]["delta_vs_baseline"] == 0.20
    assert board["cand_terse"]["delta_vs_baseline"] == -0.10
    assert board["baseline"]["delta_vs_baseline"] == 0.0


def test_n_counts_records():
    agg = report.aggregate(_rows())
    board = {r["variant"]: r for r in agg["create-plan"]["leaderboard"]}
    assert board["baseline"]["n"] == 4
    assert agg["routing"]["leaderboard"][0]["n"] == 3


def test_render_md_marks_winner_and_lowconf():
    agg = report.aggregate(_rows())
    index = {"roles": {"routing": {"low_confidence": True}, "create-plan": {"low_confidence": False}}}
    md = report.render_md(agg, index)
    assert "🏆" in md
    assert "low-confidence" in md
    assert "## routing" in md and "## create-plan" in md


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
