"""Unit tests for build_jobs (pure parts: instantiate, validate, build)."""
from build_jobs import SLOT, build_jobs, instantiate, reconstruction_ratio, validate_role

TMPL = f"You are a planner for {SLOT}. Output plan.md only."
FACTORED = {"template": TMPL,
            "records": [{"idx": 0, "scenario_context": "acme-shop (Django)"},
                        {"idx": 1, "scenario_context": "fleet-api (Go)"}]}
CANDS = [{"name": "v1_strict", "template": f"Strict planner for {SLOT}. Sections required."},
         {"name": "v2_phased", "template": f"Phased planner for {SLOT}. Plan only."}]
RECORDS = [{"system": "You are a planner for acme-shop (Django). Output plan.md only.", "user": "u0"},
           {"system": "You are a planner for fleet-api (Go). Output plan.md only.", "user": "u1"}]


def test_instantiate_substitutes_slot():
    assert instantiate(TMPL, "acme") == "You are a planner for acme. Output plan.md only."


def test_instantiate_verbatim_without_slot():
    assert instantiate("fixed router prompt", "ignored") == "fixed router prompt"


def test_validate_ok():
    assert validate_role("r", FACTORED, CANDS, RECORDS) == []


def test_validate_catches_missing_context_and_bad_slots():
    bad_factored = {"template": TMPL, "records": [{"idx": 0, "scenario_context": ""}]}
    bad_cands = [{"name": "v1", "template": "no slot here"}]
    problems = validate_role("r", bad_factored, bad_cands, RECORDS)
    assert any("empty scenario_context" in p for p in problems)
    assert any("missing from factored" in p for p in problems)
    assert any("0 slots" in p for p in problems)


def test_validate_slotless_role():
    """Scenario-free role: no slot anywhere is valid (templates used verbatim)."""
    factored = {"template": "fixed router prompt",
                "records": [{"idx": 0, "scenario_context": ""}]}
    cands = [{"name": "v1", "template": "better fixed router prompt"}]
    assert validate_role("routing", factored, cands, RECORDS[:1]) == []


def test_build_jobs_variants_and_substitution():
    jobs = build_jobs("r", FACTORED, CANDS, RECORDS)
    assert len(jobs) == 2 * (1 + len(CANDS))
    base = [j for j in jobs if j["variant"] == "baseline"]
    assert base[0]["system"] == RECORDS[0]["system"]
    v1_r1 = next(j for j in jobs if j["variant"] == "v1_strict" and j["idx"] == 1)
    assert "fleet-api (Go)" in v1_r1["system"] and SLOT not in v1_r1["system"]
    assert v1_r1["user"] == "u1"


def test_reconstruction_ratio_perfect_on_exact_factoring():
    assert reconstruction_ratio(FACTORED, RECORDS) == 1.0
