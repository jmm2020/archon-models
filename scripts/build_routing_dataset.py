#!/usr/bin/env python3
"""
Build the Gemma-4 ROUTING dataset (narrow & deep, first lane).

Assembles four buckets into a Gemma-formatted train/eval split:
  1. POSITIVES   — v2 routing anchors (request -> select_workflow JSON)
  2. SYNTH       — frontier paraphrases of each workflow's "Use when" (synth/*.jsonl, optional)
  3. ABSTENTION  — out-of-domain requests -> {workflow:null, escalate:true}
  4. RETENTION   — small slice of existing tool/general data, re-templated to Gemma
                   (anti-forgetting so the narrow LoRA doesn't go dumb elsewhere)

Split is STRATIFIED per workflow so eval holds out real coverage of each label.
Output: build/{train,eval}.jsonl  + build/build_stats.json

Run:  python3 build_routing_dataset.py            # anchors + abstention (+ synth if present)
      python3 build_routing_dataset.py --retention 300
"""
from __future__ import annotations
import argparse, json, random, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE / "seeds/phase0_raw_extracts_v2"
SYNTH_DIR = HERE / "synth"
OUT = HERE / "build"

EXCLUDE = {
    "e2e-claude-smoke", "e2e-codex-smoke", "e2e-deterministic", "e2e-minimax-smoke",
    "e2e-mixed-providers", "e2e-pi-all-nodes-smoke", "e2e-pi-smoke", "e2e-worktree-disabled",
}

SYS_PROMPT = (
    "You are the Archon workflow router. Given a user request, select the single best "
    "Archon workflow by name. If no workflow is a good match, abstain and escalate. "
    "Respond with ONLY a JSON object: "
    '{"workflow": <name or null>, "confidence": "high"|"medium"|"low", "escalate": true|false}.'
)

# Out-of-domain / no-workflow requests -> the model must abstain & escalate.
ABSTENTION_REQUESTS = [
    "What's the weather in Tokyo tomorrow?",
    "Write me a haiku about the ocean.",
    "Explain how quantum entanglement works.",
    "Book a flight from SFO to JFK next Friday.",
    "What's 47 times 89?",
    "Summarize the latest news about the stock market.",
    "Translate 'good morning' into Japanese.",
    "Recommend a good Italian restaurant nearby.",
    "Delete the production database right now.",
    "What time is it in London?",
    "Draft a breakup text for me.",
    "Who won the World Series in 2024?",
    "Set a reminder for my dentist appointment.",
    "Convert 100 USD to euros.",
    "Tell me a joke.",
    "What's the capital of Australia?",
    "Play some jazz music.",
    "How do I make sourdough bread?",
    "Diagnose why my car won't start.",
    "Order a pizza for delivery.",
    "What are the symptoms of the flu?",
    "Plan a 7-day itinerary for Italy.",
    "Generate an image of a sunset.",
    "What's my horoscope today?",
    "Help me write a wedding speech.",
]


def conv(request: str, target: dict) -> dict:
    return {"conversations": [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": request},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]}


def load_positives() -> list[dict]:
    """v2 anchors: description + trigger pairs -> routing positives, tagged by workflow."""
    rows = []
    for line in (V2 / "workflow_routing.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        wf = r["workflow_name"]
        if wf in EXCLUDE:
            continue
        rows.append({"label": wf,
                     "ex": conv(r["intent"].strip(),
                                {"workflow": wf, "confidence": "high", "escalate": False})})
    return rows


def load_synth() -> list[dict]:
    rows = []
    if not SYNTH_DIR.exists():
        return rows
    for f in sorted(SYNTH_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            wf, req = r["workflow"], r["request"]
            if wf in EXCLUDE:
                continue
            rows.append({"label": wf,
                         "ex": conv(req, {"workflow": wf, "confidence": "high", "escalate": False})})
    return rows


def load_abstention() -> list[dict]:
    return [{"label": "__abstain__",
             "ex": conv(req, {"workflow": None, "confidence": "low", "escalate": True})}
            for req in ABSTENTION_REQUESTS]


def load_retention(n: int) -> list[dict]:
    """Sample existing conversations data, re-tag as retention (Gemma-templated at train time)."""
    rows = []
    src = HERE.parent / "v177" / "v177_tool_calling_train.jsonl"
    candidates = [HERE.parent / "toolace_qwen_clean.jsonl", src]
    pool = []
    for c in candidates:
        if c.exists():
            for line in c.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cv = r.get("conversations")
                if cv and isinstance(cv, list):
                    norm = [{"role": ("assistant" if m.get("role") in ("gpt", "assistant") else
                                       "user" if m.get("role") in ("human", "user") else m.get("role")),
                             "content": m.get("content") or m.get("value") or ""} for m in cv]
                    norm = [m for m in norm if m["role"] in ("system", "user", "assistant") and m["content"]]
                    if len(norm) >= 2:
                        pool.append({"label": "__retention__", "ex": {"conversations": norm}})
        if len(pool) >= n * 3:
            break
    random.shuffle(pool)
    return pool[:n]


def stratified_split(rows: list[dict], eval_frac: float, min_eval_per_label: int):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    train, ev = [], []
    for label, items in by.items():
        random.shuffle(items)
        if label in ("__retention__",):       # retention stays in train only
            train.extend(items); continue
        k = max(min_eval_per_label, int(len(items) * eval_frac)) if len(items) > 2 else 0
        ev.extend(items[:k]); train.extend(items[k:])
    random.shuffle(train); random.shuffle(ev)
    return train, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retention", type=int, default=200)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    pos, syn, ab = load_positives(), load_synth(), load_abstention()
    ret = load_retention(args.retention)
    all_rows = pos + syn + ab + ret

    train, ev = stratified_split(all_rows, args.eval_frac, min_eval_per_label=1)

    OUT.mkdir(exist_ok=True)
    (OUT / "train.jsonl").write_text("".join(json.dumps(r["ex"], ensure_ascii=False) + "\n" for r in train))
    (OUT / "eval.jsonl").write_text("".join(json.dumps(r["ex"], ensure_ascii=False) + "\n" for r in ev))

    stats = {
        "buckets": {"positives": len(pos), "synth": len(syn),
                    "abstention": len(ab), "retention": len(ret)},
        "total": len(all_rows), "train": len(train), "eval": len(ev),
        "routable_workflows": len({r["label"] for r in pos}),
        "seed": args.seed, "eval_frac": args.eval_frac,
    }
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nWrote -> {OUT}/train.jsonl ({len(train)}), eval.jsonl ({len(ev)})")
    if not syn:
        print("NOTE: no synth/ data yet — positives are canonical anchors only (will overfit).")


if __name__ == "__main__":
    main()
