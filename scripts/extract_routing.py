#!/usr/bin/env python3
"""
Archon routing extractor (rebuilt 2026-06-05).

Reconstructs the phase0 *routing* extraction that originally ran inside the
Cortana container against /workspace/archon/ but was never committed. This
version reads the live Archon surface and emits the same raw-extract schema
found in seeds/phase0_raw_extracts/, so the downstream format/master steps
are unchanged.

Narrow lane: workflow / command / subagent ROUTING only. This is the first
"narrow & deep" specialist target (highest-frequency Archon node call).

Schema emitted (matches the original seeds):
  workflow_routing.jsonl
    {"source_file","workflow_name","intent","pair_type":"description_to_workflow"}
    {"source_file","workflow_name","intent":<trigger>,"pair_type":"trigger_to_workflow"}
  command_routing.jsonl
    {"source_file","command_name","intent":<description>,"pair_type":"description_to_command"}
  subagent_routing.jsonl
    {"source_file","subagent_name","description","tools","body_char_count"}

Usage:
  python3 extract_routing.py \
      --workflows ../../.archon/workflows \
      --commands  ../../.archon/commands \
      --agents    ../../.claude/agents \
      --out       seeds/phase0_raw_extracts_v2
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

# --- description-block keys we split on (Use when / Triggers / Does / NOT for) ---
_SECTION_KEYS = ("Does:", "NOT for:", "NOT:", "Based on", "Separates", "Capability:")


def parse_triggers(description: str) -> list[str]:
    """Pull the quoted trigger phrases out of the `Triggers:` section."""
    if not description or "Triggers:" not in description:
        return []
    after = description.split("Triggers:", 1)[1]
    # cut at the next known section key so we don't grab quotes from Does:/NOT for:
    cut = len(after)
    for key in _SECTION_KEYS:
        idx = after.find(key)
        if idx != -1:
            cut = min(cut, idx)
    section = after[:cut]
    # unique, order-preserving
    seen, out = set(), []
    for m in re.findall(r'"([^"]+)"', section):
        t = m.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return out


def extract_workflows(wf_dir: Path) -> tuple[list[dict], dict]:
    pairs, stats = [], {"files": 0, "desc_pairs": 0, "trigger_pairs": 0, "no_desc": [], "no_triggers": []}
    for path in sorted(wf_dir.glob("*.y*ml")):
        try:
            doc = yaml.safe_load(path.read_text())
        except Exception as e:
            print(f"  ! skip {path.name}: {e}", file=sys.stderr); continue
        if not isinstance(doc, dict):
            continue
        name = doc.get("name") or path.stem
        desc = (doc.get("description") or "").strip()
        src = str(path)
        stats["files"] += 1
        if desc:
            pairs.append({"source_file": src, "workflow_name": name,
                          "intent": desc, "pair_type": "description_to_workflow"})
            stats["desc_pairs"] += 1
        else:
            stats["no_desc"].append(name)
        trigs = parse_triggers(desc)
        if not trigs:
            stats["no_triggers"].append(name)
        for t in trigs:
            pairs.append({"source_file": src, "workflow_name": name,
                          "intent": t, "pair_type": "trigger_to_workflow"})
            stats["trigger_pairs"] += 1
    return pairs, stats


def extract_commands(cmd_dir: Path) -> tuple[list[dict], int]:
    pairs = []
    if not cmd_dir.exists():
        return pairs, 0
    for path in sorted(cmd_dir.glob("*.md")):
        text = path.read_text()
        desc = ""
        # YAML frontmatter description, else first non-empty line
        fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        if fm:
            try:
                meta = yaml.safe_load(fm.group(1)) or {}
                desc = (meta.get("description") or "").strip()
            except Exception:
                pass
        if not desc:
            for line in text.splitlines():
                if line.strip() and not line.startswith(("#", "---")):
                    desc = line.strip(); break
        if desc:
            pairs.append({"source_file": str(path), "command_name": path.stem,
                          "intent": desc, "pair_type": "description_to_command"})
    return pairs, len(pairs)


def extract_subagents(agent_dir: Path) -> tuple[list[dict], int]:
    out = []
    if not agent_dir.exists():
        return out, 0
    for path in sorted(agent_dir.glob("*.md")):
        text = path.read_text()
        name, desc, tools = path.stem, "", []
        fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        body = text
        if fm:
            body = text[fm.end():]
            try:
                meta = yaml.safe_load(fm.group(1)) or {}
                name = meta.get("name") or name
                desc = (meta.get("description") or "").strip()
                tl = meta.get("tools")
                if isinstance(tl, str):
                    tools = [t.strip() for t in tl.split(",") if t.strip()]
                elif isinstance(tl, list):
                    tools = tl
            except Exception:
                pass
        out.append({"source_file": str(path), "subagent_name": name,
                    "description": desc, "tools": tools,
                    "body_char_count": len(body)})
    return out, len(out)


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    root = here.parent.parent  # repo root: training_data/archon-master-dataset/ -> repo
    ap.add_argument("--workflows", default=str(root / ".archon/workflows"))
    ap.add_argument("--commands",  default=str(root / ".archon/commands"))
    ap.add_argument("--agents",    default=str(root / ".claude/agents"))
    ap.add_argument("--out",       default=str(here / "seeds/phase0_raw_extracts_v2"))
    args = ap.parse_args()

    out = Path(args.out)
    wf_pairs, wf_stats = extract_workflows(Path(args.workflows))
    cmd_pairs, n_cmd = extract_commands(Path(args.commands))
    sub_rows, n_sub = extract_subagents(Path(args.agents))

    write_jsonl(wf_pairs, out / "workflow_routing.jsonl")
    write_jsonl(cmd_pairs, out / "command_routing.jsonl")
    write_jsonl(sub_rows, out / "subagent_routing.jsonl")

    report = {
        "source": {"workflows": args.workflows, "commands": args.commands, "agents": args.agents},
        "workflows": {"parsed": wf_stats["files"],
                      "description_pairs": wf_stats["desc_pairs"],
                      "trigger_pairs": wf_stats["trigger_pairs"],
                      "workflows_missing_description": wf_stats["no_desc"],
                      "workflows_missing_triggers": wf_stats["no_triggers"]},
        "commands": {"routing_pairs": n_cmd},
        "subagents": {"parsed": n_sub},
        "total_routing_pairs": wf_stats["desc_pairs"] + wf_stats["trigger_pairs"] + n_cmd,
    }
    write_jsonl([report], out / "extraction_report.jsonl")

    print(json.dumps(report, indent=2))
    print(f"\nWrote raw extracts to: {out}")


if __name__ == "__main__":
    main()
