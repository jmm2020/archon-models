#!/usr/bin/env python3
"""
PromptLab runner, llama.cpp backend — same contract as runner.py
(jobs.jsonl in, runs/promptlab_<run>/outputs/ + outputs.jsonl out), but
generation goes through a llama-server /v1/chat/completions endpoint serving
the merged v2 GGUF. Temp 0, max_tokens by role output type.

Why this exists: unsloth's 4-bit generate path crawled at ~3.6 tok/s (15% GPU
util) for this model; llama.cpp serves the same merged weights an order of
magnitude faster. Engine choice is constant across all variants in a run, so
leaderboards remain internally consistent.

Usage:
  llama-server -m archon-gemma4-v2-q8_0.gguf --jinja -c 4096 -ngl 99 --port 8083
  python promptlab/runner_llamacpp.py --jobs promptlab/jobs.jsonl --run subset_v2
"""
from __future__ import annotations
import argparse, json, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

MAX_NEW = {"json": 512, "markdown": 2048}


def generate(server: str, system: str, user: str, max_tokens: int, timeout: int = 600) -> str:
    payload = json.dumps({
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(f"{server}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--run", required=True, help="run label -> runs/promptlab_<run>/")
    ap.add_argument("--server", default="http://127.0.0.1:8083")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads((HERE / "roles.json").read_text())
    jobs = [json.loads(l) for l in Path(args.jobs).read_text().splitlines() if l.strip()]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[runner] {len(jobs)} jobs over roles={sorted({j['role'] for j in jobs})} via {args.server}", flush=True)

    run_dir = HERE / "runs" / f"promptlab_{args.run}"
    out_dir = run_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_f = open(run_dir / "outputs.jsonl", "w")

    t0 = time.time()
    for n, job in enumerate(jobs):
        role = job["role"]
        out_type = manifest.get(role, {}).get("output_type", "markdown")
        try:
            text = generate(args.server, job["system"], job["user"], MAX_NEW[out_type])
        except Exception as e:  # gen failure -> empty output, scored 0, logged
            print(f"[runner] GEN FAIL {role}/{job['variant']}__{job['idx']}: {e}", flush=True)
            text = ""
        rel = out_dir / role / f"{job['variant']}__{job['idx']}.txt"
        rel.parent.mkdir(parents=True, exist_ok=True)
        rel.write_text(text)
        manifest_f.write(json.dumps({"variant": job["variant"], "role": role,
                                     "idx": job["idx"], "out_type": out_type,
                                     "output_file": str(rel.relative_to(run_dir))}) + "\n")
        manifest_f.flush()
        if (n + 1) % 5 == 0 or n == 0:
            print(f"[runner] {n+1}/{len(jobs)}  ({(time.time()-t0)/(n+1):.1f}s/gen)", flush=True)

    manifest_f.close()
    print(f"[runner] DONE {len(jobs)} gens in {time.time()-t0:.0f}s -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
