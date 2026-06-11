#!/usr/bin/env python3
"""
PromptLab runner — generate Gemma4-Archon outputs for (prompt variant × eval input).

Loads the model ONCE (base + v2 LoRA adapter by default; --base-only for a
pre-adapter smoke), then for every eval record runs each prompt variant through
the model with deterministic decoding (temp 0) and saves the output.

A "variant" is a system prompt under test:
  - baseline : each eval record's OWN verbatim system prompt (the trained prompt)
  - <name>   : a candidate template instantiated for the record's scenario
               (instantiation is done upstream; this runner just consumes
               {system, user} pairs from a jobs file)

Input jobs file (jsonl), one per (variant × record):
  {"variant": "...", "role": "...", "idx": 0, "system": "...", "user": "..."}
Output: runs/<run>/outputs/<role>/<variant>__<idx>.txt  + outputs.jsonl manifest.

Run on the free GPU: CUDA_VISIBLE_DEVICES=1 python promptlab/runner.py --jobs jobs.jsonl --run smoke
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ADAPTER = ROOT / "scripts/runs/archon-gemma4-v2-lora"

# token budgets by output type (json is short; plans/reviews are long)
MAX_NEW = {"json": 512, "markdown": 2048}


def build_jobs_from_eval(roles: list[str], manifest: dict, eval_dir: Path) -> list[dict]:
    """Baseline jobs: each record's own system prompt. Useful for smoke / baseline line."""
    jobs = []
    for role in roles:
        f = eval_dir / f"{role}.jsonl"
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            jobs.append({"variant": "baseline", "role": role, "idx": i,
                         "system": r["system"], "user": r["user"]})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", help="jsonl of {variant,role,idx,system,user}; if omitted, build baseline jobs from --roles")
    ap.add_argument("--roles", nargs="*", default=["create-plan", "classify-issue", "routing"])
    ap.add_argument("--run", default="smoke", help="run label -> runs/promptlab_<run>/")
    ap.add_argument("--model", default="unsloth/gemma-4-12b")
    ap.add_argument("--adapter", default=str(ADAPTER))
    ap.add_argument("--base-only", action="store_true", help="skip adapter (pre-v2 wiring smoke)")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="cap jobs (smoke)")
    args = ap.parse_args()

    manifest = json.loads((HERE / "roles.json").read_text())
    eval_dir = HERE / "eval"

    if args.jobs:
        jobs = [json.loads(l) for l in Path(args.jobs).read_text().splitlines() if l.strip()]
    else:
        jobs = build_jobs_from_eval(args.roles, manifest, eval_dir)
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[runner] {len(jobs)} jobs over roles={sorted({j['role'] for j in jobs})}", flush=True)

    print(f"[runner] loading {args.model} (4-bit){'' if args.base_only else ' + adapter'} ...", flush=True)
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    # Load the adapter dir directly (Unsloth resolves base from adapter_config);
    # transformers' model.load_adapter() integration breaks on peft version skew.
    if args.base_only:
        load_from = args.model
    else:
        adp = Path(args.adapter)
        if not adp.exists():
            raise SystemExit(f"adapter not found: {adp} (training not finished? use --base-only)")
        load_from = str(adp)
    model, tokenizer = FastModel.from_pretrained(
        model_name=load_from, max_seq_length=args.max_seq, load_in_4bit=True, dtype=None,
    )
    if not args.base_only:
        print(f"[runner] loaded adapter {adp}", flush=True)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    FastModel.for_inference(model)

    run_dir = HERE / "runs" / f"promptlab_{args.run}"
    out_dir = run_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_f = open(run_dir / "outputs.jsonl", "w")

    t0 = time.time()
    for n, job in enumerate(jobs):
        role = job["role"]
        out_type = manifest.get(role, {}).get("output_type", "markdown")
        convo = [{"role": "system", "content": job["system"]},
                 {"role": "user", "content": job["user"]}]
        inputs = tokenizer.apply_chat_template(
            convo, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        import torch
        with torch.no_grad():
            gen = model.generate(input_ids=inputs, max_new_tokens=MAX_NEW[out_type],
                                 do_sample=False, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = tokenizer.decode(gen[0][inputs.shape[1]:], skip_special_tokens=True)
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
