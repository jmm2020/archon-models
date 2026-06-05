#!/usr/bin/env python3
"""
Local LoRA training for the Archon ROUTING specialist (Gemma-4-12B, 4-bit).

Runs on a single local GPU (default GPU 1 via CUDA_VISIBLE_DEVICES set by caller).
First "narrow & deep" lane: request -> select_workflow(workflow, confidence, escalate).

Modes:
  --smoke   tiny run (few steps, no real training) to validate the full pipeline
            end-to-end: model download, 4-bit load, chat template, LoRA, train step,
            adapter save. Use this FIRST.
  (default) real run over the routing anchors.

Data source: seeds/phase0_raw_extracts_v2/{workflow_routing,command_routing}.jsonl
Output:      runs/archon-router-gemma4-lora/

NOTE: anchors alone (~220) are not enough for a *good* router — this is the
pipeline-validation + baseline. Synthesized paraphrases + hard negatives +
abstention get layered in next.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = HERE / "seeds/phase0_raw_extracts_v2"
OUT = HERE / "runs/archon-router-gemma4-lora"

# Internal/test workflows that are NOT user-routable targets — exclude from labels.
EXCLUDE = {
    "e2e-claude-smoke", "e2e-codex-smoke", "e2e-deterministic", "e2e-minimax-smoke",
    "e2e-mixed-providers", "e2e-pi-all-nodes-smoke", "e2e-pi-smoke",
    "e2e-worktree-disabled",
}

SYS_PROMPT = (
    "You are the Archon workflow router. Given a user request, select the single best "
    "Archon workflow by name. If no workflow is a good match, abstain and escalate. "
    "Respond with ONLY a JSON object: "
    '{"workflow": <name or null>, "confidence": "high"|"medium"|"low", "escalate": true|false}.'
)


def load_anchors() -> list[dict]:
    """Build (user_request -> routing JSON) examples from the v2 routing extracts."""
    rows = []
    wf_path = SEEDS / "workflow_routing.jsonl"
    for line in wf_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        name = r["workflow_name"]
        if name in EXCLUDE:
            continue
        intent = r["intent"].strip()
        # trigger pairs are short phrases; description pairs are the full block
        target = {"workflow": name, "confidence": "high", "escalate": False}
        rows.append({"request": intent, "target": target})
    return rows


def to_conversations(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({"conversations": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": r["request"]},
            {"role": "assistant", "content": json.dumps(r["target"], ensure_ascii=False)},
        ]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline-validation run")
    ap.add_argument("--model", default="unsloth/gemma-4-12b")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3.0)
    args = ap.parse_args()

    print(f"[1/6] importing unsloth ...", flush=True)
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print(f"[2/6] building dataset from {SEEDS} ...", flush=True)
    rows = load_anchors()
    convs = to_conversations(rows)
    n_labels = len({r['target']['workflow'] for r in rows})
    print(f"      {len(convs)} examples across {n_labels} routable workflows", flush=True)
    if args.smoke:
        convs = convs[:32]
        print(f"      SMOKE: trimmed to {len(convs)} examples", flush=True)

    print(f"[3/6] loading {args.model} (4-bit) ...", flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    def fmt(batch):
        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                 for c in batch["conversations"]]
        return {"text": texts}

    ds = Dataset.from_list(convs).map(fmt, batched=True, remove_columns=["conversations"])
    print(f"      sample text:\n{ds[0]['text'][:400]}\n", flush=True)

    print(f"[4/6] attaching LoRA ...", flush=True)
    model = FastModel.get_peft_model(
        model,
        r=32, lora_alpha=32, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    cfg = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=2 if args.smoke else 10,
        max_steps=5 if args.smoke else -1,
        num_train_epochs=1 if args.smoke else args.epochs,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        bf16=True, fp16=False,
        report_to="none",
        output_dir=str(OUT),
        save_strategy="no" if args.smoke else "epoch",
    )

    print(f"[5/6] training ({'SMOKE' if args.smoke else 'FULL'}) ...", flush=True)
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=cfg)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",   # Gemma-4 unified delimiters (verified from tokenizer)
        response_part="<|turn>model\n",
    )
    stats = trainer.train()
    print(f"      train loss: {stats.training_loss:.4f}", flush=True)

    if not args.smoke:
        print(f"[6/6] saving adapter -> {OUT} ...", flush=True)
        model.save_pretrained(str(OUT))
        tokenizer.save_pretrained(str(OUT))
    else:
        print("[6/6] SMOKE complete — pipeline validated, adapter not saved.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
