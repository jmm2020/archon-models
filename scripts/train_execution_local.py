#!/usr/bin/env python3
"""
Local LoRA training for Gemma4-Archon v2 — the EXECUTION corpus.

This is the second, broader lane beyond the v1 router: instead of learning
request -> routing-JSON, the model learns to *produce the execution artifact*
(plans, traces) the way the Opus teacher did. Data is the distilled v2 corpus
(buckets E+, scenario-grounded), already in the internal `conversations` format.

Differences from train_routing_local.py (v1):
  - Loads data/build/archon_v2.{train,eval}.jsonl directly (no anchor transform).
  - max_seq defaults to 4096: the v2 corpus has p95~3.3k, p99~3.9k tokens;
    at 2048 (v1's value) ~39% of records would truncate mid-plan, masking the
    very targets we train on. 4096 captures 99.6% fully.
  - Wires the held-out eval set in (eval loss is the only honest signal we have).

Modes:
  --smoke   tiny run (few steps) to validate the full pipeline end-to-end:
            model download, 4-bit load, chat template, LoRA, train+eval step,
            adapter save path. Run this FIRST.
  (default) real run over the full v2 corpus.

Pick the GPU with: CUDA_VISIBLE_DEVICES=1 python scripts/train_execution_local.py
Output: runs/archon-gemma4-v2-lora/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN = ROOT / "data/build/archon_v2.train.jsonl"
EVAL = ROOT / "data/build/archon_v2.eval.jsonl"
OUT = HERE / "runs/archon-gemma4-v2-lora"


def load_conversations(path: Path) -> list[dict]:
    """v2 records are {"conversations": [...], "meta": {...}} — keep convs only."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({"conversations": r["conversations"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline-validation run")
    ap.add_argument("--model", default="unsloth/gemma-4-12b")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1, help="per-device batch (seq 4096 is memory-heavy)")
    ap.add_argument("--grad-accum", type=int, default=16, help="effective batch = batch * grad_accum")
    args = ap.parse_args()

    print("[1/6] importing unsloth ...", flush=True)
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print(f"[2/6] loading data from {TRAIN.name} / {EVAL.name} ...", flush=True)
    train_rows = load_conversations(TRAIN)
    eval_rows = load_conversations(EVAL)
    print(f"      train={len(train_rows)}  eval={len(eval_rows)}", flush=True)
    if args.smoke:
        train_rows = train_rows[:32]
        eval_rows = eval_rows[:8]
        print(f"      SMOKE: trimmed to train={len(train_rows)} eval={len(eval_rows)}", flush=True)

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

    train_ds = Dataset.from_list(train_rows).map(fmt, batched=True, remove_columns=["conversations"])
    eval_ds = Dataset.from_list(eval_rows).map(fmt, batched=True, remove_columns=["conversations"])
    print(f"      sample text:\n{train_ds[0]['text'][:400]}\n", flush=True)

    print("[4/6] attaching LoRA ...", flush=True)
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
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=2 if args.smoke else 20,
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
        eval_strategy="steps" if args.smoke else "epoch",
        eval_steps=5 if args.smoke else None,
        per_device_eval_batch_size=args.batch,
    )

    print(f"[5/6] training ({'SMOKE' if args.smoke else 'FULL'}) ...", flush=True)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=eval_ds, args=cfg,
    )
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
