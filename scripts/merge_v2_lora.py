#!/usr/bin/env python3
"""
Merge the v2 LoRA adapter into gemma-4-12b on CPU -> bf16 HF checkpoint.

CPU on purpose: runs while the GPU is busy, and PeftModel.merge_and_unload
needs the base in full precision anyway (~24 GB RAM for 12B bf16).
Output feeds llama.cpp's convert_hf_to_gguf.py.

Usage: python scripts/merge_v2_lora.py [--out scripts/runs/archon-gemma4-v2-merged]
"""
from __future__ import annotations
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTER = HERE / "runs/archon-gemma4-v2-lora"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/gemma-4-12b")
    ap.add_argument("--adapter", default=str(ADAPTER))
    ap.add_argument("--out", default=str(HERE / "runs/archon-gemma4-v2-merged"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[1/4] loading base {args.base} (bf16, CPU) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
    )
    print(f"[2/4] attaching adapter {args.adapter} ...", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    print("[3/4] merge_and_unload ...", flush=True)
    merged = model.merge_and_unload()
    print(f"[4/4] saving -> {args.out} ...", flush=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.adapter)
    tok.save_pretrained(args.out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
