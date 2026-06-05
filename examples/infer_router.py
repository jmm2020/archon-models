#!/usr/bin/env python3
"""Route a user request to an Archon workflow using the trained LoRA adapter.

Usage:
    python examples/infer_router.py "fix github issue #412 in my repo"

Requires the Gemma base model (downloaded on first run; accept the Gemma
license at https://ai.google.dev/gemma/terms) plus this repo's adapter.
"""
import sys
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "unsloth/gemma-4-12b"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "model" / "archon-router-gemma4-lora"

SYSTEM_PROMPT = (
    "You are the Archon workflow router. Given a user request, select the single "
    "best Archon workflow by name. If no workflow is a good match, abstain and "
    "escalate. Respond with ONLY a JSON object: "
    '{"workflow": <name or null>, "confidence": "high"|"medium"|"low", '
    '"escalate": true|false}.'
)


def load():
    tok = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    return tok, model


def route(tok, model, user_request: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=64, do_sample=False)
    text = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    return text


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    user_request = " ".join(sys.argv[1:])
    tok, model = load()
    raw = route(tok, model, user_request)
    print(f"request : {user_request}")
    print(f"decision: {raw}")
    try:
        print("parsed  :", json.dumps(json.loads(raw), indent=2))
    except json.JSONDecodeError:
        print("(model did not return valid JSON — expected for an undertrained v1 on novel phrasings)")


if __name__ == "__main__":
    main()
