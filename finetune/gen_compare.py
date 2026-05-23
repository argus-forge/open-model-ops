#!/usr/bin/env python3
"""
Before/after proof: run the same held-out prompt through the base model, then
through the base plus the LoRA adapter, and print both. If the adapter trained,
the two outputs differ. This is the honest evidence that the fine-tune did
something, not just that training exited cleanly.

Usage:
  python gen_compare.py
  python gen_compare.py --prompt "Write a short note declining a meeting."
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def generate(model, tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=200, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/qwen2.5-14b")
    p.add_argument("--adapter", default="/workspace/qwen2.5-14b-dolly-lora")
    p.add_argument("--prompt", default="Explain why the sky is blue to a ten year old.")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    print("\n===== BASE =====")
    print(generate(base, tok, args.prompt))

    adapted = PeftModel.from_pretrained(base, args.adapter)
    print("\n===== BASE + LORA =====")
    print(generate(adapted, tok, args.prompt))


if __name__ == "__main__":
    main()
