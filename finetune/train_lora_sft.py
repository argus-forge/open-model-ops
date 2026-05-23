#!/usr/bin/env python3
"""
Minimal LoRA SFT demo: fine-tune a local Qwen2.5-14B on a public instruction
dataset (databricks-dolly-15k) with PEFT LoRA in bf16, a few hundred steps.
Saves a LoRA adapter. This is the fine-tune stage of the open-model-ops loop:
fine-tune, then quantize, then bench, then serve.

Kept deliberately simple and version-robust: plain transformers Trainer plus
PEFT, no trl, no quantization (an 80 GB card holds a 14B LoRA in bf16 easily).

Usage:
  python train_lora_sft.py
  python train_lora_sft.py --max_steps 300 --n_samples 3000
"""
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/qwen2.5-14b")
    p.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    p.add_argument("--out", default="/workspace/qwen2.5-14b-dolly-lora")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--n_samples", type=int, default=3000)
    p.add_argument("--bsz", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    return p.parse_args()


def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw = load_dataset(args.dataset, split="train")
    if args.n_samples and args.n_samples < len(raw):
        raw = raw.select(range(args.n_samples))

    def to_text(ex):
        instr = (ex.get("instruction") or "").strip()
        ctx = (ex.get("context") or "").strip()
        resp = (ex.get("response") or "").strip()
        user = instr if not ctx else instr + "\n\n" + ctx
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": resp},
        ]
        return {"text": tok.apply_chat_template(messages, tokenize=False)}

    ds = raw.map(to_text, remove_columns=raw.column_names)

    def tokenize(ex):
        return tok(ex["text"], truncation=True, max_length=args.max_len)

    ds = ds.map(tokenize, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.max_steps,
        bf16=True,
        report_to="none",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=collator
    )
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[*] adapter saved to {args.out}")


if __name__ == "__main__":
    main()
