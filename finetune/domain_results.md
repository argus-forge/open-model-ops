# Domain fine-tune results: targeted-task lift on ownable hardware

The `finetune/` demo shows the loop runs end to end. This is that same loop pointed at two real targeted tasks, to answer a sharper question: does a small LoRA on a 14B model you can actually own buy a measurable lift on a domain task, and does the result still deploy on consumer hardware? Both, yes.

## Setup

- Base: Qwen2.5-14B-Instruct (fits one H100, quantizes onto a 24GB consumer card).
- Fine-tune: LoRA, r=16, alpha=32, about 68.8M trainable (0.46% of the model), bf16, 400 steps, effective batch 16, 8000 training examples, lr 2e-4 cosine.
- Eval: first-token logit scoring over the option letters, on a held-out validation split, n=500, identical questions for base and adapter (apples to apples).
- Same harness for both tasks, dataset swapped. Public datasets only.

## Results

| Task | Dataset | Options | Base acc | Fine-tuned acc | Absolute lift | Relative | Train time |
|---|---|---|---|---|---|---|---|
| Medical MCQA | MedMCQA | 4 | 0.554 (277/500) | 0.650 (325/500) | +9.6 pts | +17.3% | ~12 min |
| Legal holding | CaseHOLD (LexGLUE) | 5 | 0.738 (369/500) | 0.840 (420/500) | +10.2 pts | +13.8% | ~21 min |

Both lifts are roughly 4 to 5 standard errors above their baseline at n=500, so neither is sampling noise.

Adapters:
- https://huggingface.co/ArgusForge/qwen2.5-14b-medmcqa-lora
- https://huggingface.co/ArgusForge/qwen2.5-14b-casehold-lora

## Deploy (medical lane)

The medical adapter was merged into the base, converted to f16 GGUF, and quantized to q4_K_M: about 8.4 GB (4.87 bits per weight), which fits a 24GB RTX 3090 with two thirds of the card free and fits a 12GB card. Served from the quantized file it generated coherent on-domain output at roughly 139 tokens/sec generation and 495 tokens/sec prompt on an H100. That closes the full loop for the medical task: baseline, fine-tune, lift, quantize, serve on hardware you can own.

## Honest notes

- These absolute numbers are not a claim of beating frontier models across all of medicine or law. The claim is narrower and more useful: on a targeted task, a small fine-tune on an ownable 14B buys a real, repeatable gain.
- Single run, single seed, n=500 per eval. A product-grade claim would want multi-seed confirmation, the full validation split, and an external held-out set.
- Quantized accuracy was not re-measured. The lift numbers are the bf16 adapter, and a deploy claim should re-score the q4_K_M file.
- Free-text generation from the medical model is not clinician-grade and produced at least one clinically incorrect statement in testing. Neither model is advice-grade. A deployable product in either domain needs a domain validation regime first. These are research artifacts, not medical or legal advice.

## Reproduce

Public base (Qwen2.5-14B-Instruct) and public datasets (MedMCQA, CaseHOLD via LexGLUE). The eval is first-token logit scoring over the option letters on a held-out n=500 split, with identical prompts for base and adapter. Swap the dataset, keep the harness.
