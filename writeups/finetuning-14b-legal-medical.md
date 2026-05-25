# Fine-tuning a 14B on legal and medical benchmarks for about a dollar each

I trained two LoRA adapters on a single rented H100 and moved a 14B model's accuracy on two public benchmarks: +10.2 points on CaseHOLD (legal) and +9.6 points on MedMCQA (medical). Each run cost under a dollar of GPU time. The adapters, the base model, and the datasets are all public, so you can reproduce this end to end.

## The result

Base model: Qwen2.5-14B-Instruct. Adapter: LoRA, rank 16, alpha 32, seven target modules (attention plus the three MLP projections), 68.8M trainable parameters, about 0.46% of the model. Training: 8000 examples from each public dataset, 400 steps, bf16, on one H100 SXM 80GB.

Eval: first-token logit scoring over the answer-option letters, held-out set of 500, identical prompts for base and adapter.

| Adapter | Benchmark | Base | Adapter | Gain | Train time |
|---|---|---|---|---|---|
| Medical | MedMCQA | 0.554 | 0.650 | +9.6 pts | ~12 min |
| Legal | CaseHOLD (LexGLUE) | 0.738 | 0.840 | +10.2 pts | ~21 min |

Both gains sit 4 to 5 standard errors above baseline on this eval. At about $3/hr for the H100, that is roughly $0.60 and $1.05 of compute.

## What this proves, and what it does not

It proves one narrow, useful thing. A small LoRA on a 14B, trained on a few thousand public examples for the price of pocket change, moves domain-benchmark accuracy by a real margin, and the whole pipeline runs on commodity rented hardware. No cluster, no data center.

It does not prove these are usable legal or medical tools, and I want to be flat about that. The eval is a single run and seed, n=500, and first-token logit scoring rather than free-text generation. I did not compare against a strong retrieval baseline. The medical adapter is not clinician-grade and the legal adapter is not attorney-grade. Do not use either for an actual medical or legal decision. This is a capability floor: evidence that the customization step works cheaply and reproducibly, and nothing past that.

I am stating the bounds up front on purpose. Most fine-tunes ship with an empty card and a big number. The honest version is the more useful one.

## The recipe

The stack that actually worked, with versions pinned because the bleeding edge breaks: bf16 base, bitsandbytes 4-bit nf4, vanilla transformers with peft and trl. Not Unsloth on this Qwen vocab, and not an AWQ base on current transformers. Both of those are dead ends I hit and backed out of, so you do not have to.

Deploy path: merge the adapter into the base, convert to GGUF, quantize down the ladder. The q4_K_M medical model lands around 8.4 GB and served roughly 139 tok/s on the H100. It runs on CPU as well, slower.

One ceiling worth knowing: a single H100 is the limit for 14B fine-tuning at this config. A 32B run OOM'd on one card. Bigger than 14B means more GPUs or a different quantization strategy.

## Why bother

The interesting part is not the ten points. It is that one person can take an open base model, a public dataset, and a single rented GPU, and produce a measurably better domain model in twenty minutes for a dollar. The work that used to need a team and a budget now fits in an afternoon and loose change. More people should know that floor is this low.

Adapters live at hf.co/ArgusForge. The pipeline and code are at github.com/argus-forge/open-model-ops. Reproduce it, break it, and tell me where it falls over.
