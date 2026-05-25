# Open-Model Ops on a Budget

### Fine-tune, quantize, benchmark, and serve 14B-class open models, from a rented H100 down to a consumer 3090

One person, self-funded, no cluster. This repo runs the whole open-model loop end to
end on stock public models and reports real numbers at every stage: fine-tune,
quantize, benchmark (throughput and quality), and serve. Every number here is from a
real run, and the scripts reproduce them. It is written for the people pulling GGUFs
off the Hub every day.

Three runnable pieces:
- `quant-bench/` builds a GGUF quant ladder and benchmarks throughput and perplexity on any hardware.
- `finetune/` trains a LoRA adapter and proves it changed the model.
- `retrieval-eval/` reproduces a recall@k retrieval benchmark on a public dataset on one consumer GPU.

All benchmark numbers below are on stock public Qwen2.5-14B-Instruct, measured on a
rented H100 SXM 80GB. Raw logs are in `benchmarks/`.

## The loop

LoRA fine-tune (PEFT) on a rented H100, merge, build a GGUF quant ladder with
llama.cpp, benchmark GPU and CPU throughput and perplexity, then serve the chosen
quant locally on consumer hardware. This repo does every step on a public model.

## Quant ladder (Qwen2.5-14B-Instruct)

| Quant | Size | bits/weight |
|---|---|---|
| q8_0 | 14.6 GiB | 8.0 |
| q6_K | 11.3 GiB | ~6.6 |
| q5_K_M | 9.8 GiB | ~5.7 |
| q4_K_M | 8.4 GiB | 4.87 |

## Throughput and quality, full picture (H100 SXM 80GB)

GPU is all layers offloaded (-ngl 99). CPU is 8 threads, no offload. Perplexity is
wikitext-2, lower is better.

| Quant | GPU gen tg128 (t/s) | GPU prompt pp512 (t/s) | CPU gen tg128 (t/s) | perplexity | PPL cost vs q8_0 |
|---|---|---|---|---|---|
| q8_0 | 114.2 | 5320 | 3.82 | 5.980 | baseline |
| q6_K | 101.3 | 4039 | 4.94 | 6.009 | +0.5% |
| q5_K_M | 117.4 | 4581 | 5.26 | 6.040 | +1.0% |
| q4_K_M | 126.5 | 4689 | 6.45 | 6.190 | +3.5% |

Two takeaways that matter:

- Generation favors the smaller quant. q4_K_M is the fastest at generating tokens on both GPU (126 t/s) and CPU (6.45 t/s), because generation is memory-bandwidth-bound and the smaller weights move less data. Prompt processing (pp512) is compute-bound and roughly flat across the ladder.
- The quality cost is tiny. q4_K_M gives up only about 3.5 percent perplexity versus q8_0 while being 43 percent smaller and the fastest to generate.

## Deploy pick: q4_K_M

Smallest, fastest at generation on both GPU and CPU, and only about 3.5 percent
perplexity over q8_0. For a CPU or edge deploy at about 6 tok/s that is roughly a
100-token answer in 16 seconds.

## What fits where: the rent-versus-own line

The local rig is an RTX 3090 (24 GB) with an RTX 2080 (8 GB) alongside it. One thing
worth getting right, because it is often gotten wrong: these are different generations
with no NVLink, so they do not pool into a single 32 GB space for one model. llama.cpp
can tensor-split a model across both cards, but the clean ceiling for one interactive
model is the 24 GB card.

- 8B and 35B-A3B at Q4 (about 18 to 20 GB): fit the 3090 alone, full GPU, no rental cost.
- 70B and 72B-class at Q4 (about 47 GB): do not fit either card. Spilling to system RAM drops generation to roughly 1 to 3 tok/s, which is not viable interactively.
- Rule of thumb: if it fits the 24 GB card, run it free on local iron. 70B-class and higher precision earn a rented 80 GB card.

Rental reference: H100 SXM 80GB at about $3.00/hr, or A100 80GB at about $1.39/hr when
speed can trade for cost. A full 14B fine-tune plus quant ladder plus benchmark sweep
fits in one short H100 session.

## Fine-tune: the loop, demonstrated

`finetune/train_lora_sft.py` trains a LoRA adapter on a public instruction set
(databricks-dolly-15k) on the H100, and `finetune/gen_compare.py` proves it changed the
model with a before/after generation. A worked example is in
`finetune/before_after_example.md`. The trained adapter is published at
https://huggingface.co/ArgusForge/qwen2.5-14b-dolly-lora. Merged and quantized to
q4_K_M, the fine-tune runs through the exact same serving path benchmarked above.

Training: LoRA r=16, 200 steps, bf16, effective batch 16, loss 1.98 down to 1.21, on
one H100.

## Retrieval holds up on the small card too

`retrieval-eval/run_retrieval_eval.py` embeds a public IR benchmark with BGE-large on a
single GPU and scores recall@k and MRR against the dataset's real relevance judgments.
It runs end to end on a public dataset, so anyone can reproduce a genuine number on
their own hardware. Reproduced result on BeIR/scifact (300 test queries, 5183 docs),
BGE-large-en-v1.5, on one RTX 2080 (8 GB) at batch 32:

| k | recall@k | MRR@k |
|---|---|---|
| 1 | 0.609 | 0.640 |
| 5 | 0.784 | 0.701 |
| 10 | 0.873 | 0.712 |
| 20 | 0.920 | 0.716 |

Clone it and you get these numbers back. On my own private 130K-document legal corpus
the same harness scored recall@10 of 0.96 (recall@5 of 0.92, MRR 0.86). That corpus is
not public, which is exactly why this repo ships the method against data you can
download and rerun. Two different corpora, two real numbers.

## Run it

Quant ladder, throughput, perplexity (point the scripts at a llama.cpp build/bin dir and a gguf dir):

    ./quant-bench/build_quant_ladder.sh <llama.cpp/build/bin> <model-f16.gguf> ./gguf_out
    ./quant-bench/run_llama_bench.sh <llama.cpp/build/bin> ./gguf_out
    ./quant-bench/run_perplexity.sh <llama.cpp/build/bin> ./gguf_out <wiki.test.raw>

Fine-tune and prove it changed the model:

    python finetune/train_lora_sft.py --bsz 1 --grad_accum 16 --max_steps 200
    python finetune/gen_compare.py

Retrieval eval (consumer GPU, a few minutes):

    cd retrieval-eval && pip install -r requirements.txt
    python run_retrieval_eval.py --add_query_instruction --fp16

On an 8 GB card add `--batch_size 32`. Use `--device cpu` to skip the GPU (slower, identical numbers).

## Honest notes

- All throughput and perplexity numbers are from real runs on stock public Qwen2.5-14B-Instruct. Raw logs are in `benchmarks/`. Rerun the scripts on your own hardware for your own numbers.
- The recall@10 of 0.96 is on a private legal corpus. The harness here runs on a public dataset (scifact, 0.87) so it is reproducible. The two are not the same measurement.
- CPU numbers reflect the benchmark host's CPU and are host-dependent by nature.
- Quantization quality was measured with perplexity above. The q4_K_M serving output also held the fine-tune's behavior after merge and quantization (see the before/after).

## How this was built

The work and the decisions are mine. The process is multi-agent by design: I developed
this repo with Claude and Grok in the loop, each cross-checking the other's claims while
I arbitrated. It is the same consensus-then-verify pattern that runs the Sentinel system
this came out of. The method is not decoration. When the three of us disagreed over
whether a 3090 and a 2080 pool into one memory space (they do not, there is no NVLink),
that disagreement is what caught and fixed the error before it shipped.

## Writeups

- [Fine-tuning a 14B on legal and medical benchmarks for about a dollar each](writeups/finetuning-14b-legal-medical.md)
