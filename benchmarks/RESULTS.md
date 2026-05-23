# Benchmark results

Model: stock public Qwen2.5-14B-Instruct. Hardware: rented H100 SXM 80GB.
Built and measured with the scripts in `quant-bench/`. Raw run logs
(`h100_public_bench.log`, `perplexity_ladder.log`) sit alongside this file.

## Quant ladder

| Quant | Size | bits/weight |
|---|---|---|
| q8_0 | 14.6 GiB | 8.0 |
| q6_K | 11.3 GiB | ~6.6 |
| q5_K_M | 9.8 GiB | ~5.7 |
| q4_K_M | 8.4 GiB | 4.87 |

## Throughput and perplexity

GPU: all layers offloaded (-ngl 99). CPU: 8 threads, no offload. Perplexity: wikitext-2.

| Quant | GPU gen tg128 (t/s) | GPU prompt pp512 (t/s) | CPU gen tg128 (t/s) | perplexity |
|---|---|---|---|---|
| q8_0 | 114.2 | 5320 | 3.82 | 5.9804 |
| q6_K | 101.3 | 4039 | 4.94 | 6.0090 |
| q5_K_M | 117.4 | 4581 | 5.26 | 6.0400 |
| q4_K_M | 126.5 | 4689 | 6.45 | 6.1897 |

q4_K_M is fastest at generation on both GPU and CPU and costs only about 3.5 percent
perplexity over q8_0 while being 43 percent smaller. That is the deploy pick.

## Retrieval (separate, consumer GPU)

BeIR/scifact, BGE-large-en-v1.5, one RTX 2080 (8 GB): recall@10 0.873, recall@5 0.784,
MRR@10 0.712. Same harness on a private 130K-doc legal corpus: recall@10 0.96.
