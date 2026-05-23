#!/usr/bin/env python3
"""
Reproducible retrieval-eval harness.

Embeds a public IR benchmark with a sentence-transformer and scores recall@k and
MRR@k against the dataset's real relevance judgments (qrels). Runs end to end on a
single consumer GPU. Default dataset is BeIR/scifact, pulled from the Hugging Face Hub.

This is the public, reproducible version of the harness. On my private 130K-document
legal corpus the same method scored recall@10 of 0.96; that corpus is not public, so
this script demonstrates the method on data anyone can download and rerun.

Usage:
  python run_retrieval_eval.py
  python run_retrieval_eval.py --dataset BeIR/nfcorpus --add_query_instruction
"""
import argparse
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# bge-*-en-v1.5 models expect this instruction on the query side only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def parse_args():
    p = argparse.ArgumentParser(description="Reproducible recall@k retrieval eval.")
    p.add_argument("--dataset", default="BeIR/scifact",
                   help="BeIR-style dataset repo on the HF Hub (has 'corpus' and 'queries' configs).")
    p.add_argument("--qrels_repo", default=None,
                   help="Qrels repo. Defaults to '<dataset>-qrels'.")
    p.add_argument("--qrels_split", default="test")
    p.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--ks", default="1,5,10,20")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--add_query_instruction", action="store_true",
                   help="Prepend the BGE query instruction (recommended for bge-*-en-v1.5).")
    p.add_argument("--fp16", action="store_true",
                   help="Load the model in half precision. Recommended on small GPUs (fits BGE-large on an 8 GB card).")
    return p.parse_args()


def load_corpus(name):
    ds = load_dataset(name, "corpus")["corpus"]
    ids, texts = [], []
    for row in ds:
        ids.append(str(row["_id"]))
        title = (row.get("title") or "").strip()
        body = (row.get("text") or "").strip()
        texts.append((title + " " + body).strip())
    return ids, texts


def load_queries(name):
    ds = load_dataset(name, "queries")["queries"]
    qid_to_text = {}
    for row in ds:
        qid_to_text[str(row["_id"])] = (row.get("text") or "").strip()
    return qid_to_text


def load_qrels(repo, split):
    ds = load_dataset(repo)[split]
    qrels = {}
    for row in ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row["score"])
        if score > 0:
            qrels.setdefault(qid, set()).add(did)
    return qrels


def encode(model, texts, batch_size):
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors, so dot product is cosine
        show_progress_bar=True,
    ).astype(np.float32)


def main():
    args = parse_args()
    ks = sorted(int(k) for k in args.ks.split(","))
    maxk = max(ks)
    qrels_repo = args.qrels_repo or (args.dataset + "-qrels")

    print(f"[*] dataset={args.dataset}  qrels={qrels_repo}:{args.qrels_split}  model={args.model}")
    corpus_ids, corpus_texts = load_corpus(args.dataset)
    qid_to_text = load_queries(args.dataset)
    qrels = load_qrels(qrels_repo, args.qrels_split)

    # Keep only queries that have at least one judged-relevant doc in this split.
    eval_qids = [q for q in qrels.keys() if q in qid_to_text]
    query_texts = [qid_to_text[q] for q in eval_qids]
    if args.add_query_instruction:
        query_texts = [QUERY_INSTRUCTION + t for t in query_texts]

    print(f"[*] corpus={len(corpus_ids)} docs  queries={len(eval_qids)}  loading model...")
    model = SentenceTransformer(args.model, device=args.device)
    if args.fp16 and "cuda" in args.device:
        model = model.half()

    print("[*] encoding corpus")
    corpus_emb = encode(model, corpus_texts, args.batch_size)
    print("[*] encoding queries")
    query_emb = encode(model, query_texts, args.batch_size)

    print("[*] scoring")
    sim = query_emb @ corpus_emb.T  # (num_queries, num_docs), cosine

    recall = {k: 0.0 for k in ks}
    mrr = {k: 0.0 for k in ks}
    nq = 0
    for i, qid in enumerate(eval_qids):
        rel = qrels[qid]
        nq += 1
        scores = sim[i]
        top = np.argpartition(-scores, min(maxk, len(scores) - 1))[:maxk]
        top = top[np.argsort(-scores[top])]
        ranked = [corpus_ids[j] for j in top]
        for k in ks:
            topk = ranked[:k]
            hits = sum(1 for d in topk if d in rel)
            recall[k] += hits / len(rel)
            rr = 0.0
            for rank, d in enumerate(topk, start=1):
                if d in rel:
                    rr = 1.0 / rank
                    break
            mrr[k] += rr

    for k in ks:
        recall[k] /= nq
        mrr[k] /= nq

    print()
    print(f"Results on {args.dataset} ({nq} queries, {len(corpus_ids)} docs), model {args.model}")
    print("  k     recall@k    MRR@k")
    for k in ks:
        print(f"  {k:<4}  {recall[k]:.4f}     {mrr[k]:.4f}")


if __name__ == "__main__":
    main()
