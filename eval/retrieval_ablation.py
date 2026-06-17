#!/usr/bin/env python3
"""Ablation del retrieval (FASE 3): confronto VETTORIALE vs HYBRID, senza LLM.

Per ogni domanda del dataset con risposta attesa e documenti attesi, esegue la
sola pipeline di recupero (candidati ibridi -> reranking euristico -> filtro
metadata -> top-k) in due modalità:
- `vector`: solo arm vettoriale (riproduce la baseline pre-FASE 3);
- `hybrid`: arm vettoriale + BM25 fusi con RRF.

Riporta, per ciascuna modalità: retrieval-hit, rango medio del primo documento
atteso nella lista ri-ordinata, e quante domande migliorano/peggiorano/restano
uguali passando da vector a hybrid. NON richiede Ollama (nessuna generazione).

Uso:
    python eval/retrieval_ablation.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.CRITICAL)

from langchain_community.vectorstores import Chroma  # noqa: E402

from config import (  # noqa: E402
    CHROMA_PERSIST_DIRECTORY,
    MAX_CONTEXT_DOCUMENTS,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
)
from database import _build_chroma_settings, _build_embeddings  # noqa: E402
from intent import infer_query_intent  # noqa: E402
from neural_reranker import CrossEncoderReranker  # noqa: E402
from rag_types import RagTrace  # noqa: E402
from reranking import filter_documents_by_course, rerank_documents  # noqa: E402
from retrieval import build_bm25_index, hybrid_retrieve  # noqa: E402

DATASET = os.path.join(ROOT, "eval", "questions_baseline.jsonl")
REPORTS_DIR = os.path.join(ROOT, "eval", "reports")


def load_dataset(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def basename(doc):
    return os.path.basename((doc.metadata or {}).get("source", ""))


def retrieve(vector_db, bm25_index, question, intent, use_bm25, neural_reranker=None):
    trace = RagTrace()
    docs = hybrid_retrieve(vector_db, bm25_index, question, intent, trace, use_bm25=use_bm25)
    docs = rerank_documents(question, docs, intent, trace)
    docs = filter_documents_by_course(docs, intent)
    if neural_reranker is not None:
        docs = neural_reranker.rerank(question, docs, RERANKER_TOP_N)
    return docs


def gold_rank(docs, expected_docs):
    """Rango (1-based) del primo documento atteso nella lista ri-ordinata; None se assente."""
    for i, doc in enumerate(docs):
        if basename(doc) in expected_docs:
            return i + 1
    return None


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Ablation del retrieval UniLaw Agent")
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Confronta hybrid vs hybrid+cross-encoder (carica il reranker neurale).",
    )
    args = parser.parse_args()

    print(f"Caricamento indice ChromaDB da {CHROMA_PERSIST_DIRECTORY} ...")
    vdb = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=_build_embeddings(),
        client_settings=_build_chroma_settings(),
    )
    if not vdb.get().get("ids"):
        raise SystemExit("Indice vuoto: ricostruisci la knowledge base.")

    bm25 = build_bm25_index(vdb)
    print(f"Indice BM25: {len(bm25)} chunk.\n")

    # Definisce le due configurazioni a confronto (colonna A vs colonna B).
    if args.reranker:
        reranker = CrossEncoderReranker(RERANKER_MODEL_NAME)
        t0 = time.time()
        ok = reranker.available()  # forza il caricamento e misura il costo una tantum
        print(
            f"Reranker neurale: {'caricato' if ok else 'NON disponibile (fallback euristico)'} "
            f"in {time.time() - t0:.1f}s\n"
        )
        label_a, label_b = "hybrid", "hyb+ce"
        cfg_a = dict(use_bm25=True, neural_reranker=None)
        cfg_b = dict(use_bm25=True, neural_reranker=reranker)
    else:
        label_a, label_b = "vector", "hybrid"
        cfg_a = dict(use_bm25=False, neural_reranker=None)
        cfg_b = dict(use_bm25=True, neural_reranker=None)

    dataset = [
        q
        for q in load_dataset(DATASET)
        if q["expected_behavior"] == "answer" and q.get("expected_docs")
    ]

    rows = []
    improved = worsened = equal = 0
    hit_a = hit_b = 0

    for q in dataset:
        intent = infer_query_intent(q["question"], {})
        exp = q["expected_docs"]

        docs_a = retrieve(vdb, bm25, q["question"], intent, **cfg_a)
        docs_b = retrieve(vdb, bm25, q["question"], intent, **cfg_b)

        r_a = gold_rank(docs_a, exp)
        r_b = gold_rank(docs_b, exp)

        topk_a = [basename(d) for d in docs_a[:MAX_CONTEXT_DOCUMENTS]]
        topk_b = [basename(d) for d in docs_b[:MAX_CONTEXT_DOCUMENTS]]

        in_a = any(d in topk_a for d in exp)
        in_b = any(d in topk_b for d in exp)
        hit_a += in_a
        hit_b += in_b

        if r_a is not None and r_b is not None:
            if r_b < r_a:
                improved += 1
            elif r_b > r_a:
                worsened += 1
            else:
                equal += 1

        rows.append(
            {
                "id": q["id"],
                f"rank_{label_a}": r_a,
                f"rank_{label_b}": r_b,
                f"hit_{label_a}": in_a,
                f"hit_{label_b}": in_b,
                "topk_identical": topk_a == topk_b,
            }
        )

    n = len(dataset)
    header = f"{'id':>4} | rank({label_a}) | rank({label_b}) | hit({label_a}) | hit({label_b}) | top-k uguale"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['id']:>4} | {str(r[f'rank_{label_a}']):>11} | {str(r[f'rank_{label_b}']):>11} | "
            f"{('sì' if r[f'hit_{label_a}'] else 'no'):>9} | {('sì' if r[f'hit_{label_b}'] else 'no'):>9} | "
            f"{'sì' if r['topk_identical'] else 'NO'}"
        )

    print("\n=== SOMMARIO ABLATION (no LLM) ===")
    print(f"  domande valutate (answer + doc attesi): {n}")
    print(f"  retrieval-hit  {label_a}: {hit_a}/{n} ({hit_a / n:.3f})")
    print(f"  retrieval-hit  {label_b}: {hit_b}/{n} ({hit_b / n:.3f})")
    print(f"  rango gold migliorato/uguale/peggiorato ({label_b} vs {label_a}): "
          f"{improved}/{equal}/{worsened}")
    identical = sum(1 for r in rows if r["topk_identical"])
    print(f"  top-k identico tra le due configurazioni: {identical}/{n}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(
        REPORTS_DIR,
        "retrieval_ablation_reranker.json" if args.reranker else "retrieval_ablation.json",
    )
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "comparison": [label_a, label_b],
                "n": n,
                f"hit_{label_a}": hit_a,
                f"hit_{label_b}": hit_b,
                "improved": improved,
                "equal": equal,
                "worsened": worsened,
                "topk_identical": identical,
                "rows": rows,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
