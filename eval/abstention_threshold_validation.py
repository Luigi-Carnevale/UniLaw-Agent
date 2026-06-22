#!/usr/bin/env python3
"""Validazione held-out della soglia di astensione (Ciclo 2 — FASE 6).

La distinzione fra le due cause di astensione *governate da una soglia* —
`fuori_dominio` (le fonti non coprono i termini della query) vs
`evidenza_insufficiente` (fonti pertinenti ma risposta assente) — dipende da
`ABSTENTION_OOD_MAX_STRENGTH` (config). La soglia era stata fissata a mano (0,37)
sugli stessi casi su cui veniva poi misurata: una stima ottimistica della sua
bontà.

Questo harness separa **calibrazione** e **validazione**:

1. misura la `retrieval_strength` di ogni negativo *threshold-relevant* eseguendo
   la sola pipeline di recupero (Chroma + BM25 + RRF, ri-ordinamento, filtro
   corso). NON richiede Ollama: la forza non dipende dalla generazione;
2. **calibra** la soglia dai soli negativi storici (set di calibrazione) con una
   regola esplicita (`abstention.calibrate_ood_threshold`);
3. **valida** la soglia sui negativi *held-out* (q34–q39, riservati in FASE 4 e
   mai usati per calibrare), riportando l'accuratezza fuori campione.

Riferimento per i due insiemi: il blocco held-out q34–q39 è quello introdotto
dalla Ciclo 2 — FASE 4 (cfr. `tests/test_eval_dataset.py`).

Uso:
    python eval/abstention_threshold_validation.py
Report salvato in eval/reports/abstention_threshold_validation.{json,md}.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from langchain_community.vectorstores import Chroma  # noqa: E402

from abstention import (  # noqa: E402
    INSUFFICIENT_EVIDENCE,
    OUT_OF_DOMAIN,
    calibrate_ood_threshold,
    classify_by_strength,
    retrieval_strength,
    semantic_retrieval_strength,
    threshold_accuracy,
)
from agent import UniLawResponder  # noqa: E402
from config import (  # noqa: E402
    ABSTENTION_OOD_MAX_STRENGTH,
    ABSTENTION_OOD_SEMANTIC_MAX_STRENGTH,
    CHROMA_PERSIST_DIRECTORY,
    DEFAULT_MODEL_NAME,
)
from database import _build_chroma_settings, _build_embeddings  # noqa: E402

DATASET = os.path.join(ROOT, "eval", "questions_baseline.jsonl")
REPORTS_DIR = os.path.join(ROOT, "eval", "reports")

# Cause di astensione la cui distinzione è governata dalla soglia.
THRESHOLD_REASONS = {OUT_OF_DOMAIN, INSUFFICIENT_EVIDENCE}

# Negativi held-out riservati alla validazione (Ciclo 2 — FASE 4), mai usati per
# calibrare: gli stessi id verificati da tests/test_eval_dataset.py.
HELDOUT_IDS = {f"q{n}" for n in range(34, 40)}


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def measure_strength(
    responder: UniLawResponder, question: str, embedder
) -> tuple[float, float, int]:
    """Forza lessicale e semantica della domanda sulle fonti recuperate (no LLM).

    Restituisce `(lexical_strength, semantic_strength, n_sources)`. La
    `retrieval_strength` semantica usa l'embedder del vector store (lo stesso del
    retrieval); con `embedder=None` vale 0.0 (la sezione semantica resta vuota).
    """
    intent = responder._infer_query_intent(question, {})
    docs = responder._retrieve_documents(question, intent)
    sources = responder._prepare_sources(docs) if docs else []
    lexical = retrieval_strength(question, sources)
    semantic = semantic_retrieval_strength(question, sources, embedder) if embedder else 0.0
    return lexical, semantic, len(sources)


def evaluate_split(rows: list[dict], threshold: float, key: str = "strength") -> dict:
    """Accuratezza e dettaglio per riga della classificazione a una data soglia.

    `key` seleziona quale forza usare (`strength` lessicale o `semantic_strength`).
    """
    detail = []
    for r in rows:
        predicted = classify_by_strength(r[key], threshold)
        detail.append(
            {
                "id": r["id"],
                "strength": round(r[key], 4),
                "expected": r["expected"],
                "predicted": predicted,
                "ok": predicted == r["expected"],
            }
        )
    labeled = [(r[key], r["expected"]) for r in rows]
    return {"accuracy": threshold_accuracy(labeled, threshold), "detail": detail}


def main() -> None:
    print(f"Caricamento indice ChromaDB da {CHROMA_PERSIST_DIRECTORY} ...")
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=_build_embeddings(),
        client_settings=_build_chroma_settings(),
    )
    if not db.get().get("ids"):
        raise SystemExit("Indice vuoto: ricostruisci la knowledge base.")
    responder = UniLawResponder(db)
    # Embedder del vector store per la forza SEMANTICA (Ciclo 2 — FASE 13); è lo
    # stesso modello usato dal retrieval, nessun download/caricamento aggiuntivo.
    embedder = responder._embedder_from_vector_db()

    # Seleziona i negativi threshold-relevant e misura le loro retrieval strength
    # (lessicale e semantica) con la sola pipeline di recupero (no LLM).
    calibration, heldout = [], []
    for item in load_dataset(DATASET):
        reason = item.get("expected_abstention_reason")
        if reason not in THRESHOLD_REASONS:
            continue
        lexical, semantic, n_sources = measure_strength(responder, item["question"], embedder)
        row = {
            "id": item["id"],
            "question": item["question"],
            "expected": reason,
            "strength": lexical,
            "semantic_strength": semantic,
            "n_sources": n_sources,
        }
        (heldout if item["id"] in HELDOUT_IDS else calibration).append(row)

    if not calibration or not heldout:
        raise SystemExit(
            "Servono negativi threshold-relevant sia nel set di calibrazione sia "
            "in quello held-out."
        )

    def analyze(key: str, config_threshold: float) -> dict:
        """Calibra dai soli negativi storici e valida sugli held-out per una data forza."""
        labeled_cal = [(r[key], r["expected"]) for r in calibration]
        calibrated = calibrate_ood_threshold(labeled_cal)
        cal_eval = evaluate_split(calibration, calibrated, key)
        # Validazione held-out con DUE soglie: quella di config (in uso) e quella
        # appena calibrata, per mostrarne la coerenza fuori campione.
        heldout_config = evaluate_split(heldout, config_threshold, key)
        heldout_calibrated = evaluate_split(heldout, calibrated, key)
        return {
            "config_threshold": config_threshold,
            "calibrated": calibrated,
            "calibration": cal_eval,
            "heldout_config": heldout_config,
            "heldout_calibrated": heldout_calibrated,
        }

    lex = analyze("strength", ABSTENTION_OOD_MAX_STRENGTH)
    sem = analyze("semantic_strength", ABSTENTION_OOD_SEMANTIC_MAX_STRENGTH)

    summary = {
        "config_threshold": lex["config_threshold"],
        "calibrated_threshold": round(lex["calibrated"], 4),
        "calibration_ids": [r["id"] for r in calibration],
        "heldout_ids": [r["id"] for r in heldout],
        "calibration_accuracy": lex["calibration"]["accuracy"],
        "heldout_accuracy_config_threshold": lex["heldout_config"]["accuracy"],
        "heldout_accuracy_calibrated_threshold": lex["heldout_calibrated"]["accuracy"],
        # Sezione semantica (Ciclo 2 — FASE 13): stessi insiemi, forza per embedding.
        "semantic_config_threshold": sem["config_threshold"],
        "semantic_calibrated_threshold": round(sem["calibrated"], 4),
        "semantic_calibration_accuracy": sem["calibration"]["accuracy"],
        "semantic_heldout_accuracy_config_threshold": sem["heldout_config"]["accuracy"],
        "semantic_heldout_accuracy_calibrated_threshold": sem["heldout_calibrated"]["accuracy"],
    }

    # --- stampa a console ---------------------------------------------------
    def _print_split(title, rows):
        print(title)
        print(f"  {'id':>4} | {'strength':>8} | {'attesa':>22} | {'predetta':>22} | ok")
        for d in rows:
            print(
                f"  {d['id']:>4} | {d['strength']:>8.4f} | {d['expected']:>22} | "
                f"{d['predicted']:>22} | {'sì' if d['ok'] else 'NO'}"
            )

    def _print_analysis(label, an):
        print(f"\n=== {label} ===")
        print(f"  soglia in config:                           {an['config_threshold']}")
        print(f"  soglia calibrata sui soli negativi storici: {an['calibrated']:.4f}")
        _print_split(f"Calibrazione (soglia {an['calibrated']:.4f}):", an["calibration"]["detail"])
        print(f"  accuratezza calibrazione: {an['calibration']['accuracy']}")
        _print_split(
            f"Held-out (soglia di config {an['config_threshold']}):",
            an["heldout_config"]["detail"],
        )
        print(f"  accuratezza held-out @ config:     {an['heldout_config']['accuracy']}")
        print(f"  accuratezza held-out @ calibrata:  {an['heldout_calibrated']['accuracy']}")

    print("\n############ VALIDAZIONE HELD-OUT DELLA SOGLIA DI ASTENSIONE ############")
    print(f"  set di calibrazione: {summary['calibration_ids']}")
    print(f"  set held-out:        {summary['heldout_ids']}")
    _print_analysis("LESSICALE (overlap di token) — FASE 6", lex)
    _print_analysis("SEMANTICA (similarità di embedding) — FASE 13", sem)

    # --- report -------------------------------------------------------------
    os.makedirs(REPORTS_DIR, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": DEFAULT_MODEL_NAME,
        "note": "retrieval deterministico; nessun LLM coinvolto nella misura della strength",
        "summary": summary,
        # Sezioni LESSICALI (nomi storici, FASE 6 — non rinominare: usati dai test).
        "calibration": lex["calibration"]["detail"],
        "heldout_config_threshold": lex["heldout_config"]["detail"],
        "heldout_calibrated_threshold": lex["heldout_calibrated"]["detail"],
        # Sezioni SEMANTICHE (FASE 13).
        "semantic_calibration": sem["calibration"]["detail"],
        "semantic_heldout_config_threshold": sem["heldout_config"]["detail"],
        "semantic_heldout_calibrated_threshold": sem["heldout_calibrated"]["detail"],
    }
    json_path = os.path.join(REPORTS_DIR, "abstention_threshold_validation.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    md_path = os.path.join(REPORTS_DIR, "abstention_threshold_validation.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Validazione held-out della soglia di astensione (Ciclo 2 — FASE 6 + 13)\n\n")
        fh.write(f"- Generato: {payload['generated_at']}\n")
        fh.write(f"- Set di calibrazione: {summary['calibration_ids']}\n")
        fh.write(f"- Set held-out (mai visto in calibrazione): {summary['heldout_ids']}\n\n")

        def _write_analysis(title, an, config_name):
            fh.write(f"## {title}\n\n")
            fh.write(f"- Soglia in config (`{config_name}`): **{an['config_threshold']}**\n")
            fh.write(
                f"- Soglia calibrata sui soli negativi storici: **{an['calibrated']:.4f}**\n\n"
            )
            fh.write("| Insieme | Soglia | Accuratezza |\n|---|---|---|\n")
            fh.write(
                f"| calibrazione | {an['calibrated']:.4f} | {an['calibration']['accuracy']} |\n"
            )
            fh.write(
                f"| held-out | {an['config_threshold']} (config) | "
                f"{an['heldout_config']['accuracy']} |\n"
            )
            fh.write(
                f"| held-out | {an['calibrated']:.4f} (calibrata) | "
                f"{an['heldout_calibrated']['accuracy']} |\n\n"
            )
            for sub, rows in (
                ("Calibrazione", an["calibration"]["detail"]),
                ("Held-out (soglia di config)", an["heldout_config"]["detail"]),
            ):
                fh.write(f"### {sub}\n\n")
                fh.write("| id | strength | causa attesa | causa predetta | ok |\n")
                fh.write("|---|---|---|---|:--:|\n")
                for d in rows:
                    fh.write(
                        f"| {d['id']} | {d['strength']} | {d['expected']} | "
                        f"{d['predicted']} | {'✓' if d['ok'] else '✗'} |\n"
                    )
                fh.write("\n")

        _write_analysis(
            "Forza LESSICALE (overlap di token) — FASE 6",
            lex,
            "ABSTENTION_OOD_MAX_STRENGTH",
        )
        _write_analysis(
            "Forza SEMANTICA (similarità di embedding) — FASE 13",
            sem,
            "ABSTENTION_OOD_SEMANTIC_MAX_STRENGTH",
        )

    print(f"\nReport salvati:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
