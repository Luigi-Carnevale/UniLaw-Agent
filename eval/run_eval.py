#!/usr/bin/env python3
"""Harness di valutazione baseline per UniLaw Agent (FASE 1).

Carica l'indice ChromaDB persistito, esegue ogni domanda del dataset attraverso
`UniLawResponder` e calcola metriche di baseline sul comportamento attuale, PRIMA
di qualsiasi miglioramento del retrieval.

Metriche calcolate
------------------
- behavior_accuracy   : esito previsto (answer/abstain/clarify/unknown_course)
- course_accuracy     : corso rilevato dall'intent == corso atteso
- topic_accuracy      : argomento rilevato dall'intent == argomento atteso
- retrieval_hit_rate  : almeno un documento atteso tra le fonti selezionate
                        (solo per le domande con risposta attesa e doc attesi)
- citation_hit_rate   : almeno un documento atteso tra quelli citati in risposta
- abstention_rate     : sulle domande "negative" (out_of_domain/no_answer/ambiguous),
                        frazione in cui il sistema NON ha fabbricato una risposta

Note sull'esecuzione
--------------------
- Le domande deterministiche e di astensione NON richiedono Ollama.
- Le domande che cadono sul modello (ramo di generazione) richiedono Ollama
  attivo; se Ollama non risponde, l'esito viene marcato `llm_unavailable` e quelle
  domande sono escluse dalle metriche di correttezza ma conteggiate a parte.

Uso
---
    python eval/run_eval.py                 # esegue tutto il dataset
    python eval/run_eval.py --limit 9       # solo le prime 9 (smoke test, no LLM)
    python eval/run_eval.py --questions eval/questions_baseline.jsonl
    python eval/run_eval.py --repeat 5      # 5 esecuzioni: media±σ delle metriche
I report vengono salvati in eval/reports/ (JSON + Markdown).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import warnings
from datetime import datetime

# --- Rende importabile la root del progetto ---------------------------------
# Solo l'aggiunta della root al path resta a livello di modulo: le dipendenze
# pesanti (Chroma, agent, database) e gli effetti collaterali globali (chdir,
# logging, warnings) sono spostati dentro `main()`/`load_vector_db()`. Così
# questo modulo è importabile dai test per le sole funzioni di scoring, offline
# e senza Ollama (Ciclo 2 — FASE 5).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from abstention import (  # noqa: E402  (import leggero: serve allo scoring)
    AMBIGUOUS,
    INSUFFICIENT_EVIDENCE,
    OUT_OF_DOMAIN,
    OUT_OF_DOMAIN_COURSE,
    WEAK_RETRIEVAL,
)


REPORTS_DIR = os.path.join(ROOT, "eval", "reports")

# Marcatori testuali stabili usati dal codice per i rami senza LLM.
MARK_UNKNOWN_COURSE = "Non posso rispondere in modo affidabile su questo corso"
MARK_CLARIFY = "La domanda è ambigua"
MARK_NO_EVIDENCE = "Non lo so in base ai documenti disponibili"
MARK_LLM_DOWN = (
    "Ollama non è raggiungibile",
    "il modello Ollama",
    "Errore durante la chiamata a Ollama",
)

# Frasi che indicano un'astensione anche quando la risposta passa dal modello.
ABSTAIN_PHRASES = (
    "non lo so in base ai documenti disponibili",
    "non sono presenti informazioni",
    "non è presente nei documenti",
    "non risulta nei documenti",
    "i documenti disponibili non",
)

# Cattura il nome file completo dopo l'etichetta [F#], gestendo anche le doppie
# estensioni (es. "bando-erasmus25-26.pdf.pdf"): il lookahead forza il ".pdf"
# finale a essere quello immediatamente prima di virgola/spazio/fine stringa.
PDF_RE = re.compile(r"\[F\d+\]\s*(.+?\.pdf)(?=[,\s]|$)", re.IGNORECASE)

NEGATIVE_CATEGORIES = {"out_of_domain", "no_answer", "ambiguous"}

# Mappa la causa di astensione strutturata (`trace.abstention_reason`, impostata
# dall'agente su ogni ramo) all'esito osservabile. È la fonte di verità preferita
# dallo scoring: vedi `classify_behavior` (Ciclo 2 — FASE 5).
ABSTAIN_REASONS = {WEAK_RETRIEVAL, OUT_OF_DOMAIN, INSUFFICIENT_EVIDENCE}

# Metriche-tasso aggregabili tra esecuzioni ripetute per stimare la banda di
# rumore del modello locale (Ciclo 2 — FASE 7). Sono le stesse metriche mostrate
# nei report di singola esecuzione; le deterministiche (course/topic/retrieval)
# avranno σ≈0, le generative (behavior/citation/abstention) la variabilità reale.
RATE_METRIC_KEYS = (
    "behavior_accuracy",
    "course_accuracy",
    "topic_accuracy",
    "retrieval_hit_rate",
    "citation_hit_rate",
    "abstention_rate",
    "abstention_reason_accuracy",
)


def load_dataset(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_vector_db():
    # Import pesanti differiti: questo modulo deve restare importabile offline
    # dai test (Ciclo 2 — FASE 5) senza caricare Chroma/embeddings.
    from langchain_community.vectorstores import Chroma
    from config import CHROMA_PERSIST_DIRECTORY
    from database import _build_chroma_settings, _build_embeddings

    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=_build_embeddings(),
        client_settings=_build_chroma_settings(),
    )
    got = db.get()
    if not got.get("ids"):
        raise SystemExit(
            "Indice ChromaDB vuoto o assente. Avvia l'app e ricostruisci la "
            "knowledge base prima di lanciare la valutazione."
        )
    return db


def filenames_in(text: str) -> list[str]:
    return [m.strip() for m in PDF_RE.findall(text or "")]


def classify_behavior_from_text(answer: str) -> str:
    """Classificazione di *fallback* basata sui soli marcatori testuali stabili.

    Usata quando il trace non porta un segnale di astensione strutturato (es.
    formulazioni non ancora coperte dal layer di astensione). I marcatori dei
    rami senza LLM (`MARK_UNKNOWN_COURSE`/`MARK_CLARIFY`) sono esatti e stabili;
    le `ABSTAIN_PHRASES` sono un sottoinsieme prudente delle frasi di incertezza.
    """
    low = (answer or "").lower()
    if MARK_UNKNOWN_COURSE in answer:
        return "unknown_course"
    if MARK_CLARIFY in answer:
        return "clarify"
    if any(m.lower() in low for m in MARK_LLM_DOWN):
        return "llm_unavailable"
    if any(p in low for p in ABSTAIN_PHRASES):
        return "abstain"
    return "answer"


def classify_behavior(answer: str, trace) -> tuple[str, str]:
    """Classifica l'esito dai segnali STRUTTURATI del trace (Ciclo 2 — FASE 5).

    Restituisce `(esito, fonte)` con `fonte ∈ {"trace", "text"}` per tracciabilità.

    Motivazione: dedurre answer/abstain/clarify dai soli marcatori testuali è
    fragile — basta che il modello cambi formulazione perché un'astensione reale
    venga letta come "answer". L'agente, invece, imposta `trace.abstention_reason`
    su OGNI ramo di astensione, sia deterministico (corso fuori dominio, ambigua,
    retrieval debole) sia generativo (la sua `is_abstention` usa un insieme di
    marcatori più ampio di queste `ABSTAIN_PHRASES`). Il trace è quindi la fonte
    più affidabile e stabile al variare del wording.

    Unica eccezione: l'LLM non raggiungibile. La chiamata fallisce *prima* di
    impostare una causa, quindi quel caso resta riconosciuto dal testo.
    """
    # 1. LLM non disponibile: non lascia un segnale strutturato → solo testo.
    low = (answer or "").lower()
    if any(m.lower() in low for m in MARK_LLM_DOWN):
        return "llm_unavailable", "text"

    # 2. Segnale strutturato del trace: la causa di astensione impostata dall'agente.
    reason = getattr(trace, "abstention_reason", "") or ""
    if reason == OUT_OF_DOMAIN_COURSE:
        return "unknown_course", "trace"
    if reason == AMBIGUOUS:
        return "clarify", "trace"
    if reason in ABSTAIN_REASONS:
        return "abstain", "trace"
    if reason:
        # Causa presente ma non mappata: comunque un'astensione.
        return "abstain", "trace"

    # 3. Nessuna astensione nel trace → risposta, con un fallback testuale di
    #    sicurezza per formulazioni di astensione non ancora riflesse nel trace.
    text_verdict = classify_behavior_from_text(answer)
    if text_verdict != "answer":
        return text_verdict, "text"
    return "answer", "trace"


def evaluate_question(responder: UniLawResponder, item: dict) -> dict:
    answer = responder.answer(
        item["question"],
        memory={},
        show_interpretation=False,
        show_confidence=False,
    )
    trace = responder.last_trace

    selected = []
    for label in trace.selected_sources or []:
        selected.extend(filenames_in(label))
    cited = filenames_in(answer)

    predicted, predicted_source = classify_behavior(answer, trace)
    expected = item["expected_behavior"]
    expected_docs = item.get("expected_docs") or []

    course_ok = trace.course_tag == item.get("expected_course")
    topic_ok = trace.topic == item.get("expected_topic")

    retrieval_hit = bool(expected_docs) and any(d in selected for d in expected_docs)
    citation_hit = bool(expected_docs) and any(d in cited for d in expected_docs)

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_behavior": expected,
        "predicted_behavior": predicted,
        "behavior_source": predicted_source,
        "behavior_ok": predicted == expected,
        "expected_course": item.get("expected_course"),
        "detected_course": trace.course_tag,
        "course_ok": course_ok,
        "expected_topic": item.get("expected_topic"),
        "detected_topic": trace.topic,
        "topic_ok": topic_ok,
        "deterministic_rule": trace.deterministic_rule_used,
        "confidence": trace.confidence,
        "expected_abstention_reason": item.get("expected_abstention_reason"),
        "detected_abstention_reason": getattr(trace, "abstention_reason", "") or None,
        "expected_docs": expected_docs,
        "selected_docs": sorted(set(selected)),
        "cited_docs": sorted(set(cited)),
        "retrieval_hit": retrieval_hit,
        "citation_hit": citation_hit,
        "answer_excerpt": " ".join(answer.split())[:300],
    }


def aggregate(results: list[dict]) -> dict:
    n = len(results)

    def rate(predicate, universe):
        universe = list(universe)
        if not universe:
            return None
        return round(sum(1 for r in universe if predicate(r)) / len(universe), 3)

    answerable = [r for r in results if r["expected_behavior"] == "answer"]
    with_docs = [r for r in answerable if r["expected_docs"]]
    negatives = [r for r in results if r["category"] in NEGATIVE_CATEGORIES]
    llm_unavailable = [r for r in results if r["predicted_behavior"] == "llm_unavailable"]
    # Accuratezza della CAUSA di astensione: solo dove è attesa una causa.
    with_reason = [r for r in results if r.get("expected_abstention_reason")]

    return {
        "total_questions": n,
        "behavior_accuracy": rate(lambda r: r["behavior_ok"], results),
        "course_accuracy": rate(lambda r: r["course_ok"], results),
        "topic_accuracy": rate(lambda r: r["topic_ok"], results),
        "retrieval_hit_rate": rate(lambda r: r["retrieval_hit"], with_docs),
        "citation_hit_rate": rate(lambda r: r["citation_hit"], with_docs),
        "abstention_rate": rate(
            lambda r: r["predicted_behavior"] in {"abstain", "clarify", "unknown_course"},
            negatives,
        ),
        "abstention_reason_accuracy": rate(
            lambda r: r["detected_abstention_reason"] == r["expected_abstention_reason"],
            with_reason,
        ),
        "answerable_count": len(answerable),
        "answerable_with_docs_count": len(with_docs),
        "negative_count": len(negatives),
        "reason_labelled_count": len(with_reason),
        "llm_unavailable_count": len(llm_unavailable),
        # Quanti verdetti vengono dai segnali del trace vs dal fallback testuale
        # (Ciclo 2 — FASE 5): più alto è "trace", più lo scoring è robusto al wording.
        "behavior_from_trace": sum(1 for r in results if r.get("behavior_source") == "trace"),
        "behavior_from_text": sum(1 for r in results if r.get("behavior_source") == "text"),
    }


def aggregate_repeats(summaries: list[dict]) -> dict:
    """Aggrega le metriche-tasso di N esecuzioni in media e deviazione standard.

    Quantifica la *banda di rumore* del modello locale, che a `temperature=0`
    resta comunque non deterministico: la stessa pipeline può cambiare l'esito di
    una domanda da una run all'altra (osservato su q17 in ESP-06/07). Funzione
    pura — testabile offline (Ciclo 2 — FASE 7).

    Per ogni metrica raccoglie i valori non-None delle N esecuzioni e restituisce
    `{n, mean, std, min, max, values}`. `std` è la deviazione standard campionaria
    (ddof=1, stimatore non distorto della dispersione), 0.0 con meno di due valori.
    Le metriche assenti in tutte le run (universo vuoto → None) sono omesse.
    """
    if not summaries:
        raise ValueError("nessun sommario da aggregare")

    out: dict = {}
    for key in RATE_METRIC_KEYS:
        values = [s[key] for s in summaries if s.get(key) is not None]
        if not values:
            continue
        out[key] = {
            "n": len(values),
            "mean": round(statistics.fmean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "values": [round(v, 4) for v in values],
        }
    return out


def aggregate_per_question(runs: list[list[dict]]) -> list[dict]:
    """Riassume, per ogni domanda, la stabilità del verdetto tra le N esecuzioni.

    `runs` è la lista dei result-set (uno per esecuzione, nell'ordine del dataset).
    Per ogni `id` riporta quante run l'hanno classificata `behavior_ok`, i verdetti
    distinti osservati e un flag `oscillates` (vero quando l'esito cambia tra le
    run). Serve a individuare *quali* domande generano la variabilità aggregata —
    tipicamente i rami generativi (es. q17) — separandole da quelle deterministiche
    stabili. Funzione pura (Ciclo 2 — FASE 7).
    """
    if not runs:
        return []

    n = len(runs)
    order = [r["id"] for r in runs[0]]
    by_id: dict = {}
    for run in runs:
        for r in run:
            by_id.setdefault(r["id"], []).append(r)

    summary = []
    for qid in order:
        items = by_id.get(qid, [])
        ok_count = sum(1 for r in items if r.get("behavior_ok"))
        distinct = sorted({r.get("predicted_behavior") for r in items})
        summary.append(
            {
                "id": qid,
                "category": items[0].get("category"),
                "expected_behavior": items[0].get("expected_behavior"),
                "ok_count": ok_count,
                "runs": n,
                "distinct_verdicts": distinct,
                "oscillates": len(distinct) > 1,
            }
        )
    return summary


def write_reports(results: list[dict], summary: dict, model: str) -> tuple[str, str]:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(REPORTS_DIR, f"baseline_{stamp}.json")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "summary": summary,
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    md_path = os.path.join(REPORTS_DIR, f"baseline_{stamp}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Report baseline UniLaw Agent — {stamp}\n\n")
        fh.write(f"- Modello LLM: `{model}`\n")
        fh.write(f"- Domande totali: {summary['total_questions']}\n\n")
        fh.write("## Metriche aggregate\n\n")
        fh.write("| Metrica | Valore |\n|---|---|\n")
        for key in (
            "behavior_accuracy",
            "course_accuracy",
            "topic_accuracy",
            "retrieval_hit_rate",
            "citation_hit_rate",
            "abstention_rate",
            "abstention_reason_accuracy",
        ):
            fh.write(f"| {key} | {summary[key]} |\n")
        fh.write(f"| domande LLM non disponibili | {summary['llm_unavailable_count']} |\n")
        fh.write(
            f"| esiti dai segnali del trace / dal testo | "
            f"{summary['behavior_from_trace']} / {summary['behavior_from_text']} |\n"
        )
        fh.write("\n## Dettaglio per domanda\n\n")
        fh.write(
            "| id | cat | atteso | predetto | ok | corso | topic | retr | cit | regola |\n"
            "|---|---|---|---|:--:|:--:|:--:|:--:|:--:|---|\n"
        )
        for r in results:
            fh.write(
                f"| {r['id']} | {r['category']} | {r['expected_behavior']} | "
                f"{r['predicted_behavior']} | {'✓' if r['behavior_ok'] else '✗'} | "
                f"{'✓' if r['course_ok'] else '✗'} | {'✓' if r['topic_ok'] else '✗'} | "
                f"{'✓' if r['retrieval_hit'] else ('—' if not r['expected_docs'] else '✗')} | "
                f"{'✓' if r['citation_hit'] else ('—' if not r['expected_docs'] else '✗')} | "
                f"{r['deterministic_rule'] or '-'} |\n"
            )
    return json_path, md_path


def write_variability_report(
    summaries: list[dict],
    repeats: dict,
    per_question: list[dict],
    model: str,
    repeat: int,
) -> tuple[str, str]:
    """Scrive il report di variabilità: media±σ su N run + stabilità per domanda.

    Distinto dai report `baseline_*` di singola esecuzione: file `variability_*`
    (Ciclo 2 — FASE 7).
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    questions_per_run = summaries[0]["total_questions"] if summaries else 0

    json_path = os.path.join(REPORTS_DIR, f"variability_{stamp}.json")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "repeat": repeat,
        "questions_per_run": questions_per_run,
        "metric_variability": repeats,
        "per_question_stability": per_question,
        "run_summaries": summaries,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    md_path = os.path.join(REPORTS_DIR, f"variability_{stamp}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Report di variabilità UniLaw Agent — {repeat} esecuzioni — {stamp}\n\n")
        fh.write(f"- Modello LLM: `{model}`\n")
        fh.write(f"- Numero di esecuzioni (repeat): {repeat}\n")
        fh.write(f"- Domande per esecuzione: {questions_per_run}\n\n")

        fh.write("## Metriche aggregate (media ± deviazione standard su N esecuzioni)\n\n")
        fh.write("| Metrica | Media | Dev. std | Min | Max |\n|---|---|---|---|---|\n")
        for key in RATE_METRIC_KEYS:
            st = repeats.get(key)
            if st is None:
                continue
            fh.write(f"| {key} | {st['mean']} | {st['std']} | {st['min']} | {st['max']} |\n")
        fh.write(
            "\n_Deviazione standard campionaria (ddof=1) sulle "
            f"{repeat} esecuzioni; `0.0` indica una metrica costante (ramo "
            "deterministico)._\n\n"
        )

        oscillating = [q for q in per_question if q["oscillates"]]
        fh.write("## Domande che oscillano tra le esecuzioni\n\n")
        if not oscillating:
            fh.write(
                f"_Nessuna: tutti i verdetti sono stati stabili sulle {repeat} esecuzioni._\n\n"
            )
        else:
            fh.write(
                "| id | cat | atteso | ok / N | verdetti distinti |\n"
                "|---|---|---|:--:|---|\n"
            )
            for q in oscillating:
                fh.write(
                    f"| {q['id']} | {q['category']} | {q['expected_behavior']} | "
                    f"{q['ok_count']}/{q['runs']} | {', '.join(q['distinct_verdicts'])} |\n"
                )
            fh.write("\n")

        fh.write("## Sommari per esecuzione\n\n")
        fh.write(
            "| run | behavior | course | topic | retrieval | citation | abstention | reason |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        for i, s in enumerate(summaries, 1):
            fh.write(
                f"| {i} | {s['behavior_accuracy']} | {s['course_accuracy']} | "
                f"{s['topic_accuracy']} | {s['retrieval_hit_rate']} | "
                f"{s['citation_hit_rate']} | {s['abstention_rate']} | "
                f"{s['abstention_reason_accuracy']} |\n"
            )
    return json_path, md_path


def variability_from_reports(report_paths: list[str]) -> tuple[str, str]:
    """Costruisce un report di variabilità da report baseline GIÀ salvati, offline.

    Recupero robusto (Ciclo 2 — FASE 7): se un'esecuzione `--repeat` viene
    interrotta, le run completate restano su disco come `baseline_*.json` (ognuno
    con `summary` e `results`). Questa funzione li ri-aggrega — senza Ollama né
    indice — riusando `aggregate_repeats` e `aggregate_per_question`, così le run
    già fatte non vanno sprecate.
    """
    if not report_paths:
        raise ValueError("nessun report da aggregare")

    summaries, runs = [], []
    model = "?"
    for path in report_paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        summaries.append(data["summary"])
        runs.append(data["results"])
        model = data.get("model", model)

    repeats = aggregate_repeats(summaries)
    per_question = aggregate_per_question(runs)
    return write_variability_report(summaries, repeats, per_question, model, len(summaries))


def _error_result(item: dict, exc: Exception) -> dict:
    """Risultato segnaposto quando una domanda solleva un'eccezione a runtime."""
    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_behavior": item["expected_behavior"],
        "predicted_behavior": "error",
        "behavior_source": "error",
        "behavior_ok": False,
        "expected_course": item.get("expected_course"),
        "detected_course": None,
        "course_ok": False,
        "expected_topic": item.get("expected_topic"),
        "detected_topic": None,
        "topic_ok": False,
        "deterministic_rule": None,
        "confidence": None,
        "expected_abstention_reason": item.get("expected_abstention_reason"),
        "detected_abstention_reason": None,
        "expected_docs": item.get("expected_docs") or [],
        "selected_docs": [],
        "cited_docs": [],
        "retrieval_hit": False,
        "citation_hit": False,
        "answer_excerpt": f"ERRORE: {exc}",
    }


def evaluate_dataset(responder: UniLawResponder, dataset: list[dict]) -> list[dict]:
    """Esegue tutte le domande del dataset una volta; una domanda che fallisce
    non blocca le altre (risultato `error`). Riusato a ogni ripetizione."""
    results = []
    for i, item in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] {item['id']}: {item['question'][:60]}")
        try:
            results.append(evaluate_question(responder, item))
        except Exception as exc:  # robustezza: una domanda non blocca tutto
            results.append(_error_result(item, exc))
    return results


def main():
    # Effetti collaterali globali applicati solo all'esecuzione come script
    # (non all'import del modulo da parte dei test).
    os.chdir(ROOT)
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="Valutazione baseline UniLaw Agent")
    parser.add_argument(
        "--questions",
        default=os.path.join(ROOT, "eval", "questions_baseline.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=0, help="Esegui solo le prime N domande")
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Abilita il reranker neurale (cross-encoder) per questa esecuzione.",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Disattiva l'evidence selection (passa i chunk interi al modello).",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Disattiva TUTTE le regole deterministiche: ogni domanda passa al RAG generativo.",
    )
    parser.add_argument(
        "--prose-templates",
        action="store_true",
        help="Riabilita i 5 template di prosa (default: solo guard numerico TOLC).",
    )
    parser.add_argument(
        "--semantic-intent",
        action="store_true",
        help="Abilita l'intent detection semantica (opt-in) che affianca le keyword "
        "(Ciclo 2 — FASE 11).",
    )
    parser.add_argument(
        "--semantic-grounding",
        action="store_true",
        help="Abilita il grounding semantico delle citazioni (opt-in): recupera per "
        "similarità di embedding le frasi parafrasate bocciate dal lessicale "
        "(Ciclo 2 — FASE 12).",
    )
    parser.add_argument(
        "--semantic-abstention",
        action="store_true",
        help="Abilita la retrieval strength semantica per la causa di astensione "
        "(opt-in): decide fuori_dominio vs evidenza_insufficiente per similarità di "
        "embedding query↔fonte, con soglia ricalibrata (Ciclo 2 — FASE 13).",
    )
    parser.add_argument(
        "--no-general-tesi-hint",
        action="store_true",
        help="Disattiva l'hint di mitigazione q14 (Ciclo 2 — FASE 14): senza, il "
        "profilo di risposta non autorizza l'uso della regola generale quando un "
        "regolamento generale sulla tesi è fra le fonti (serve all'A/B).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Esegui l'intero dataset N volte e riporta media±σ delle metriche "
        "(quantifica la variabilità del modello locale, Ciclo 2 — FASE 7).",
    )
    parser.add_argument(
        "--aggregate-reports",
        nargs="+",
        default=None,
        metavar="baseline_*.json",
        help="NON esegue il modello: costruisce il report di variabilità "
        "aggregando 'summary'/'results' di report baseline già salvati "
        "(recupero di run completate prima di un'interruzione). Ciclo 2 — FASE 7.",
    )
    args = parser.parse_args()

    # Modalità di sola aggregazione (offline, nessun Ollama né indice).
    if args.aggregate_reports:
        vjson, vmd = variability_from_reports(args.aggregate_reports)
        print(
            f"Report di variabilità costruito da {len(args.aggregate_reports)} "
            f"report:\n  {vjson}\n  {vmd}"
        )
        return

    from agent import UniLawResponder
    from config import CHROMA_PERSIST_DIRECTORY, DEFAULT_MODEL_NAME

    dataset = load_dataset(args.questions)
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"Caricamento indice ChromaDB da {CHROMA_PERSIST_DIRECTORY} ...")
    db = load_vector_db()
    responder = UniLawResponder(
        db,
        use_neural_reranker=True if args.reranker else None,
        use_evidence=False if args.no_evidence else None,
        use_deterministic=False if args.no_deterministic else None,
        use_prose_templates=True if args.prose_templates else None,
        use_semantic_intent=True if args.semantic_intent else None,
        use_semantic_grounding=True if args.semantic_grounding else None,
        use_semantic_abstention=True if args.semantic_abstention else None,
        use_general_tesi_hint=False if args.no_general_tesi_hint else None,
    )

    repeat = max(1, args.repeat)
    runs_results: list[list[dict]] = []
    runs_summaries: list[dict] = []
    for run_idx in range(1, repeat + 1):
        if repeat > 1:
            print(f"\n--- Esecuzione {run_idx}/{repeat} ---")
        results = evaluate_dataset(responder, dataset)
        summary = aggregate(results)
        runs_results.append(results)
        runs_summaries.append(summary)
        # Scrive SUBITO il report di QUESTA esecuzione: con --repeat (run lunghe su
        # CPU) un'interruzione non fa perdere le run già completate, che restano
        # ricomponibili in un report di variabilità con `--aggregate-reports`
        # (Ciclo 2 — FASE 7).
        json_path, md_path = write_reports(results, summary, DEFAULT_MODEL_NAME)
        if repeat > 1:
            print(f"  report esecuzione {run_idx}: {json_path}")

    # L'ultima esecuzione resta il report di dettaglio "corrente" (con repeat=1
    # comportamento storico invariato: un solo baseline_*.json scritto).
    last_summary = runs_summaries[-1]

    print(
        "\n=== SOMMARIO BASELINE"
        + (" (ultima esecuzione)" if repeat > 1 else "")
        + " ==="
    )
    for k, v in last_summary.items():
        print(f"  {k}: {v}")
    print(f"\nReport salvati:\n  {json_path}\n  {md_path}")

    if repeat > 1:
        repeats = aggregate_repeats(runs_summaries)
        per_question = aggregate_per_question(runs_results)
        vjson, vmd = write_variability_report(
            runs_summaries, repeats, per_question, DEFAULT_MODEL_NAME, repeat
        )
        print(f"\n=== VARIABILITÀ SU {repeat} ESECUZIONI (media ± σ) ===")
        for key in RATE_METRIC_KEYS:
            st = repeats.get(key)
            if st is None:
                continue
            print(
                f"  {key}: {st['mean']} ± {st['std']} "
                f"(min {st['min']}, max {st['max']})"
            )
        oscillating = [q["id"] for q in per_question if q["oscillates"]]
        print(
            "  domande che oscillano: "
            + (", ".join(oscillating) if oscillating else "nessuna")
        )
        print(f"\nReport variabilità salvati:\n  {vjson}\n  {vmd}")


if __name__ == "__main__":
    main()
