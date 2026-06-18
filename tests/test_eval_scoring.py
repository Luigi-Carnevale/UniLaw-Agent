"""Test dello scoring dell'eval basato sui segnali del trace (Ciclo 2 — FASE 5).

`eval/run_eval.classify_behavior` non deduce più l'esito dai soli marcatori
testuali (fragili al variare della formulazione del modello), ma dai segnali
STRUTTURATI del `RagTrace` — in primo luogo `abstention_reason`, che l'agente
imposta su ogni ramo di astensione — con i marcatori come *fallback*.

Questi test girano offline (nessun Ollama, nessun indice): il modulo `run_eval`
è importabile perché le dipendenze pesanti e gli effetti collaterali globali
sono stati spostati dentro `main()`/`load_vector_db()`. Si usano trace sintetici
(`SimpleNamespace`) per esercitare la mappa causa→esito in isolamento.
"""

import glob
import json
import os
from types import SimpleNamespace

import pytest

from abstention import (
    AMBIGUOUS,
    INSUFFICIENT_EVIDENCE,
    OUT_OF_DOMAIN,
    OUT_OF_DOMAIN_COURSE,
    WEAK_RETRIEVAL,
)
from eval.run_eval import classify_behavior, classify_behavior_from_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, "eval", "reports")


def trace(reason: str = ""):
    """Trace minimale: per lo scoring conta solo `abstention_reason`."""
    return SimpleNamespace(abstention_reason=reason)


# --- 1. I segnali strutturati del trace mappano sull'esito atteso -------------

def test_trace_reason_maps_to_behavior():
    """Ogni causa di astensione del trace produce l'esito corretto, dal trace."""
    assert classify_behavior("qualunque testo", trace(OUT_OF_DOMAIN_COURSE)) == (
        "unknown_course",
        "trace",
    )
    assert classify_behavior("qualunque testo", trace(AMBIGUOUS)) == ("clarify", "trace")
    assert classify_behavior("qualunque testo", trace(WEAK_RETRIEVAL)) == ("abstain", "trace")
    assert classify_behavior("qualunque testo", trace(OUT_OF_DOMAIN)) == ("abstain", "trace")
    assert classify_behavior("qualunque testo", trace(INSUFFICIENT_EVIDENCE)) == (
        "abstain",
        "trace",
    )


def test_trace_wins_over_text():
    """Il trace ha priorità: una causa strutturata vince sul testo di risposta.

    Anche se la risposta *sembrasse* una risposta normale, la presenza di una
    causa di astensione nel trace determina l'esito (e la fonte è "trace").
    """
    answer = "Il termine per la domanda è il 30 settembre secondo [F1] bando.pdf."
    assert classify_behavior(answer, trace(INSUFFICIENT_EVIDENCE)) == ("abstain", "trace")


def test_unmapped_reason_is_still_abstention():
    """Una causa presente ma non mappata è comunque un'astensione (difensivo)."""
    assert classify_behavior("...", trace("causa_ignota")) == ("abstain", "trace")


def test_plain_answer_has_no_reason():
    """Senza causa nel trace e con testo di risposta normale → answer/trace."""
    answer = "Per l'accesso a Informatica L-31 il punteggio TOLC-I richiesto è 9 [F1] regolamento.pdf."
    assert classify_behavior(answer, trace("")) == ("answer", "trace")


# --- 2. Robustezza: il caso che motiva la FASE 5 -----------------------------

def test_trace_catches_abstention_that_text_scorer_misses():
    """Caso centrale: una formulazione di astensione non coperta dalle
    `ABSTAIN_PHRASES` testuali (es. "Non ho trovato ...") sarebbe letta come
    "answer" dal vecchio scorer testuale, ma l'agente l'ha riconosciuta come
    astensione (la sua `is_abstention` è più ampia) e ha impostato la causa nel
    trace. Lo scoring basato sul trace la classifica correttamente come abstain.
    """
    answer = "Non ho trovato questa informazione nei documenti consultati."
    # Il fallback testuale puro la mancherebbe:
    assert classify_behavior_from_text(answer) == "answer"
    # Lo scoring del trace la coglie:
    assert classify_behavior(answer, trace(INSUFFICIENT_EVIDENCE)) == ("abstain", "trace")


# --- 3. Fallback testuale quando il trace non porta segnali -------------------

def test_llm_unavailable_recognised_from_text():
    """L'LLM non raggiungibile non lascia causa nel trace → riconosciuto dal testo."""
    answer = "Errore: Ollama non è raggiungibile in questo momento."
    assert classify_behavior(answer, trace("")) == ("llm_unavailable", "text")


def test_text_fallback_for_unknown_course_and_abstain():
    """Se per qualunque motivo il trace è vuoto, i marcatori esatti dei rami
    senza LLM restano un fallback affidabile (fonte "text")."""
    unknown = "Non posso rispondere in modo affidabile su questo corso usando i documenti."
    assert classify_behavior(unknown, trace("")) == ("unknown_course", "text")

    clarify = "La domanda è ambigua: indica il corso di laurea."
    assert classify_behavior(clarify, trace("")) == ("clarify", "text")

    abstain = "Non lo so in base ai documenti disponibili."
    assert classify_behavior(abstain, trace("")) == ("abstain", "text")


# --- 4. Consistenza con i verdetti già registrati (no regressione) -----------

def _latest_report():
    reports = sorted(glob.glob(os.path.join(REPORTS_DIR, "baseline_*.json")))
    return reports[-1] if reports else None


def test_scoring_reproduces_recorded_verdicts():
    """Ricostruendo il trace dalla causa registrata, lo scoring riproduce il
    `predicted_behavior` del report più recente: garanzia che il passaggio ai
    segnali del trace non altera i verdetti sui casi noti (verifica FASE 5).
    """
    path = _latest_report()
    if not path:
        pytest.skip("nessun report baseline disponibile")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    mismatches = []
    for r in data["results"]:
        recorded = r["predicted_behavior"]
        if recorded in {"error", "llm_unavailable"}:
            # error: eccezione runtime; llm_unavailable: dipende dal testo, non dal trace.
            continue
        reason = r.get("detected_abstention_reason") or ""
        predicted, _ = classify_behavior(r.get("answer_excerpt", ""), trace(reason))
        if predicted != recorded:
            mismatches.append(f"{r['id']}: registrato {recorded!r} != ricostruito {predicted!r}")
    assert not mismatches, "verdetti non riprodotti:\n" + "\n".join(mismatches)
