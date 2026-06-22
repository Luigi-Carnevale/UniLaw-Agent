"""Calibrazione e validazione della soglia di astensione (Ciclo 2 — FASE 6).

Test offline e puri delle funzioni che rendono la soglia OOD
(`ABSTENTION_OOD_MAX_STRENGTH`) **derivabile dai dati** e **validabile fuori
campione**, anziché un numero fissato a mano:

- `classify_by_strength` — la decisione binaria governata dalla soglia;
- `calibrate_ood_threshold` — sceglie la soglia dai negativi etichettati;
- `threshold_accuracy` — accuratezza di una soglia su esempi etichettati.

Include un controllo di consistenza sul report dell'harness
(`eval/abstention_threshold_validation.py`), quando presente: le accuratezze
salvate devono essere riproducibili ricalcolandole con le funzioni pure dalle
strength registrate (lega il report alla logica testata, senza richiedere Ollama
né l'indice).
"""

import json
import os

import pytest

from abstention import (
    AMBIGUOUS,
    INSUFFICIENT_EVIDENCE,
    OUT_OF_DOMAIN,
    calibrate_ood_threshold,
    classify_by_strength,
    threshold_accuracy,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "eval", "reports", "abstention_threshold_validation.json")


# --- classify_by_strength ---------------------------------------------------

def test_classify_by_strength_below_threshold_is_out_of_domain():
    assert classify_by_strength(0.30, 0.37) == OUT_OF_DOMAIN


def test_classify_by_strength_at_or_above_threshold_is_insufficient():
    assert classify_by_strength(0.37, 0.37) == INSUFFICIENT_EVIDENCE
    assert classify_by_strength(0.50, 0.37) == INSUFFICIENT_EVIDENCE


def test_classify_by_strength_matches_legacy_classifier():
    """Identica alla logica storica `classify_llm_abstention` (con fonti)."""
    from abstention import classify_llm_abstention
    from rag_types import RetrievedSource

    src = [RetrievedSource(1, "d.pdf", 0, "la tesi è consultabile dopo la laurea", "informatica", "tesi")]
    expected = classify_llm_abstention("la tesi è consultabile?", src, 0.37)
    from abstention import retrieval_strength

    assert classify_by_strength(retrieval_strength("la tesi è consultabile?", src), 0.37) == expected


# --- calibrate_ood_threshold ------------------------------------------------

def test_calibrate_separable_returns_max_margin_midpoint():
    labeled = [(0.20, OUT_OF_DOMAIN), (0.60, INSUFFICIENT_EVIDENCE)]
    assert calibrate_ood_threshold(labeled) == pytest.approx(0.40)


def test_calibrate_reproduces_hand_set_threshold():
    """Sui valori reali di q17/q18/q19 la regola ritrova ~0,37 (la soglia di config)."""
    labeled = [
        (1 / 3, OUT_OF_DOMAIN),          # q19
        (0.40, INSUFFICIENT_EVIDENCE),   # q17
        (2 / 3, INSUFFICIENT_EVIDENCE),  # q18
    ]
    t = calibrate_ood_threshold(labeled)
    assert t == pytest.approx(0.3667, abs=1e-3)
    assert 0.34 < t < 0.40


def test_calibrate_requires_both_classes():
    with pytest.raises(ValueError):
        calibrate_ood_threshold([(0.2, OUT_OF_DOMAIN), (0.3, OUT_OF_DOMAIN)])


def test_calibrate_overlapping_maximizes_accuracy():
    """Classi sovrapposte: sceglie la soglia ad accuratezza massima (prima a parità)."""
    labeled = [
        (0.3, OUT_OF_DOMAIN),
        (0.5, OUT_OF_DOMAIN),
        (0.4, INSUFFICIENT_EVIDENCE),
        (0.6, INSUFFICIENT_EVIDENCE),
    ]
    t = calibrate_ood_threshold(labeled)
    assert t == pytest.approx(0.35)
    assert threshold_accuracy(labeled, t) == pytest.approx(0.75)


# --- threshold_accuracy -----------------------------------------------------

def test_threshold_accuracy_perfect_separation():
    labeled = [(0.30, OUT_OF_DOMAIN), (0.50, INSUFFICIENT_EVIDENCE)]
    assert threshold_accuracy(labeled, 0.37) == 1.0


def test_threshold_accuracy_ignores_non_threshold_reasons():
    """Le cause non governate dalla soglia (es. ambigua) sono escluse dal conteggio."""
    labeled = [
        (0.30, OUT_OF_DOMAIN),
        (0.50, INSUFFICIENT_EVIDENCE),
        (0.99, AMBIGUOUS),  # ignorata
    ]
    assert threshold_accuracy(labeled, 0.37) == 1.0


def test_threshold_accuracy_none_on_empty():
    assert threshold_accuracy([], 0.37) is None
    assert threshold_accuracy([(0.5, AMBIGUOUS)], 0.37) is None


# --- consistenza col report dell'harness (se presente) ----------------------

def test_validation_report_is_internally_consistent():
    if not os.path.exists(REPORT):
        pytest.skip("report non presente: eseguire eval/abstention_threshold_validation.py")

    with open(REPORT, encoding="utf-8") as fh:
        payload = json.load(fh)

    summary = payload["summary"]

    # Le strength registrate riproducono le accuratezze del summary.
    cal = [(d["strength"], d["expected"]) for d in payload["calibration"]]
    heldout = [(d["strength"], d["expected"]) for d in payload["heldout_config_threshold"]]

    assert calibrate_ood_threshold(cal) == pytest.approx(summary["calibrated_threshold"], abs=1e-3)
    assert threshold_accuracy(cal, summary["calibrated_threshold"]) == summary["calibration_accuracy"]
    assert (
        threshold_accuracy(heldout, summary["config_threshold"])
        == summary["heldout_accuracy_config_threshold"]
    )

    # Il set held-out non deve mai intersecare quello di calibrazione.
    assert not set(summary["calibration_ids"]) & set(summary["heldout_ids"])

    # Sezione SEMANTICA (Ciclo 2 — FASE 13), se presente: stessa logica pura sulle
    # strength semantiche registrate riproduce le accuratezze del summary.
    if "semantic_calibration" in payload:
        sem_cal = [(d["strength"], d["expected"]) for d in payload["semantic_calibration"]]
        sem_heldout = [
            (d["strength"], d["expected"])
            for d in payload["semantic_heldout_config_threshold"]
        ]
        assert calibrate_ood_threshold(sem_cal) == pytest.approx(
            summary["semantic_calibrated_threshold"], abs=1e-3
        )
        assert (
            threshold_accuracy(sem_cal, summary["semantic_calibrated_threshold"])
            == summary["semantic_calibration_accuracy"]
        )
        assert (
            threshold_accuracy(sem_heldout, summary["semantic_config_threshold"])
            == summary["semantic_heldout_accuracy_config_threshold"]
        )


def test_semantic_abstention_disabled_by_default():
    """Ciclo 2 — FASE 13: la retrieval strength semantica è opt-in (default OFF)."""
    import config

    assert config.ABSTENTION_SEMANTIC_STRENGTH_ENABLED is False
