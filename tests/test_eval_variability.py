"""Test dell'aggregazione di variabilità su esecuzioni ripetute (Ciclo 2 — FASE 7).

`eval/run_eval` espone due funzioni pure per quantificare la *banda di rumore* del
modello locale (che a `temperature=0` resta non deterministico):

- `aggregate_repeats(summaries)` → media e deviazione standard campionaria delle
  metriche-tasso su N esecuzioni;
- `aggregate_per_question(runs)` → stabilità del verdetto domanda per domanda,
  con il flag delle domande che oscillano.

Questi test girano offline (nessun Ollama, nessun indice): si usano sommari e
result-set sintetici per esercitare la matematica e la logica di stabilità in
isolamento, com'è stato fatto per lo scoring in FASE 5.
"""

import glob
import json
import os

import pytest

from eval.run_eval import (
    RATE_METRIC_KEYS,
    aggregate_per_question,
    aggregate_repeats,
    variability_from_reports,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, "eval", "reports")


def _summary(**overrides) -> dict:
    """Sommario sintetico con tutte le metriche-tasso valorizzate."""
    base = {k: 1.0 for k in RATE_METRIC_KEYS}
    base["total_questions"] = 40
    base.update(overrides)
    return base


def _result(qid, predicted, ok, category="hard", expected="abstain") -> dict:
    return {
        "id": qid,
        "category": category,
        "expected_behavior": expected,
        "predicted_behavior": predicted,
        "behavior_ok": ok,
    }


# --- 1. aggregate_repeats: media, deviazione, min, max ------------------------

def test_aggregate_repeats_mean_std_min_max():
    """Media e deviazione standard campionaria (ddof=1) calcolate correttamente."""
    summaries = [_summary(behavior_accuracy=0.90), _summary(behavior_accuracy=0.95)]
    agg = aggregate_repeats(summaries)
    b = agg["behavior_accuracy"]
    assert b["n"] == 2
    assert b["mean"] == pytest.approx(0.925)
    # stdev campionaria di [0.90, 0.95] = sqrt(0.00125) ≈ 0.035355
    assert b["std"] == pytest.approx(0.0354, abs=1e-4)
    assert b["min"] == 0.90
    assert b["max"] == 0.95
    assert b["values"] == [0.90, 0.95]


def test_constant_metric_has_zero_std():
    """Una metrica deterministica (sempre uguale) ha deviazione standard 0.0."""
    summaries = [_summary(), _summary(), _summary()]
    agg = aggregate_repeats(summaries)
    assert agg["course_accuracy"]["std"] == 0.0
    assert agg["retrieval_hit_rate"]["std"] == 0.0


def test_single_run_std_is_zero():
    """Con una sola esecuzione la deviazione non è misurabile → 0.0 (non errore)."""
    agg = aggregate_repeats([_summary(behavior_accuracy=0.9)])
    assert agg["behavior_accuracy"]["n"] == 1
    assert agg["behavior_accuracy"]["std"] == 0.0
    assert agg["behavior_accuracy"]["mean"] == pytest.approx(0.9)


def test_none_values_are_skipped():
    """I valori None (universo vuoto in quella run) non entrano nella statistica."""
    summaries = [
        _summary(abstention_reason_accuracy=None),
        _summary(abstention_reason_accuracy=0.8),
        _summary(abstention_reason_accuracy=1.0),
    ]
    a = aggregate_repeats(summaries)["abstention_reason_accuracy"]
    assert a["n"] == 2
    assert a["values"] == [0.8, 1.0]
    assert a["mean"] == pytest.approx(0.9)


def test_metric_none_in_all_runs_is_omitted():
    """Se una metrica è None in tutte le run, viene omessa dall'output."""
    summaries = [_summary(citation_hit_rate=None), _summary(citation_hit_rate=None)]
    agg = aggregate_repeats(summaries)
    assert "citation_hit_rate" not in agg
    assert "behavior_accuracy" in agg  # le altre restano


def test_empty_summaries_raises():
    with pytest.raises(ValueError):
        aggregate_repeats([])


# --- 2. aggregate_per_question: stabilità e oscillazione ----------------------

def test_oscillating_question_flagged():
    """Una domanda con verdetti diversi tra le run è marcata `oscillates`."""
    runs = [
        [_result("q01", "answer", True), _result("q17", "abstain", True)],
        [_result("q01", "answer", True), _result("q17", "answer", False)],
    ]
    pq = {q["id"]: q for q in aggregate_per_question(runs)}

    assert pq["q01"]["oscillates"] is False
    assert pq["q01"]["ok_count"] == 2
    assert pq["q01"]["distinct_verdicts"] == ["answer"]

    assert pq["q17"]["oscillates"] is True
    assert pq["q17"]["ok_count"] == 1
    assert pq["q17"]["runs"] == 2
    assert pq["q17"]["distinct_verdicts"] == ["abstain", "answer"]


def test_per_question_preserves_dataset_order():
    """L'ordine di output segue il dataset (prima run)."""
    runs = [
        [_result("q05", "answer", True), _result("q02", "abstain", True)],
        [_result("q05", "answer", True), _result("q02", "abstain", True)],
    ]
    ids = [q["id"] for q in aggregate_per_question(runs)]
    assert ids == ["q05", "q02"]


def test_single_run_nothing_oscillates():
    runs = [[_result("q01", "answer", True), _result("q17", "abstain", False)]]
    pq = aggregate_per_question(runs)
    assert all(not q["oscillates"] for q in pq)
    assert all(q["runs"] == 1 for q in pq)


def test_empty_runs_returns_empty():
    assert aggregate_per_question([]) == []


# --- 3. Recupero offline da report baseline già salvati -----------------------

def _write_baseline(path, *, behavior, results, model="llama3.1:8b"):
    """Scrive un report baseline minimale (schema `write_reports`)."""
    summary = {k: 1.0 for k in RATE_METRIC_KEYS}
    summary["behavior_accuracy"] = behavior
    summary["total_questions"] = len(results)
    payload = {"model": model, "summary": summary, "results": results}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def test_variability_from_reports(tmp_path, monkeypatch):
    """Ricompone il report di variabilità da due report baseline salvati, senza
    rieseguire il modello (recupero dopo interruzione, Ciclo 2 — FASE 7)."""
    # I report di variabilità vengono scritti in REPORTS_DIR: lo redirigo su tmp.
    monkeypatch.setattr("eval.run_eval.REPORTS_DIR", str(tmp_path))

    r1 = str(tmp_path / "baseline_A.json")
    r2 = str(tmp_path / "baseline_B.json")
    _write_baseline(
        r1, behavior=0.90,
        results=[_result("q01", "answer", True), _result("q17", "abstain", True)],
    )
    _write_baseline(
        r2, behavior=0.95,
        results=[_result("q01", "answer", True), _result("q17", "answer", False)],
    )

    vjson, vmd = variability_from_reports([r1, r2])
    assert os.path.exists(vjson) and os.path.exists(vmd)

    data = json.load(open(vjson, encoding="utf-8"))
    assert data["repeat"] == 2
    assert data["model"] == "llama3.1:8b"
    assert data["metric_variability"]["behavior_accuracy"]["mean"] == pytest.approx(0.925)
    osc = [q["id"] for q in data["per_question_stability"] if q["oscillates"]]
    assert osc == ["q17"]


def test_variability_from_reports_empty_raises():
    with pytest.raises(ValueError):
        variability_from_reports([])


# --- 4. Consistenza con un eventuale report di variabilità registrato ---------

def _latest_variability_report():
    reports = sorted(glob.glob(os.path.join(REPORTS_DIR, "variability_*.json")))
    return reports[-1] if reports else None


def test_recorded_variability_report_is_consistent():
    """Se esiste un report di variabilità, media/min/max sono coerenti coi valori
    grezzi registrati (garanzia che il report scritto rispecchia l'aggregazione)."""
    path = _latest_variability_report()
    if not path:
        pytest.skip("nessun report di variabilità disponibile")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    for key, st in data["metric_variability"].items():
        values = st["values"]
        assert st["n"] == len(values)
        assert st["min"] == pytest.approx(min(values))
        assert st["max"] == pytest.approx(max(values))
        assert st["mean"] == pytest.approx(sum(values) / len(values), abs=1e-3)
