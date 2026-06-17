"""Test dell'esportazione del trace (FASE 8). Offline, funzioni pure."""

import json

from rag_types import RagTrace
from trace_export import trace_to_dict, trace_to_json, trace_to_markdown


def _trace():
    return RagTrace(
        question="Ho preso 11 al TOLC-I per Informatica L-31?",
        course_tag="informatica",
        topic="accesso",
        used_memory=False,
        confidence="alta",
        confidence_reason="Fonti coerenti.",
        answer_profile="Template accesso.",
        query_variants=["q1", "q2"],
        selected_sources=["[F1] regolamento-di-accesso-informatical-31-.pdf, pag. 1"],
        rejected_hint=["altro.pdf"],
        deterministic_rule_used="accesso_informatica_tolc",
        retrieval_mode="hybrid",
        fusion_scores=["regolamento... p.1 | rrf=0.0312 | bm25+vector"],
        reranker="euristico",
        evidence_chars="900 → 700 caratteri",
        grounding="100% frasi citanti supportate",
        abstention_reason="",
    )


def test_dict_has_core_sections():
    d = trace_to_dict(_trace())
    assert d["question"].startswith("Ho preso 11")
    assert d["interpretation"]["course_tag"] == "informatica"
    assert d["retrieval"]["mode"] == "hybrid"
    assert d["confidence"]["level"] == "alta"
    assert d["selected_sources"]


def test_json_is_valid_and_roundtrips():
    payload = trace_to_json(_trace())
    parsed = json.loads(payload)  # deve essere JSON valido
    assert parsed["interpretation"]["topic"] == "accesso"
    assert parsed["retrieval"]["query_variants"] == ["q1", "q2"]


def test_markdown_contains_key_sections():
    md = trace_to_markdown(_trace())
    assert "# UniLaw Agent — RAG trace" in md
    assert "Ho preso 11" in md
    assert "## Retrieval" in md
    assert "### Query generate" in md
    assert "Fonti selezionate" in md
    assert "regolamento-di-accesso-informatical-31-.pdf" in md


def test_none_trace_is_handled():
    assert trace_to_dict(None) == {}
    assert json.loads(trace_to_json(None)) == {}
    assert "nessun trace disponibile" in trace_to_markdown(None)


def test_empty_lists_render_placeholder():
    md = trace_to_markdown(RagTrace(question="vuota"))
    assert "(nessuno)" in md  # liste vuote -> placeholder, non crash
