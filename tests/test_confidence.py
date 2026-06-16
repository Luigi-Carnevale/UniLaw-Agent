"""Test della stima di affidabilità (`confidence.estimate_confidence`).

Modulo estratto da `agent.py` in FASE 2: questi test ne fissano il comportamento
(caso vuoto, forte coerenza, fonte di corso diverso) e verificano che il metodo
delegante `UniLawResponder._estimate_confidence` resti allineato.
"""

from confidence import estimate_confidence
from rag_types import QueryIntent


def test_no_sources_is_low(source_factory):
    level, reason = estimate_confidence(QueryIntent("informatica", "accesso"), [])
    assert level == "bassa"
    assert reason == "Nessuna fonte selezionata."


def test_strong_match_is_high(source_factory):
    intent = QueryIntent("informatica", "accesso")
    sources = [
        source_factory(
            index=1,
            filename="regolamento-di-accesso-informatical-31-.pdf",
            content="tolc ofa ris_test immatricolazione soglie 16 9",
            course_tag="informatica",
            doc_type="accesso",
        )
    ]
    level, _ = estimate_confidence(intent, sources)
    assert level == "alta"


def test_off_course_source_lowers_confidence(source_factory):
    intent = QueryIntent("informatica", "accesso")
    sources = [
        source_factory(
            index=1,
            filename="regolamento-l16-amministrazione.pdf",
            content="contenuto non pertinente",
            course_tag="amministrazione",
            doc_type="accesso",
        )
    ]
    level, reason = estimate_confidence(intent, sources)
    assert level in {"media", "bassa"}
    assert "corsi diversi" in reason


def test_responder_delegates_to_module(responder, source_factory):
    intent = QueryIntent("informatica", "accesso")
    sources = [
        source_factory(content="tolc ofa ris_test immatricolazione 16 9",
                        course_tag="informatica", doc_type="accesso"),
    ]
    assert responder._estimate_confidence(intent, sources) == estimate_confidence(
        intent, sources
    )
