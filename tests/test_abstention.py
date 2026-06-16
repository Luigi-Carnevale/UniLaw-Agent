"""Test della capacità di astensione e richiesta di chiarimento.

Copre i tre rami che NON richiedono Ollama:
- corso non riconosciuto (fuori dominio noto);
- domanda ambigua (argomento senza corso) -> richiesta di chiarimento;
- nessuna evidenza recuperata -> astensione esplicita.
"""

from agent import UniLawResponder


def test_unknown_course_abstains(responder):
    answer = responder.answer(
        "Quali sono le regole di accesso a Medicina e Chirurgia?",
        show_interpretation=False,
        show_confidence=False,
    )
    assert "Non posso rispondere in modo affidabile su questo corso" in answer
    assert responder.last_trace.confidence == "bassa"


def test_ambiguous_question_asks_clarification(responder):
    answer = responder.answer(
        "E per la tesi?",
        show_interpretation=False,
        show_confidence=False,
    )
    assert "ambigua" in answer.lower()
    assert responder.last_trace.confidence == "bassa"


def test_no_evidence_abstains(empty_vector_db):
    responder = UniLawResponder(vector_db=empty_vector_db)
    answer = responder.answer(
        "Quanti CFU sono previsti per la tesi di Informatica L-31?",
        show_interpretation=False,
        show_confidence=False,
    )
    assert "Non lo so in base ai documenti disponibili." in answer
    assert responder.last_trace.confidence == "bassa"


def test_empty_question(responder):
    assert responder.answer("   ") == "Inserisci una domanda."
