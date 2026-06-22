"""Test del post-processing della risposta (`_postprocess_answer`).

Copre la pulizia del caso speciale di astensione del Ciclo 2 — FASE 10:
il ramo che, per `doc_type ∈ {accesso, regolamento, altro}` e argomento
"accesso", *riscriveva* l'astensione con un testo generico è stato rimosso
perché ridondante con il layer di astensione (FASE 6) e perché — riscrivendo
la risposta — mascherava l'astensione ai marcatori di `is_abstention`.

Questi test sono offline (nessun Ollama): `_postprocess_answer` è puro rispetto
allo stato del `last_trace` e non chiama il modello.
"""

from abstention import is_abstention

# Testo del vecchio ramo di riscrittura: non deve più comparire (anti-regressione).
LEGACY_REWRITE_FRAGMENT = "Le fonti recuperate contengono documenti potenzialmente rilevanti"


def test_empty_answer_returns_safety_net(responder):
    result = responder._postprocess_answer("   ")
    assert result == "Non lo so in base ai documenti disponibili."
    assert responder.last_trace.confidence == "bassa"
    assert "vuota" in responder.last_trace.confidence_reason.lower()


def test_whitespace_is_stripped(responder):
    result = responder._postprocess_answer("  La prova prevede 80 quesiti.  ")
    assert result == "La prova prevede 80 quesiti."


def test_confident_answer_is_left_unchanged(responder):
    # Una risposta senza marcatori di incertezza non viene toccata e la
    # confidenza stimata a monte non viene forzata a "bassa".
    responder.last_trace.confidence = "alta"
    answer = "La prova di ammissione a Scienze dell'Educazione L-19 prevede 80 quesiti [F1]."
    result = responder._postprocess_answer(answer)
    assert result == answer
    assert responder.last_trace.confidence == "alta"


def test_abstention_is_preserved_not_rewritten(responder):
    # Astensione su argomento "accesso": in passato sarebbe stata riscritta con
    # un testo generico; ora viene restituita verbatim e la confidenza scende.
    answer = "Non lo so in base ai documenti disponibili: non ho trovato la procedura richiesta."
    result = responder._postprocess_answer(answer)
    assert result == answer
    assert LEGACY_REWRITE_FRAGMENT not in result
    assert responder.last_trace.confidence == "bassa"


def test_abstention_stays_detectable_downstream(responder):
    # Proprietà chiave che il vecchio ramo rompeva: dopo il post-processing
    # l'astensione resta riconoscibile da `is_abstention`, così il layer FASE 6
    # può classificarne la causa.
    answer = "Non sono in grado di fornire una risposta affidabile sulla base dei documenti."
    result = responder._postprocess_answer(answer)
    assert is_abstention(result) is True


def test_no_legacy_rewrite_for_any_abstention(responder):
    # Indipendentemente dal contenuto, il messaggio generico del vecchio caso
    # speciale non deve mai essere prodotto.
    for answer in (
        "Non lo so.",
        "I documenti disponibili non contengono questa informazione.",
        "Non sono in possesso di dati sufficienti per rispondere.",
    ):
        result = responder._postprocess_answer(answer)
        assert LEGACY_REWRITE_FRAGMENT not in result
