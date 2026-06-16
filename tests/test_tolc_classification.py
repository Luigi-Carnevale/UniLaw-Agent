"""Test della classificazione deterministica del punteggio TOLC-I (Informatica L-31).

Copre le funzioni `_classify_tolc_score` e `_extract_tolc_score`, che codificano
le soglie normative (< 9 / >= 9 e < 16 / >= 16). Sono test di *characterization*:
fissano il comportamento attuale prima di una eventuale trasformazione in dato
strutturato derivato dalle fonti (FASE 7).
"""

import pytest


@pytest.mark.parametrize(
    "score,expected_range",
    [
        (0, "< 9"),
        (8, "< 9"),
        (8.9, "< 9"),
        (9, ">= 9 e < 16"),
        (11, ">= 9 e < 16"),
        (15.9, ">= 9 e < 16"),
        (16, ">= 16"),
        (20, ">= 16"),
        (50, ">= 16"),
    ],
)
def test_classify_boundaries(responder, score, expected_range):
    assert responder._classify_tolc_score(score)["range"] == expected_range


def test_classify_consequences(responder):
    assert "OFA" in responder._classify_tolc_score(11)["consequence"]
    assert "senza OFA" in responder._classify_tolc_score(20)["consequence"]
    assert "recupero" in responder._classify_tolc_score(5)["consequence"]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Ho preso 11 al TOLC-I per Informatica L-31", 11.0),
        ("Ho ottenuto 7 al test", 7.0),
        ("Il mio punteggio di 14,5 basta?", 14.5),
        ("con 16 posso immatricolarmi?", 16.0),
    ],
)
def test_extract_score(responder, question, expected):
    assert responder._extract_tolc_score(question) == expected


def test_extract_score_does_not_confuse_l31(responder):
    # "L-31" non deve essere interpretato come punteggio 31.
    assert responder._extract_tolc_score("accesso a Informatica L-31") is None


def test_extract_score_ambiguous_multiple_numbers(responder):
    # Più numeri senza un indizio esplicito => nessun punteggio estratto.
    assert responder._extract_tolc_score("tra 9 e 16 punti") is None
