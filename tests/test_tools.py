"""Test del modulo di calcolo numerico sicuro (`tools.py`).

Verifica:
- espressioni aritmetiche pure;
- percentuali in linguaggio naturale (formato italiano);
- rifiuto di input non matematici e di caratteri non ammessi.
"""

import pytest

from tools import (
    _normalizza_numero_italiano,
    calcola_espressione_sicura,
    prova_calcolo_sicuro,
)


def test_simple_arithmetic():
    assert prova_calcolo_sicuro("2+2") == "4"
    assert prova_calcolo_sicuro("(5+3)*2") == "16"
    assert prova_calcolo_sicuro("10/2") == "5"


def test_division_italian_decimal():
    # 10/4 = 2.5 -> formato italiano con virgola
    assert prova_calcolo_sicuro("10/4") == "2,5"


def test_percentage_natural_language():
    # Casi senza separatore di migliaia: comportamento corretto.
    assert prova_calcolo_sicuro("Quanto è il 20% di 1350?") == "270"
    assert prova_calcolo_sicuro("15% su 200") == "30"


def test_percentage_thousands_separator():
    # CORRETTO (Ciclo 2 — FASE 1): un singolo punto usato come separatore di
    # migliaia ("20.000") era interpretato da float() come decimale (20.0),
    # perché `_normalizza_numero_italiano` rimuoveva i punti solo quando ce
    # n'era più di uno; "5% di 20.000€" restituiva "1" invece di "1.000".
    # Ora un punto seguito da esattamente 3 cifre è trattato come separatore
    # di migliaia. Questo test, in origine di caratterizzazione del bug,
    # certifica il comportamento corretto.
    # Cfr. docs/valutazione_rag.md e docs/changelog_tecnico.md.
    assert prova_calcolo_sicuro("Calcola il 5% di 20.000€") == "1.000"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("20.000", 20000.0),       # un punto + 3 cifre -> migliaia
        ("20.000,50", 20000.50),   # punto=migliaia, virgola=decimale
        ("20000.50", 20000.50),    # un punto + 2 cifre -> decimale
        ("1.234.567", 1234567.0),  # più punti -> tutti migliaia
        ("20.5", 20.5),            # un punto + 1 cifra -> decimale
        ("5,5", 5.5),              # solo virgola -> decimale
        ("1350", 1350.0),          # intero puro
        ("20.000€", 20000.0),      # con simbolo valuta e spazi
    ],
)
def test_normalizza_numero_italiano(raw, expected):
    assert _normalizza_numero_italiano(raw) == pytest.approx(expected)


def test_non_math_returns_none():
    # Una domanda normale non deve attivare il calcolo deterministico.
    assert prova_calcolo_sicuro("Come funziona l'accesso a Informatica?") is None


def test_rejects_illegal_characters():
    out = calcola_espressione_sicura("__import__('os')")
    assert out.startswith("Errore")


def test_rejects_code_injection_via_entrypoint():
    # Anche passando dall'entrypoint, testo non aritmetico => None (nessun calcolo).
    assert prova_calcolo_sicuro("os.system('rm -rf /')") is None
