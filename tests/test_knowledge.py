"""Test del layer di conoscenza normativa strutturata (FASE 7).

Verifica che i dati centralizzati e le tabelle generate riproducano ESATTAMENTE
i valori e le stringhe prima codificate nei template (comportamento invariato),
e che la provenienza sia presente. Offline.

Include anche i guard di non-regressione del Ciclo 2 — FASE 8 (le soglie numeriche
TOLC-I 9/16 NON devono tornare nel prompt `ANSWER_STYLE_GUIDE`, restando un'unica
fonte di verità in `knowledge.py` applicata dal guard deterministico) e della FASE 9
(le regole di routing per corso, dopo l'A/B che ne ha misurato l'utilità per il modello
8B, RESTANO nel prompt: il guard impedisce una rimozione accidentale — risultato negativo
documentato in ESP-09).
"""

import config
import knowledge


def test_tolc_thresholds():
    assert knowledge.TOLC_INFORMATICA["ofa_min"] == 9
    assert knowledge.TOLC_INFORMATICA["no_ofa_min"] == 16


def test_tolc_band_labels():
    labels = knowledge.tolc_band_labels()
    assert labels == {"below": "< 9", "middle": ">= 9 e < 16", "above": ">= 16"}


def test_classify_tolc_parity():
    assert knowledge.classify_tolc(5)["range"] == "< 9"
    assert knowledge.classify_tolc(11)["range"] == ">= 9 e < 16"
    assert knowledge.classify_tolc(16)["range"] == ">= 16"
    assert "con OFA" in knowledge.classify_tolc(11)["consequence"]
    assert "senza OFA" in knowledge.classify_tolc(20)["consequence"]


def test_tolc_bands_table_matches_legacy_ofa():
    expected = (
        "| Punteggio Ris\\_Test | Esito | OFA |\n"
        "|---|---|---|\n"
        "| < 9 | Immatricolazione sconsigliata / non diretta | Percorso di preparazione o recupero |\n"
        "| >= 9 e < 16 | Immatricolazione consentita | Con OFA |\n"
        "| >= 16 | Immatricolazione consentita | Senza OFA |\n\n"
    )
    assert knowledge.tolc_bands_table("Esito", "OFA") == expected


def test_tolc_bands_table_matches_legacy_accesso():
    expected = (
        "| Punteggio Ris\\_Test | Esito per l'immatricolazione | Conseguenza |\n"
        "|---|---|---|\n"
        "| < 9 | Immatricolazione sconsigliata / non diretta | Percorso di preparazione o recupero |\n"
        "| >= 9 e < 16 | Immatricolazione consentita | Con OFA |\n"
        "| >= 16 | Immatricolazione consentita | Senza OFA |\n\n"
    )
    assert knowledge.tolc_bands_table("Esito per l'immatricolazione", "Conseguenza") == expected


def test_l19_table_matches_legacy():
    expected = (
        "| Area della prova | Quesiti |\n"
        "|---|---:|\n"
        "| Cultura generale | 30 |\n"
        "| Conoscenze di base di lingua inglese | 10 |\n"
        "| Abilità logiche e analitiche | 20 |\n"
        "| Comprensione del testo | 20 |\n"
        "| **Totale** | **80** |\n"
    )
    assert knowledge.l19_test_table_markdown() == expected


def test_l19_total_consistent():
    assert sum(n for _, n in knowledge.L19_ADMISSION["areas"]) == knowledge.L19_ADMISSION["total_questions"]


def test_provenance_present():
    assert knowledge.TOLC_INFORMATICA_SOURCE.file.endswith(".pdf")
    assert "16" in knowledge.TOLC_INFORMATICA_SOURCE.quote
    assert "80 quesiti" in knowledge.L19_ADMISSION_SOURCE.quote


# ---------------------------------------------------------------------------
# Ciclo 2 — FASE 8: le fasce numeriche TOLC sono rimosse dal prompt
# (unica fonte di verità = knowledge.py + guard deterministico).
# ---------------------------------------------------------------------------
def test_style_guide_has_no_tolc_numeric_bands():
    """Il prompt non deve più ripetere la classificazione numerica TOLC-I.

    Sono le righe rimosse in FASE 8: le soglie 9/16 e le tre fasce vivono solo in
    `knowledge.py` e sono applicate dal guard deterministico (`_try_deterministic_
    accesso_informatica_answer`), non dal prompt.
    """
    guide = config.ANSWER_STYLE_GUIDE
    removed_signatures = [
        "fasce < 9, >= 9 e < 16, >= 16",
        "punteggio TOLC-I è inferiore a 9",
        "maggiore o uguale a 9 e minore di 16",
        "maggiore o uguale a 16",
    ]
    for sig in removed_signatures:
        assert sig not in guide, f"Fascia numerica TOLC reintrodotta nel prompt: {sig!r}"

    # Le soglie restano la fonte di verità nel layer di conoscenza.
    assert knowledge.TOLC_INFORMATICA["ofa_min"] == 9
    assert knowledge.TOLC_INFORMATICA["no_ofa_min"] == 16


def test_style_guide_keeps_course_routing_rules():
    """Ciclo 2 — FASE 9 (risultato negativo / trade-off): il routing per corso RESTA
    nel prompt.

    Si era valutato di rimuoverlo perché l'instradamento *dei documenti* è già fatto
    sui metadata dal reranker + filtro per corso (`reranking.py`). L'A/B nella stessa
    sessione (within-session σ=0) ha però misurato una regressione (behavior 0,90 →
    0,875/0,85): la prosa fa anche da *framing di commit* per il modello 8B, e senza
    aumentano le false astensioni su domande answerable borderline. Come per il reranker
    neurale (FASE 4), il risultato negativo è riportato e il routing è mantenuto. Questo
    guard impedisce una rimozione accidentale (cfr. ESP-09). Le fasce numeriche TOLC
    restano comunque fuori (sono il guard FASE 8 qui sopra).
    """
    guide = config.ANSWER_STYLE_GUIDE
    assert "Regole specifiche per accesso/TOLC/OFA:" in guide
    assert "regolamento-di-accesso-informatical-31-.pdf" in guide
    # almeno una sezione di routing per ciascun topic principale
    for header in [
        "Regole specifiche per accesso/ammissione Scienze dell'Educazione L-19:",
        "Regole specifiche per prova finale:",
        "Regole specifiche per Erasmus:",
        "Regole specifiche per piano di studi:",
    ]:
        assert header in guide, f"Sezione di routing mancante: {header!r}"
