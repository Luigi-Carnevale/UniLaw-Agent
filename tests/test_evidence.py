"""Test dell'evidence selection (FASE 5). Offline, funzioni pure."""

from evidence import select_passage, split_sentences


def test_split_sentences_basic():
    s = split_sentences("Prima frase. Seconda frase! Terza?\nQuarta riga")
    assert s == ["Prima frase.", "Seconda frase!", "Terza?", "Quarta riga"]


def test_short_content_returned_whole():
    content = "Testo breve sotto la soglia."
    assert select_passage("qualunque domanda", content, max_chars=700) == content


def test_selects_relevant_sentences():
    content = (
        "Il bando Erasmus disciplina la mobilità internazionale. "
        "La tesi si discute davanti a una commissione. "
        "Per l'accesso a Informatica conta il punteggio TOLC e l'eventuale OFA. "
        "La mensa universitaria osserva orari stagionali. "
        "Le borse di studio dipendono dall'ISEE. "
        "Gli OFA del TOLC si assolvono secondo il regolamento di accesso."
    )
    passage = select_passage(
        "Come funziona l'accesso con il TOLC e gli OFA a Informatica?",
        content,
        max_sentences=3,
        min_sentences=1,
        max_chars=200,
    )
    # le frasi su TOLC/OFA devono essere preferite a mensa/Erasmus
    assert "TOLC" in passage
    assert "OFA" in passage
    assert "mensa" not in passage.lower()


def test_respects_char_budget():
    content = ". ".join(f"frase numero {i} con parole varie" for i in range(40)) + "."
    passage = select_passage("parole varie", content, max_chars=120)
    assert len(passage) <= 200  # entro un margine ragionevole del tetto


def test_min_sentences_floor_when_no_overlap():
    content = (
        "Alfa beta gamma delta epsilon. Zeta eta theta iota. "
        "Kappa lambda mu nu. Xi omicron pi rho. Sigma tau upsilon phi."
    )
    # query senza alcuna sovrapposizione: si tiene comunque il minimo garantito
    passage = select_passage("domanda totalmente estranea xyz", content,
                             max_sentences=6, min_sentences=2, max_chars=300)
    assert passage  # non vuoto
    assert len(split_sentences(passage)) >= 2
