"""Test della gestione delle citazioni.

Verifica:
- estrazione dei soli riferimenti [F#] realmente presenti tra le fonti;
- blocco fonti coerente (citate vs fallback "utilizzate");
- etichette di citazione con/senza pagina;
- citazione ancorata a una fonte reale nel template deterministico TOLC.
"""

import re

from agent import QueryIntent, RetrievedSource
from citations import grounding_report, normalize_citations, strip_invalid_citations


def test_extract_only_valid_indexes(responder, source_factory):
    sources = [source_factory(index=i, filename=f"doc{i}.pdf") for i in (1, 2, 3)]
    got = responder._extract_cited_source_indexes("vedi [F1] e [F3] e [F9]", sources)
    assert got == {1, 3}


def test_sources_block_lists_only_cited(responder, source_factory):
    sources = [source_factory(index=i, filename=f"doc{i}.pdf") for i in (1, 2, 3, 4)]
    block = responder._format_sources_block("Affermazione sostanziale [F2].", sources)
    assert "Fonti citate:" in block
    assert "doc2.pdf" in block
    assert "doc1.pdf" not in block


def test_sources_block_fallback_when_no_citation(responder, source_factory):
    sources = [source_factory(index=i, filename=f"doc{i}.pdf") for i in (1, 2, 3, 4)]
    block = responder._format_sources_block("Nessun riferimento esplicito.", sources)
    # Risposta non astenuta senza citazioni: fallback alle prime 3 fonti come "utilizzate".
    assert "Fonti utilizzate:" in block
    assert "doc1.pdf" in block


def test_sources_block_abstaining_relabels_fallback(responder, source_factory):
    # Ciclo 2 — FASE 3: in astensione senza citazioni valide il blocco non deve
    # rivendicare fonti "utilizzate" (falsa attribuzione, cfr. q19), ma mostrarle
    # onestamente come documenti consultati e non usati.
    sources = [source_factory(index=i, filename=f"doc{i}.pdf") for i in (1, 2, 3, 4)]
    block = responder._format_sources_block(
        "Non lo so in base ai documenti disponibili.", sources, abstaining=True
    )
    assert "Fonti utilizzate:" not in block
    assert "Documenti consultati (nessuno utilizzato per la risposta):" in block
    # I documenti consultati restano comunque elencati (trasparenza del retrieval).
    assert "doc1.pdf" in block


def test_sources_block_abstaining_keeps_real_citations(responder, source_factory):
    # Anche in astensione, se la risposta cita davvero una fonte valida [F#],
    # quella citazione reale resta mostrata come "Fonti citate".
    sources = [source_factory(index=i, filename=f"doc{i}.pdf") for i in (1, 2, 3, 4)]
    block = responder._format_sources_block(
        "Non lo so con certezza, ma cfr. [F2].", sources, abstaining=True
    )
    assert "Fonti citate:" in block
    assert "Documenti consultati" not in block
    assert "doc2.pdf" in block
    assert "doc1.pdf" not in block


def test_citation_label_with_page():
    s = RetrievedSource(1, "x.pdf", 0, "c", "informatica", "accesso")
    assert s.citation_label == "[F1] x.pdf, pag. 1"


def test_citation_label_without_page():
    s = RetrievedSource(1, "x.pdf", None, "c", "informatica", "accesso")
    assert s.citation_label == "[F1] x.pdf"


def test_deterministic_answer_has_grounded_citation(responder):
    intent = QueryIntent(course_tag="informatica", topic="accesso")
    sources = [
        RetrievedSource(
            index=1,
            filename="regolamento-di-accesso-informatical-31-.pdf",
            page=0,
            content="tolc-i ofa immatricolazione ris_test soglie 9 16",
            course_tag="informatica",
            doc_type="accesso",
        )
    ]
    answer = responder._try_deterministic_accesso_informatica_answer(
        question="Ho preso 11 al TOLC-I per Informatica L-31",
        intent=intent,
        sources=sources,
        show_interpretation=False,
        show_confidence=False,
    )
    assert answer is not None
    assert "con OFA" in answer
    # La citazione deve corrispondere a una fonte realmente passata.
    assert "regolamento-di-accesso-informatical-31-.pdf" in answer
    assert responder.last_trace.deterministic_rule_used == "accesso_informatica_tolc"


# --- FASE 5: verifica delle citazioni ---------------------------------------

def _src(index, content):
    return RetrievedSource(index=index, filename=f"doc{index}.pdf", page=0,
                           content=content, course_tag="informatica", doc_type="accesso")


def test_strip_invalid_citations_removes_phantom():
    sources = [_src(1, "x"), _src(2, "y")]
    out = strip_invalid_citations("Vero [F1], inventato [F9].", sources)
    assert "[F1]" in out
    assert "[F9]" not in out


def test_strip_invalid_citations_keeps_all_valid():
    sources = [_src(1, "x"), _src(2, "y")]
    out = strip_invalid_citations("Affermazione [F1] e [F2].", sources)
    assert "[F1]" in out and "[F2]" in out


def test_grounding_none_without_citations():
    sources = [_src(1, "tolc ofa immatricolazione")]
    ratio, unsupported = grounding_report("Una frase senza riferimenti.", sources)
    assert ratio is None
    assert unsupported == []


def test_grounding_detects_supported_sentence():
    sources = [_src(1, "il punteggio tolc determina gli ofa e l'immatricolazione")]
    ratio, unsupported = grounding_report(
        "Il punteggio TOLC determina gli OFA per l'immatricolazione [F1].", sources
    )
    assert ratio == 1.0
    assert unsupported == []


def test_grounding_flags_unsupported_sentence():
    sources = [_src(1, "il bando erasmus disciplina la mobilità internazionale")]
    # la frase cita [F1] ma parla d'altro: supporto lessicale debole
    ratio, unsupported = grounding_report(
        "La tassa di iscrizione annuale ammonta a tremila euro esatti [F1].", sources
    )
    assert ratio == 0.0
    assert len(unsupported) == 1


# --- Ciclo 2 — FASE 2: normalizzazione del formato citazioni -----------------

def test_normalize_parenthesized_citations():
    sources = [_src(1, "x"), _src(2, "y")]
    out = normalize_citations("Come da (F1) e (F2) la regola.", sources)
    assert out == "Come da [F1] e [F2] la regola."


def test_normalize_bare_citations():
    sources = [_src(1, "x"), _src(2, "y")]
    out = normalize_citations("Vedi F1 e poi F2.", sources)
    assert out == "Vedi [F1] e poi [F2]."


def test_normalize_paren_group_with_multiple_refs():
    sources = [_src(1, "x"), _src(2, "y"), _src(3, "z")]
    assert normalize_citations("(F1, F2) insieme", sources) == "[F1] [F2] insieme"
    assert normalize_citations("(F1 e F3) congiunti", sources) == "[F1] [F3] congiunti"


def test_normalize_leaves_canonical_citations_untouched():
    sources = [_src(1, "x"), _src(3, "z")]
    text = "Già corretto [F1] e [F3]."
    assert normalize_citations(text, sources) == text


def test_normalize_only_valid_indexes():
    # Solo F1/F2 sono fonti reali: (F9) e il nudo F9 non vanno toccati,
    # per non alterare la prosa con riferimenti inesistenti.
    sources = [_src(1, "x"), _src(2, "y")]
    assert normalize_citations("Riferimento (F9) inesistente", sources) == (
        "Riferimento (F9) inesistente"
    )
    assert normalize_citations("Bare F9 inesistente", sources) == "Bare F9 inesistente"


def test_normalize_does_not_match_inside_words_or_codes():
    sources = [_src(2, "y")]
    # "xF2" (parte di parola) e "F2a" (codice) non sono citazioni.
    assert normalize_citations("Codice xF2 e modulo F2a.", sources) == (
        "Codice xF2 e modulo F2a."
    )


def test_normalize_then_grounding_recognises_citation():
    # Senza normalizzazione, "(F1)" sfuggirebbe a grounding_report.
    sources = [_src(1, "il punteggio tolc determina gli ofa e l'immatricolazione")]
    raw = "Il punteggio TOLC determina gli OFA per l'immatricolazione (F1)."
    normalized = normalize_citations(raw, sources)
    ratio, unsupported = grounding_report(normalized, sources)
    assert ratio == 1.0
    assert unsupported == []


# --- Ciclo 2 — FASE 12: grounding semantico delle citazioni (opt-in) ----------

# Embedder finto e deterministico: mappa ogni testo su un vettore di "concetti"
# in base alle parole presenti, così frasi che NON condividono token (parafrasi)
# possono comunque risultare semanticamente vicine. Permette di testare offline la
# rete di recupero semantica senza modello né `sentence-transformers`.
_CONCEPTS = ["consultazione", "tesi", "erasmus"]
_CONCEPT_WORDS = {
    "consultazione": {"consultabile", "consultare", "visionata", "visionare", "leggere", "accessibile"},
    "tesi": {"tesi", "elaborato", "discussione", "laurea"},
    "erasmus": {"erasmus", "mobilità", "internazionale", "estero"},
}


def _fake_embedder(texts):
    vecs = []
    for text in texts:
        tokens = set(re.findall(r"\w+", text.lower()))
        vecs.append([1.0 if _CONCEPT_WORDS[c] & tokens else 0.0 for c in _CONCEPTS])
    return vecs


def test_grounding_lexical_fails_on_paraphrase_without_embedder():
    # Frase corretta ma parafrasata: nessun token in comune con la fonte → il solo
    # lessicale la boccia (è la motivazione della FASE 12).
    sources = [_src(1, "L'elaborato finale resta consultabile presso la segreteria.")]
    ratio, unsupported = grounding_report(
        "La tesi può essere visionata dopo la discussione [F1].", sources
    )
    assert ratio == 0.0
    assert len(unsupported) == 1


def test_grounding_semantic_rescues_paraphrase():
    # Stessa frase: con l'embedder, la similarità semantica la recupera.
    sources = [_src(1, "L'elaborato finale resta consultabile presso la segreteria.")]
    ratio, unsupported = grounding_report(
        "La tesi può essere visionata dopo la discussione [F1].",
        sources,
        embedder=_fake_embedder,
    )
    assert ratio == 1.0
    assert unsupported == []


def test_grounding_semantic_does_not_rescue_unrelated_sentence():
    # La rete semantica non deve "salvare" una frase davvero estranea alla fonte.
    sources = [_src(1, "Il bando Erasmus disciplina la mobilità internazionale.")]
    ratio, unsupported = grounding_report(
        "La tassa annuale ammonta a tremila euro esatti [F1].",
        sources,
        embedder=_fake_embedder,
    )
    assert ratio == 0.0
    assert len(unsupported) == 1


def test_grounding_embedder_none_is_byte_identical():
    # Con embedder=None il risultato è quello del solo lessicale (neutralità per
    # costruzione: il default OFF non cambia nulla).
    sources = [_src(1, "L'elaborato finale resta consultabile presso la segreteria.")]
    answer = "La tesi può essere visionata dopo la discussione [F1]."
    assert grounding_report(answer, sources) == grounding_report(
        answer, sources, embedder=None
    )


def test_grounding_semantic_cannot_remove_lexical_support():
    # Una frase già supportata lessicalmente resta supportata anche con un embedder
    # che restituirebbe similarità nulla: il semantico si AGGIUNGE, non sostituisce.
    sources = [_src(1, "il punteggio tolc determina gli ofa e l'immatricolazione")]
    answer = "Il punteggio TOLC determina gli OFA per l'immatricolazione [F1]."
    zero_embedder = lambda texts: [[0.0, 0.0, 0.0] for _ in texts]
    ratio, unsupported = grounding_report(answer, sources, embedder=zero_embedder)
    assert ratio == 1.0
    assert unsupported == []


def test_grounding_semantic_safe_fallback_on_embedder_error():
    # Se l'embedder solleva un'eccezione, si ricade sul solo lessicale senza errori.
    sources = [_src(1, "L'elaborato finale resta consultabile presso la segreteria.")]

    def broken_embedder(texts):
        raise RuntimeError("embedder non disponibile")

    ratio, unsupported = grounding_report(
        "La tesi può essere visionata dopo la discussione [F1].",
        sources,
        embedder=broken_embedder,
    )
    assert ratio == 0.0
    assert len(unsupported) == 1


def test_grounding_semantic_below_threshold_stays_unsupported():
    # Similarità positiva ma sotto soglia → non recuperata.
    sources = [_src(1, "L'elaborato finale resta consultabile presso la segreteria.")]
    weak_embedder = lambda texts: [[0.3, 0.0, 0.0] if i == 0 else [0.3, 0.4, 0.0]
                                   for i, _ in enumerate(texts)]
    ratio, unsupported = grounding_report(
        "La tesi può essere visionata dopo la discussione [F1].",
        sources,
        embedder=weak_embedder,
        min_semantic=0.95,
    )
    assert ratio == 0.0
    assert len(unsupported) == 1


# --- Ciclo 2 — FASE 12: configurazione e cablaggio nel responder -----------------

def test_semantic_grounding_disabled_by_default():
    import config
    assert config.CITATION_GROUNDING_SEMANTIC_ENABLED is False


def test_responder_default_has_no_grounding_embedder(responder):
    assert responder.use_semantic_grounding is False
    assert responder.semantic_grounding_embedder is None


def test_responder_semantic_grounding_safe_without_vector_db():
    # Forzando l'opt-in senza vector store, il responder non recupera alcun embedder
    # (nessuno disponibile) e non solleva errori: il grounding resta lessicale.
    from agent import UniLawResponder

    responder = UniLawResponder(vector_db=None, use_semantic_grounding=True)
    assert responder.use_semantic_grounding is True
    assert responder.semantic_grounding_embedder is None
