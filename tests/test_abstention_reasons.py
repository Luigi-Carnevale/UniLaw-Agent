"""Test del layer di astensione (FASE 6). Offline, funzioni pure + trace.

Verifica la classificazione della causa di astensione e che `answer()` imposti
`trace.abstention_reason` sui rami deterministici (senza Ollama).
"""

from langchain_core.documents import Document

import abstention
import re

from abstention import (
    AMBIGUOUS,
    INSUFFICIENT_EVIDENCE,
    OUT_OF_DOMAIN,
    OUT_OF_DOMAIN_COURSE,
    WEAK_RETRIEVAL,
    classify_llm_abstention,
    format_reason,
    is_abstention,
    retrieval_strength,
    semantic_retrieval_strength,
)
from agent import UniLawResponder
from rag_types import RetrievedSource


def _src(content):
    return RetrievedSource(1, "d.pdf", 0, content, "informatica", "accesso")


def test_is_abstention():
    assert is_abstention("Non lo so in base ai documenti disponibili.") is True
    assert is_abstention("Risposta breve: sì, con OFA.") is False


def test_retrieval_strength_high_when_on_topic():
    s = [_src("il punteggio tolc determina gli ofa per l'immatricolazione a informatica")]
    assert retrieval_strength("accesso informatica con tolc e ofa", s) > 0.5


def test_retrieval_strength_low_when_off_topic():
    s = [_src("il bando erasmus disciplina la mobilità internazionale")]
    assert retrieval_strength("qual è la capitale della francia", s) < 0.34


def test_classify_weak_retrieval_without_sources():
    assert classify_llm_abstention("qualunque", [], 0.37) == WEAK_RETRIEVAL


def test_classify_out_of_domain():
    s = [_src("il bando erasmus disciplina la mobilità internazionale")]
    assert classify_llm_abstention("capitale della francia", s, 0.37) == OUT_OF_DOMAIN


def test_classify_insufficient_evidence():
    s = [_src("la tesi è consultabile, con embargo di 24 mesi, dopo la laurea")]
    reason = classify_llm_abstention("la tesi è consultabile dopo la laurea?", s, 0.37)
    assert reason == INSUFFICIENT_EVIDENCE


def test_format_reason_for_known_and_unknown():
    assert "Fuori dominio" in format_reason(OUT_OF_DOMAIN)
    assert format_reason(None) == ""
    assert format_reason("codice_inesistente") == ""


# --- trace.abstention_reason sui rami deterministici (no Ollama) ------------

def test_trace_reason_unknown_course():
    r = UniLawResponder(vector_db=None)
    r.answer("Regole di accesso a Medicina e Chirurgia?",
             show_interpretation=False, show_confidence=False)
    assert r.last_trace.abstention_reason == OUT_OF_DOMAIN_COURSE


def test_trace_reason_ambiguous():
    r = UniLawResponder(vector_db=None)
    r.answer("E per la tesi?", show_interpretation=False, show_confidence=False)
    assert r.last_trace.abstention_reason == AMBIGUOUS


class _EmptyDB:
    def get(self):
        return {"ids": [], "documents": [], "metadatas": []}

    def max_marginal_relevance_search(self, *a, **k):
        return []

    def similarity_search(self, *a, **k):
        return []


def test_trace_reason_weak_retrieval():
    r = UniLawResponder(vector_db=_EmptyDB())
    r.answer("Quanti CFU per la tesi di Informatica L-31?",
             show_interpretation=False, show_confidence=False)
    assert r.last_trace.abstention_reason == WEAK_RETRIEVAL


# --- Ciclo 2 — FASE 13: retrieval strength semantica (opt-in) ---------------

# Embedder finto e deterministico (stesso schema della FASE 12): mappa ogni testo
# su un vettore di "concetti" in base alle parole presenti, così una domanda
# parafrasata e la fonte — che NON condividono token — risultano comunque vicine
# nello spazio di embedding. Permette di testare offline la forza semantica senza
# modello né `sentence-transformers`.
_CONCEPTS = ["tesi", "erasmus", "geografia"]
_CONCEPT_WORDS = {
    # La domanda usa "tesi/visionata/percorso", la fonte "elaborato/accessibile/
    # biblioteca/seduta": nessun token in comune, stesso concetto.
    "tesi": {"tesi", "elaborato", "visionata", "consultabile", "accessibile",
             "biblioteca", "seduta", "laurea", "percorso"},
    "erasmus": {"erasmus", "mobilità", "internazionale", "estero"},
    "geografia": {"capitale", "francia", "parigi", "città"},
}


def _concept_embedder(texts):
    vecs = []
    for text in texts:
        tokens = set(re.findall(r"\w+", text.lower()))
        vecs.append([1.0 if _CONCEPT_WORDS[c] & tokens else 0.0 for c in _CONCEPTS])
    return vecs


def test_semantic_strength_high_on_paraphrase_without_shared_tokens():
    # Domanda parafrasata e fonte non condividono token: il lessicale sottostima,
    # il semantico (stesso concetto "tesi") riconosce la pertinenza.
    s = [_src("L'elaborato finale resta accessibile in biblioteca dopo la seduta.")]
    q = "La tesi può essere visionata terminato il percorso?"
    assert retrieval_strength(q, s) < 0.2          # poco overlap lessicale
    assert semantic_retrieval_strength(q, s, _concept_embedder) >= 0.9


def test_semantic_strength_low_when_off_topic():
    s = [_src("L'elaborato finale resta accessibile in biblioteca dopo la seduta.")]
    assert semantic_retrieval_strength("qual è la capitale della francia", s, _concept_embedder) == 0.0


def test_semantic_strength_zero_without_sources_or_embedder():
    s = [_src("contenuto qualsiasi")]
    assert semantic_retrieval_strength("domanda", [], _concept_embedder) == 0.0
    assert semantic_retrieval_strength("", s, _concept_embedder) == 0.0
    assert semantic_retrieval_strength("domanda", s, None) == 0.0


def test_semantic_strength_safe_fallback_on_embedder_error():
    def boom(texts):
        raise RuntimeError("embedder non disponibile")

    s = [_src("L'elaborato finale resta accessibile in biblioteca.")]
    assert semantic_retrieval_strength("la tesi è consultabile?", s, boom) == 0.0


def test_classify_semantic_branch_rescues_paraphrase_as_insufficient():
    # Stesso (domanda, fonte) classificato in due modi: il LESSICALE, con poco
    # overlap, la marca fuori_dominio; il SEMANTICO la riconosce in dominio →
    # evidenza_insufficiente (è la motivazione della FASE 13).
    s = [_src("L'elaborato finale resta accessibile in biblioteca dopo la seduta.")]
    q = "La tesi può essere visionata terminato il percorso?"
    assert classify_llm_abstention(q, s, 0.37) == OUT_OF_DOMAIN
    assert (
        classify_llm_abstention(
            q, s, 0.37, embedder=_concept_embedder, semantic_ood_max_strength=0.53
        )
        == INSUFFICIENT_EVIDENCE
    )


def test_classify_semantic_branch_marks_off_topic_out_of_domain():
    s = [_src("L'elaborato finale resta accessibile in biblioteca dopo la seduta.")]
    assert (
        classify_llm_abstention(
            "qual è la capitale della francia",
            s,
            0.37,
            embedder=_concept_embedder,
            semantic_ood_max_strength=0.53,
        )
        == OUT_OF_DOMAIN
    )


def test_classify_embedder_none_is_byte_identical_to_lexical():
    # Con embedder=None la decisione resta quella lessicale (neutralità del default).
    s = [_src("il bando erasmus disciplina la mobilità internazionale")]
    q = "capitale della francia"
    assert classify_llm_abstention(q, s, 0.37) == classify_llm_abstention(
        q, s, 0.37, embedder=None, semantic_ood_max_strength=0.53
    )


def test_classify_semantic_falls_back_to_lexical_threshold_if_unset():
    # Con embedder ma senza soglia semantica esplicita, usa `ood_max_strength`.
    s = [_src("L'elaborato finale resta accessibile in biblioteca dopo la seduta.")]
    q = "La tesi può essere visionata terminato il percorso?"
    # semantic strength 1.0 >= 0.37 → insufficiente anche con la sola soglia lessicale.
    assert (
        classify_llm_abstention(q, s, 0.37, embedder=_concept_embedder)
        == INSUFFICIENT_EVIDENCE
    )


# --- cablaggio nel responder (default OFF) ----------------------------------

def test_responder_default_has_no_semantic_abstention():
    r = UniLawResponder(vector_db=None)
    assert r.use_semantic_abstention is False
    assert r.semantic_abstention_embedder is None


def test_responder_semantic_abstention_safe_without_vector_db():
    # Forzando l'opt-in senza vector store, il responder non recupera alcun embedder
    # e ricade in modo sicuro sulla classificazione lessicale (nessun errore).
    r = UniLawResponder(vector_db=None, use_semantic_abstention=True)
    assert r.use_semantic_abstention is True
    assert r.semantic_abstention_embedder is None
