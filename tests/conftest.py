"""Fixture condivise per la suite di test di UniLaw Agent (FASE 1).

I test sono progettati per girare offline:
- NON richiedono che Ollama sia attivo (il modello viene chiamato solo in
  `UniLawResponder.answer()` lungo il ramo di generazione, che i test evitano o
  sostituiscono con un vector store finto);
- NON richiedono l'indice ChromaDB persistito.

Per questo costruiamo `UniLawResponder(vector_db=None)`: il costruttore istanzia
solo il client `ChatOllama` (nessuna connessione di rete al momento della
creazione) e le funzioni testate sono pure o lavorano su dati passati a mano.
"""

import os
import sys
import warnings

import pytest

# Rende importabile la root del progetto quando si lancia `python -m pytest`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from agent import RetrievedSource, UniLawResponder  # noqa: E402


@pytest.fixture
def responder():
    """Responder senza vector store: adatto ai test di funzioni pure."""
    return UniLawResponder(vector_db=None)


class FakeEmptyVectorDB:
    """Vector store finto che non restituisce mai documenti.

    Serve a esercitare il ramo di astensione 'nessuna evidenza' senza dipendere
    dall'indice reale né da Ollama.
    """

    def max_marginal_relevance_search(self, *args, **kwargs):
        return []

    def similarity_search(self, *args, **kwargs):
        return []


@pytest.fixture
def empty_vector_db():
    return FakeEmptyVectorDB()


@pytest.fixture
def source_factory():
    """Factory per costruire `RetrievedSource` di test."""

    def _make(
        index=1,
        filename="doc.pdf",
        page=0,
        content="contenuto",
        course_tag="informatica",
        doc_type="accesso",
    ):
        return RetrievedSource(
            index=index,
            filename=filename,
            page=page,
            content=content,
            course_tag=course_tag,
            doc_type=doc_type,
        )

    return _make
