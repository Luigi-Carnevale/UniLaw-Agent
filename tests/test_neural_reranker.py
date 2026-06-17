"""Test del reranker neurale opzionale (FASE 4).

Tutti offline: usano uno `scorer` iniettato, quindi NON richiedono il download
del modello né `sentence-transformers`. Verificano la logica di riordino, il
fallback quando il modello non è disponibile, e la gestione dei casi limite.
"""

from langchain_core.documents import Document

from neural_reranker import CrossEncoderReranker, rerank_by_scores


def _doc(source, text=""):
    return Document(page_content=text, metadata={"source": source})


def test_rerank_by_scores_orders_desc():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    out = rerank_by_scores(docs, [0.1, 0.9, 0.5])
    assert [d.metadata["source"] for d in out] == ["b", "c", "a"]


def test_rerank_by_scores_is_stable_on_ties():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    out = rerank_by_scores(docs, [1.0, 1.0, 0.0])
    # a e b hanno lo stesso punteggio: ordine d'ingresso preservato
    assert [d.metadata["source"] for d in out] == ["a", "b", "c"]


def _fake_scorer(question, docs):
    # punteggio = 1.0 se 'tolc' è nel testo, altrimenti 0.0
    return [1.0 if "tolc" in d.page_content.lower() else 0.0 for d in docs]


def test_rerank_reorders_top_n_with_injected_scorer():
    reranker = CrossEncoderReranker("fake-model", scorer=_fake_scorer)
    docs = [
        _doc("a", "bando erasmus mobilità"),
        _doc("b", "punteggio TOLC e OFA"),
        _doc("c", "piano di studi"),
    ]
    out = reranker.rerank("accesso tolc", docs, top_n=3)
    assert out[0].metadata["source"] == "b"  # l'unico con 'tolc' va in cima


def test_rerank_keeps_tail_beyond_top_n():
    reranker = CrossEncoderReranker("fake-model", scorer=_fake_scorer)
    docs = [_doc("a", "x"), _doc("b", "tolc"), _doc("c", "y"), _doc("d", "z")]
    out = reranker.rerank("q", docs, top_n=2)
    # solo i primi 2 vengono riordinati; 'c' e 'd' restano in coda nell'ordine dato
    assert [d.metadata["source"] for d in out[-2:]] == ["c", "d"]


def test_available_true_with_injected_scorer():
    assert CrossEncoderReranker("fake-model", scorer=_fake_scorer).available() is True


def test_fallback_when_model_unavailable():
    # scorer che restituisce None simula un modello non disponibile
    reranker = CrossEncoderReranker("fake-model", scorer=lambda q, docs: None)
    docs = [_doc("a", "x"), _doc("b", "tolc")]
    out = reranker.rerank("q", docs, top_n=2)
    assert [d.metadata["source"] for d in out] == ["a", "b"]  # ordine invariato


def test_rerank_empty_docs():
    reranker = CrossEncoderReranker("fake-model", scorer=_fake_scorer)
    assert reranker.rerank("q", [], top_n=5) == []
