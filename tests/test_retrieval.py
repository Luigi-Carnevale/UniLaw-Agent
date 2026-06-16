"""Test del retrieval ibrido (FASE 3): tokenizzazione, BM25, RRF, fusione.

Tutti offline: non richiedono Ollama né l'indice ChromaDB reale (si usano
documenti finti in memoria e un vector store finto).
"""

from langchain_core.documents import Document

from rag_types import QueryIntent, RagTrace
from retrieval import (
    Bm25Index,
    build_query_variants,
    hybrid_retrieve,
    reciprocal_rank_fusion,
    tokenize,
)


def _doc(source, text, page=0):
    return Document(page_content=text, metadata={"source": source, "page": page})


# --- tokenize ---------------------------------------------------------------

def test_tokenize_removes_stopwords_and_short_tokens():
    tokens = tokenize("Il TOLC-I per l'accesso a Informatica")
    assert "tolc" in tokens
    assert "accesso" in tokens
    assert "informatica" in tokens
    assert "il" not in tokens   # stopword
    assert "per" not in tokens  # stopword
    assert "i" not in tokens    # lunghezza 1


# --- RRF --------------------------------------------------------------------

def test_rrf_single_list_preserves_order():
    a, b, c = _doc("a", "x"), _doc("b", "y"), _doc("c", "z")
    fused = reciprocal_rank_fusion([("vector", [a, b, c])])
    assert [d.metadata["source"] for d, _, _ in fused] == ["a", "b", "c"]


def test_rrf_rewards_documents_in_both_arms():
    a, b, c = _doc("a", "x"), _doc("b", "y"), _doc("c", "z")
    # 'a' compare in entrambe le liste, 'b' e 'c' in una sola.
    fused = reciprocal_rank_fusion([("vector", [a, b]), ("bm25", [a, c])])
    keys = [d.metadata["source"] for d, _, _ in fused]
    assert keys[0] == "a"          # presente in entrambi gli arm
    assert set(keys) == {"a", "b", "c"}  # dedup, nessun doppione
    # 'a' deve avere score di fusione maggiore di 'b' e 'c'
    score_by_key = {d.metadata["source"]: s for d, s, _ in fused}
    assert score_by_key["a"] > score_by_key["b"]
    assert score_by_key["a"] > score_by_key["c"]


def test_rrf_reports_arms():
    a = _doc("a", "x")
    fused = reciprocal_rank_fusion([("vector", [a]), ("bm25", [a])])
    _, _, arms = fused[0]
    assert arms == ["bm25", "vector"]


# --- BM25 -------------------------------------------------------------------

def test_bm25_ranks_matching_document_first():
    docs = [
        _doc("acc", "regolamento di accesso e tolc per informatica"),
        _doc("era", "bando erasmus mobilità internazionale learning agreement"),
        _doc("piano", "piano di studi insegnamenti e cfu"),
    ]
    index = Bm25Index(docs)
    results = index.search("erasmus mobilità", k=3)
    assert results
    assert results[0].metadata["source"] == "era"


def test_bm25_empty_index_returns_nothing():
    assert Bm25Index([]).search("qualunque", k=5) == []


# --- query variants ---------------------------------------------------------

def test_query_variants_includes_question_and_expansions():
    intent = QueryIntent("informatica", "accesso")
    variants = build_query_variants("posso immatricolarmi?", intent)
    assert variants[0] == "posso immatricolarmi?"
    assert len(variants) > 1  # aggiunge espansioni per corso/argomento


# --- hybrid_retrieve (con vector store finto) -------------------------------

class _FakeVectorDB:
    def __init__(self, docs):
        self._docs = docs

    def max_marginal_relevance_search(self, query, k, fetch_k):
        return list(self._docs)

    def similarity_search(self, query, k):
        return list(self._docs)


def test_hybrid_mode_adds_lexical_recall():
    # Corpus BM25 con abbastanza documenti perché l'IDF sia significativo
    # (in un corpus di 2 documenti l'IDF degenera a 0).
    a = _doc("a", "tolc accesso informatica")
    b = _doc("b", "ofa immatricolazione")
    c = _doc("c", "erasmus mobilità internazionale learning agreement")
    filler = [_doc(f"f{i}", f"testo generico di riempimento documento {i}") for i in range(5)]
    vdb = _FakeVectorDB([a, b])          # l'arm vettoriale NON trova 'c'
    bm25 = Bm25Index([b, c] + filler)
    trace = RagTrace()
    docs = hybrid_retrieve(vdb, bm25, "erasmus mobilità", QueryIntent("informatica", "accesso"),
                           trace, use_bm25=True)
    sources = {d.metadata["source"] for d in docs}
    assert trace.retrieval_mode == "hybrid"
    assert "a" in sources           # recupero vettoriale
    assert "c" in sources           # recall lessicale aggiunta da BM25
    assert trace.fusion_scores      # scoring tracciato


def test_vector_only_mode_preserves_vector_order():
    a, b = _doc("a", "x"), _doc("b", "y")
    vdb = _FakeVectorDB([a, b])
    bm25 = Bm25Index([b])
    trace = RagTrace()
    docs = hybrid_retrieve(vdb, bm25, "qualcosa", QueryIntent(None, None),
                           trace, use_bm25=False)
    assert trace.retrieval_mode == "vettoriale"
    assert [d.metadata["source"] for d in docs] == ["a", "b"]
