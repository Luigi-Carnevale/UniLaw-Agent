"""Retrieval ibrido: vettoriale (ChromaDB) + lessicale (BM25), fusi con RRF.

Introdotto in FASE 3. Il recupero dei candidati avviene su due "arm":
- vettoriale: ricerca semantica multi-query (MMR) su ChromaDB (logica preesistente,
  estratta qui da `agent.py`);
- lessicale: BM25 (`rank_bm25`) sui chunk già indicizzati in ChromaDB, utile per i
  termini esatti, i codici e le sigle che la ricerca semantica può non cogliere.

I due ranking vengono fusi con Reciprocal Rank Fusion (RRF), agnostica rispetto
alla scala dei punteggi. La fusione genera i *candidati*; l'ordinamento finale
resta affidato al reranker euristico (vedi `reranking.py`). Con `use_bm25=False`
l'RRF su un solo ranking preserva l'ordine vettoriale: la modalità "vettoriale"
riproduce quindi esattamente il comportamento pre-FASE 3 (baseline).
"""

import logging
import os
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import DEFAULT_K_RETRIEVAL
from intent import asks_tesi_consultazione
from rag_types import QueryIntent, RagTrace

logger = logging.getLogger(__name__)

RRF_K = 60

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Stopword italiane minime (parole funzione molto frequenti). Rimuoverle migliora
# la precisione lessicale di BM25 senza nascondere i termini di contenuto.
_STOPWORDS = {
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra", "il", "lo", "la",
    "i", "gli", "le", "un", "uno", "una", "e", "ed", "o", "ma", "che", "chi",
    "cui", "non", "come", "dove", "quando", "quale", "quali", "quanto", "è",
    "sono", "del", "dello", "della", "dei", "degli", "delle", "al", "allo",
    "alla", "ai", "agli", "alle", "dal", "dalla", "nel", "nella", "si", "se",
    "ci", "mi", "ti", "vi", "ho", "hai", "ha", "posso", "può", "devo", "cosa",
    "sul", "sulla", "questo", "questa",
}


def tokenize(text: str) -> list[str]:
    """Tokenizzazione semplice e trasparente per BM25 (lowercase, no stopword)."""
    return [
        t
        for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]


def _doc_key(doc):
    metadata = doc.metadata or {}
    return (
        metadata.get("source", ""),
        metadata.get("page", None),
        doc.page_content[:150],
    )


def build_query_variants(question: str, intent: QueryIntent) -> list[str]:
    """Costruisce le query di espansione per l'arm vettoriale (logica preesistente)."""
    query_variants = [question]

    course_queries = {
        "informatica": "informatica L-31 regolamento",
        "scienze_educazione": "scienze dell'educazione L-19 regolamento",
        "amministrazione": "scienze dell'amministrazione L-16 regolamento",
        "economia": "scienze economiche statistiche regolamento",
    }

    if intent.course_tag in course_queries:
        query_variants.append(f"{question} {course_queries[intent.course_tag]}")

    topic_queries = {
        "accesso": "regolamento di accesso TOLC OFA immatricolazione punteggio soglie tabella",
        "borsa": "bando borsa di studio ISEE ISPE benefici requisiti graduatoria",
        "erasmus": "bando Erasmus mobilità internazionale graduatoria destinazioni requisiti learning agreement",
        "tesi": "regolamento prova finale tesi elaborato finale requisiti modalità discussione relatore seduta laurea",
        "piano_studi": "piano di studi insegnamenti CFU esami corsi a scelta attività formative",
    }

    if intent.topic in topic_queries:
        query_variants.append(f"{question} {topic_queries[intent.topic]}")

    if intent.course_tag == "informatica" and intent.topic == "accesso":
        query_variants.extend(
            [
                "regolamento di accesso informatica L-31 TOLC-I OFA Ris_Test tabella punteggio",
                "Informatica L-31 TOLC-I punteggio inferiore 9 maggiore 9 minore 16 maggiore 16 OFA",
                "regolamento-di-accesso-informatical-31 TOLC OFA immatricolazione",
            ]
        )

    if intent.course_tag == "scienze_educazione" and intent.topic == "accesso":
        query_variants.extend(
            [
                "immatricolazione scienze dell'educazione L-19 prova di ammissione 80 quesiti risposta multipla",
                "Scienze dell'Educazione L-19 accesso ammissione immatricolazione numero programmato test",
                "prova di ammissione scienze educazione durata 2 ore 30 minuti 80 quesiti",
                "immatricolazione scienze dell'educazione l-19 requisiti domanda prova valutazione",
                "test ammissione scienze dell'educazione cultura generale inglese logica comprensione testo",
            ]
        )

    if intent.course_tag == "scienze_educazione" and intent.topic == "tesi":
        query_variants.extend(
            [
                "scienze dell'educazione L-19 prova finale tesi elaborato relatore CFU",
                "regolamento scienze educazione prova finale elaborato scritto relatore",
                "linee guida tesi scienze educazione elaborato finale discussione",
            ]
        )

    if intent.course_tag == "scienze_educazione" and intent.topic == "piano_studi":
        query_variants.extend(
            [
                "piano di studi scienze dell'educazione L-19 insegnamenti CFU",
                "scienze educazione L19 piano di studi attività formative esami",
                "regolamento scienze educazione L19 insegnamenti crediti formativi",
            ]
        )

    if intent.course_tag == "amministrazione" and intent.topic == "accesso":
        query_variants.extend(
            [
                "scienze dell'amministrazione L-16 accesso immatricolazione requisiti ammissione",
                "regolamento L16 scienze amministrazione accesso immatricolazione",
            ]
        )

    if intent.course_tag == "informatica" and intent.topic == "tesi":
        if asks_tesi_consultazione(question):
            query_variants.extend(
                [
                    "regolamento tesi consultazione embargo accessibile deposito diritti autore lucro",
                    "tesi consultabile dopo laurea embargo autorizzazione relatore consultazione",
                    "regolamento-tesi-2023 consultabile embargo deposito tesi",
                ]
            )

        else:
            query_variants.extend(
                [
                    "Regolamento della prova finale informatica L31 elaborato finale relatore discussione commissione",
                    "prova finale Informatica L-31 modalità requisiti elaborato relatore commissione seduta laurea",
                    "tesi Informatica L31 elaborato finale docente relatore discussione prova finale",
                ]
            )

    if intent.topic == "erasmus":
        query_variants.extend(
            [
                "bando erasmus mobilità internazionale requisiti domanda scadenze graduatoria",
                "Erasmus 25 26 mobilità internazionale studenti requisiti partecipazione learning agreement",
                "termine mobilità Erasmus attestato permanenza learning agreement relazione finale giustificativi viaggio riconoscimento attività",
                "fine mobilità Erasmus documenti da consegnare attestato permanenza allegato C learning agreement relazione",
            ]
        )

    if intent.topic == "borsa":
        query_variants.extend(
            [
                "bando borsa di studio graduatoria provvisoria definitiva assestata idonei beneficiari scorrimento",
                "borsa di studio graduatorie pubblicazione idoneità posizione benefici graduatoria definitiva",
                "bando borsa requisiti economici merito iscrizione ISEE ISPE graduatoria",
            ]
        )

    if intent.course_tag == "informatica" and intent.topic == "piano_studi":
        query_variants.extend(
            [
                "piano di studi informatica L-31 insegnamenti CFU esami scelta",
                "regolamento informatica L31 piano di studi attività formative CFU",
            ]
        )

    return query_variants


def run_vector_queries(vector_db, query_variants: list[str], k: int = DEFAULT_K_RETRIEVAL) -> list:
    """Esegue le query vettoriali (MMR, con fallback a similarity) e deduplica."""
    all_docs = []
    seen = set()

    for qv in query_variants:
        try:
            partial = vector_db.max_marginal_relevance_search(
                qv,
                k=k,
                fetch_k=max(k * 2, 20),
            )

        except Exception:
            partial = vector_db.similarity_search(qv, k=k)

        for doc in partial:
            key = _doc_key(doc)

            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    return all_docs


class Bm25Index:
    """Indice BM25 (Okapi) costruito sui chunk già presenti in ChromaDB."""

    def __init__(self, documents: list):
        self._docs = list(documents)
        # Un chunk che si riduce a 0 token (solo stopword/punteggiatura) farebbe
        # fallire BM25Okapi (avgdl=0 -> ZeroDivisionError): gli assegniamo un
        # token sentinella che non compare nelle query reali.
        tokenized = [tokenize(d.page_content) or ["∅"] for d in self._docs]
        self._bm25 = BM25Okapi(tokenized) if self._docs else None

    def __len__(self) -> int:
        return len(self._docs)

    def search(self, query: str, k: int) -> list:
        if not self._bm25 or not self._docs:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self._docs)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in order[:k]:
            if scores[i] <= 0:
                break
            results.append(self._docs[i])

        return results


def build_bm25_index(vector_db):
    """Costruisce l'indice BM25 leggendo i chunk persistiti in ChromaDB.

    Restituisce ``None`` se il vector store è assente, privo del metodo ``get`` o
    vuoto (così i test con vector store finto e i casi senza indice restano validi).
    """
    if vector_db is None or not hasattr(vector_db, "get"):
        return None

    try:
        data = vector_db.get()
    except Exception as exc:
        logger.warning("Indice BM25 non costruito: %s", exc)
        return None

    texts = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    if not texts:
        return None

    documents = [
        Document(page_content=text or "", metadata=(metadata or {}))
        for text, metadata in zip(texts, metadatas)
    ]

    return Bm25Index(documents)


def reciprocal_rank_fusion(ranked_lists: list[tuple[str, list]], k_rrf: int = RRF_K):
    """Fonde più ranking con RRF: score = somma di 1/(k + rank).

    `ranked_lists` è una lista di coppie (nome_arm, documenti_ordinati).
    Restituisce una lista ordinata di triple (doc, score_rrf, arm_che_lo_hanno_trovato).
    """
    scores: dict = {}
    doc_by_key: dict = {}
    arms_by_key: dict = {}

    for arm_name, docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank + 1)
            doc_by_key.setdefault(key, doc)
            arms_by_key.setdefault(key, set()).add(arm_name)

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)

    return [
        (doc_by_key[key], scores[key], sorted(arms_by_key[key]))
        for key in ordered_keys
    ]


def hybrid_retrieve(
    vector_db,
    bm25_index,
    question: str,
    intent: QueryIntent,
    trace: RagTrace,
    k: int = DEFAULT_K_RETRIEVAL,
    use_bm25: bool = True,
) -> list:
    """Genera i candidati fondendo arm vettoriale e lessicale (RRF) e ne traccia lo scoring."""
    variants = build_query_variants(question, intent)
    trace.query_variants = variants

    vector_docs = run_vector_queries(vector_db, variants, k)
    ranked_lists = [("vector", vector_docs)]

    if use_bm25 and bm25_index is not None and len(bm25_index):
        bm25_docs = bm25_index.search(question, k)
        if bm25_docs:
            ranked_lists.append(("bm25", bm25_docs))

    trace.retrieval_mode = "hybrid" if len(ranked_lists) > 1 else "vettoriale"

    fused = reciprocal_rank_fusion(ranked_lists)

    trace.fusion_scores = [
        "{name} p.{page} | rrf={score:.4f} | {arms}".format(
            name=os.path.basename((doc.metadata or {}).get("source", "?")),
            page=(doc.metadata or {}).get("page", "?"),
            score=score,
            arms="+".join(arms),
        )
        for doc, score, arms in fused[:8]
    ]

    return [doc for doc, _, _ in fused]
