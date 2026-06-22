"""Classificatore d'intento semantico opzionale (Ciclo 2 — FASE 11).

Affianca — non sostituisce — il riconoscimento a parole chiave di `intent.py`.
Il riconoscimento storico (`infer_query_intent`) usa liste di keyword: preciso
sulle formulazioni note, fragile sulle parafrasi («aiuto economico per chi ha un
reddito basso» non contiene «borsa»). Questo modulo aggiunge un classificatore
per **similarità di embedding**: confronta la domanda con alcune frasi-ancora per
ogni corso e per ogni argomento e propone l'etichetta più vicina, se supera una
soglia di similarità.

Caratteristiche di progetto (coerenti con `neural_reranker.py`, FASE 4):
- **Opt-in**: disattivato di default (`config.SEMANTIC_INTENT_ENABLED`). Quando è
  disattivato la pipeline non lo costruisce nemmeno, quindi il comportamento
  predefinito resta byte-identico al riconoscimento a keyword.
- **Affianca le keyword**: in `infer_query_intent` il semantico riempie SOLO le
  caselle che le keyword hanno lasciato vuote (corso e/o argomento), senza mai
  sovrascrivere un riconoscimento a keyword. I corsi fuori dominio
  (`detected_unknown_course`) restano gestiti dalle sole keyword.
- **Riusa il modello già caricato**: l'`embedder` è il modello di embedding del
  vector store (lo stesso usato per il retrieval), iniettato dall'esterno; il
  modulo non carica nulla per conto proprio.
- **Testabile offline**: l'`embedder` è iniettabile, così i test non richiedono
  né il download del modello né `sentence-transformers`.
- **Fallback sicuro**: se l'embedder non è disponibile o solleva un'eccezione, il
  classificatore restituisce `None` (nessuna proposta) e la pipeline ricade sul
  solo riconoscimento a keyword, senza errori.

Le soglie predefinite sono **provvisorie** e vanno validate (rischio medio-alto
dichiarato in roadmap): finché la validazione non è completa il classificatore
resta disattivato di default.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Embedder: funzione che mappa una lista di testi in una lista di vettori.
# Compatibile con `HuggingFaceEmbeddings.embed_documents`.
Embedder = Callable[[list[str]], list[list[float]]]


# Frasi-ancora per ogni etichetta. Sono formulazioni naturali (non keyword): il
# classificatore le confronta con la domanda per similarità di embedding e sceglie
# l'etichetta il cui anchor è più vicino. Le etichette coincidono con quelle del
# riconoscimento a keyword (`intent.py`), così il semantico è un sostituto
# trasparente quando riempie una casella vuota.
COURSE_ANCHORS: dict[str, list[str]] = {
    "informatica": [
        "corso di laurea in informatica",
        "laurea triennale in informatica L-31",
        "computer science",
    ],
    "scienze_educazione": [
        "corso di laurea in scienze dell'educazione",
        "scienze della formazione L-19",
        "diventare educatore",
    ],
    "amministrazione": [
        "corso di laurea in amministrazione e organizzazione",
        "scienze dell'amministrazione L-16",
        "gestione e organizzazione della pubblica amministrazione",
    ],
    "economia": [
        "corso di laurea in economia",
        "scienze economiche e statistiche",
        "studi economici e aziendali",
    ],
}

TOPIC_ANCHORS: dict[str, list[str]] = {
    "accesso": [
        "accesso e immatricolazione al corso di laurea",
        "ammissione, TOLC e OFA",
        "iscriversi al corso e punteggio di ammissione",
    ],
    "borsa": [
        "borsa di studio per il diritto allo studio",
        "requisiti di reddito e ISEE per il beneficio economico",
        "aiuto economico per studenti con basso reddito",
    ],
    "erasmus": [
        "erasmus e mobilità internazionale",
        "bando per un periodo di studio all'estero",
        "partire per studiare all'estero",
    ],
    "tesi": [
        "tesi di laurea e prova finale",
        "elaborato finale e seduta di laurea",
        "consultabilità della tesi dopo la laurea",
    ],
    "piano_studi": [
        "piano di studi e insegnamenti del corso",
        "crediti CFU e corsi a scelta",
        "quali esami sono previsti nel corso",
    ],
}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similarità del coseno fra due vettori. Funzione pura, 0.0 se un vettore è nullo."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def best_label_by_similarity(
    query_vec: list[float],
    anchor_vecs: dict[str, list[list[float]]],
    min_similarity: float,
) -> Optional[tuple[str, float]]:
    """Etichetta il cui anchor più vicino supera la soglia; `None` altrimenti.

    Per ogni etichetta si prende la similarità massima fra i suoi anchor (più
    robusta della media verso formulazioni varie), poi si sceglie l'etichetta con
    il punteggio più alto, purché superi `min_similarity`. Funzione pura.
    """
    best_label: Optional[str] = None
    best_score = -1.0

    for label, vecs in anchor_vecs.items():
        label_score = max((cosine_similarity(query_vec, v) for v in vecs), default=-1.0)
        if label_score > best_score:
            best_score = label_score
            best_label = label

    if best_label is None or best_score < min_similarity:
        return None

    return best_label, best_score


class SemanticIntentClassifier:
    """Classificatore d'intento per similarità di embedding (opt-in).

    Gli anchor vengono incorporati una sola volta, in modo pigro, al primo uso
    effettivo (nessun costo se il classificatore non viene mai interrogato).
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        course_min_similarity: float = 0.5,
        topic_min_similarity: float = 0.45,
        course_anchors: Optional[dict[str, list[str]]] = None,
        topic_anchors: Optional[dict[str, list[str]]] = None,
    ):
        self._embedder = embedder
        self.course_min_similarity = course_min_similarity
        self.topic_min_similarity = topic_min_similarity
        self._course_anchors = course_anchors or COURSE_ANCHORS
        self._topic_anchors = topic_anchors or TOPIC_ANCHORS

        self._course_vecs: Optional[dict[str, list[list[float]]]] = None
        self._topic_vecs: Optional[dict[str, list[list[float]]]] = None
        self._embed_failed = False

    def available(self) -> bool:
        """True se un embedder è stato iniettato (il modulo non ne carica di propri)."""
        return self._embedder is not None and not self._embed_failed

    def _embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        if self._embedder is None or self._embed_failed:
            return None
        try:
            return self._embedder(texts)
        except Exception as exc:  # embedder non disponibile o errore a runtime
            logger.warning(
                "Intent semantico non disponibile (%s): uso il solo riconoscimento a keyword.",
                exc,
            )
            self._embed_failed = True
            return None

    def _ensure_anchor_vecs(self) -> bool:
        """Incorpora gli anchor (una volta). True se i vettori sono pronti."""
        if self._course_vecs is not None and self._topic_vecs is not None:
            return True
        if not self.available():
            return False

        course_vecs = self._embed_anchor_group(self._course_anchors)
        topic_vecs = self._embed_anchor_group(self._topic_anchors)
        if course_vecs is None or topic_vecs is None:
            return False

        self._course_vecs = course_vecs
        self._topic_vecs = topic_vecs
        return True

    def _embed_anchor_group(
        self, anchors: dict[str, list[str]]
    ) -> Optional[dict[str, list[list[float]]]]:
        labels: list[str] = []
        phrases: list[str] = []
        for label, group in anchors.items():
            for phrase in group:
                labels.append(label)
                phrases.append(phrase)

        vectors = self._embed(phrases)
        if vectors is None or len(vectors) != len(phrases):
            return None

        grouped: dict[str, list[list[float]]] = {label: [] for label in anchors}
        for label, vec in zip(labels, vectors):
            grouped[label].append(vec)
        return grouped

    def classify_course(self, question: str) -> Optional[str]:
        """Corso più vicino alla domanda sopra soglia; `None` se nessuno o embedder assente."""
        if not self._ensure_anchor_vecs():
            return None
        result = self._classify(question, self._course_vecs, self.course_min_similarity)
        return result[0] if result else None

    def classify_topic(self, question: str) -> Optional[str]:
        """Argomento più vicino alla domanda sopra soglia; `None` se nessuno o embedder assente."""
        if not self._ensure_anchor_vecs():
            return None
        result = self._classify(question, self._topic_vecs, self.topic_min_similarity)
        return result[0] if result else None

    def _classify(
        self,
        question: str,
        anchor_vecs: Optional[dict[str, list[list[float]]]],
        min_similarity: float,
    ) -> Optional[tuple[str, float]]:
        if not anchor_vecs:
            return None
        query_vectors = self._embed([question])
        if not query_vectors:
            return None
        return best_label_by_similarity(query_vectors[0], anchor_vecs, min_similarity)
