"""Layer di astensione (FASE 6).

Classifica la *causa* dell'astensione, così da rendere il "non lo so" affidabile,
distinguibile e testabile. Categorie:

- `fuori_dominio_corso`     : la domanda cita un corso non presente nel corpus;
- `ambigua`                 : manca il corso necessario a selezionare le fonti;
- `retrieval_debole`        : nessun documento pertinente recuperato;
- `fuori_dominio`           : la domanda esula dall'ambito dei documenti indicizzati
                              (le fonti recuperate non coprono i termini della query);
- `evidenza_insufficiente`  : sono presenti documenti pertinenti, ma non contengono
                              in modo esplicito la risposta (caso "fonte presente ma
                              insufficiente").

Le funzioni sono pure e testabili offline.
"""

import logging
from typing import List, Optional, Sequence, Tuple

from rag_types import RetrievedSource
from retrieval import tokenize
from semantic_intent import Embedder, cosine_similarity

logger = logging.getLogger(__name__)

OUT_OF_DOMAIN_COURSE = "fuori_dominio_corso"
AMBIGUOUS = "ambigua"
WEAK_RETRIEVAL = "retrieval_debole"
OUT_OF_DOMAIN = "fuori_dominio"
INSUFFICIENT_EVIDENCE = "evidenza_insufficiente"

REASON_LABELS = {
    OUT_OF_DOMAIN_COURSE: "Corso fuori dominio",
    AMBIGUOUS: "Domanda ambigua",
    WEAK_RETRIEVAL: "Retrieval debole",
    OUT_OF_DOMAIN: "Fuori dominio",
    INSUFFICIENT_EVIDENCE: "Fonte presente ma insufficiente",
}

REASON_MESSAGES = {
    OUT_OF_DOMAIN_COURSE: (
        "Il corso indicato non è tra quelli coperti dai documenti indicizzati."
    ),
    AMBIGUOUS: (
        "La domanda non specifica il corso di laurea necessario a selezionare le fonti."
    ),
    WEAK_RETRIEVAL: (
        "Il sistema non ha recuperato documenti pertinenti alla domanda."
    ),
    OUT_OF_DOMAIN: (
        "La domanda sembra esulare dall'ambito dei documenti universitari indicizzati."
    ),
    INSUFFICIENT_EVIDENCE: (
        "Sono stati individuati documenti pertinenti all'argomento, ma non contengono "
        "in modo sufficientemente esplicito la risposta; conviene verificare le fonti indicate."
    ),
}

# Marcatori di incertezza usati per riconoscere un'astensione nel testo generato.
UNCERTAINTY_MARKERS = (
    "non lo so",
    "non sono in possesso",
    "non ho trovato",
    "non sono presenti informazioni",
    "non è presente nei documenti",
    "non risulta nei documenti",
    "non sono in grado di fornire",
    "i documenti disponibili non",
)


def is_abstention(answer: str) -> bool:
    """True se il testo generato segnala un'astensione (incertezza)."""
    low = (answer or "").lower()
    return any(marker in low for marker in UNCERTAINTY_MARKERS)


def retrieval_strength(question: str, sources: List[RetrievedSource]) -> float:
    """Quota dei token di contenuto della domanda coperti dalla migliore fonte.

    Valore in [0, 1]: alto = le fonti recuperate parlano davvero della domanda
    (in dominio); basso = le fonti non coprono i termini della query (fuori dominio).
    """
    query_tokens = set(tokenize(question))
    if not query_tokens or not sources:
        return 0.0

    best = 0.0
    for source in sources:
        source_tokens = set(tokenize(source.content))
        overlap = len(query_tokens & source_tokens) / len(query_tokens)
        if overlap > best:
            best = overlap
    return best


def semantic_retrieval_strength(
    question: str,
    sources: List[RetrievedSource],
    embedder: Embedder,
) -> float:
    """Variante SEMANTICA di `retrieval_strength` (Ciclo 2 — FASE 13).

    Invece della sovrapposizione lessicale di token, misura quanto la domanda è
    vicina — per **similarità di embedding** — alla fonte più pertinente: è la
    massima similarità del coseno fra la domanda e il contenuto di ciascuna fonte
    recuperata. È più robusta verso le parafrasi (sinonimi e riformulazioni non
    condividono token ma restano vicini nello spazio di embedding), dove il
    lessicale sottostima la pertinenza.

    Valore in [0, 1]: le similarità negative sono trattate come 0 (la base parte
    da 0.0), così la scala resta confrontabile con quella lessicale. Riusa
    l'embedder del vector store (nessuna nuova dipendenza). Fallback sicuro: con
    domanda/fonti vuote, embedder assente o in errore restituisce 0.0.
    """
    if not (question or "").strip() or not sources or embedder is None:
        return 0.0

    texts = [question] + [source.content for source in sources]
    try:
        vectors = embedder(texts)
    except Exception as exc:  # embedder non disponibile o errore a runtime
        logger.warning(
            "Retrieval strength semantica non disponibile (%s): uso il solo lessicale.",
            exc,
        )
        return 0.0

    if not vectors or len(vectors) != len(texts):
        return 0.0

    query_vec = vectors[0]
    best = 0.0
    for vec in vectors[1:]:
        sim = cosine_similarity(query_vec, vec)
        if sim > best:
            best = sim
    return best


def classify_by_strength(strength: float, threshold: float) -> str:
    """Mappa una *retrieval strength* alla causa di astensione, data la soglia.

    Sotto soglia → `fuori_dominio` (le fonti non coprono i termini della query);
    a soglia o sopra → `evidenza_insufficiente` (fonti pertinenti ma risposta
    assente). È la decisione binaria governata da `ABSTENTION_OOD_MAX_STRENGTH`,
    estratta come funzione pura per poterla calibrare e validare offline
    (Ciclo 2 — FASE 6).
    """
    return OUT_OF_DOMAIN if strength < threshold else INSUFFICIENT_EVIDENCE


def classify_llm_abstention(
    question: str,
    sources: List[RetrievedSource],
    ood_max_strength: float,
    embedder: Optional[Embedder] = None,
    semantic_ood_max_strength: Optional[float] = None,
) -> str:
    """Classifica un'astensione prodotta dal modello in presenza (o meno) di fonti.

    Con `embedder=None` (default) la decisione fuori_dominio vs
    evidenza_insufficiente usa la `retrieval_strength` **lessicale** e la soglia
    `ood_max_strength`: comportamento **byte-identico** a quello storico.

    Con un `embedder` (Ciclo 2 — FASE 13, opt-in) usa la
    `semantic_retrieval_strength` (similarità di embedding query↔fonte) e una
    soglia **ricalibrata** `semantic_ood_max_strength` — più robusta verso le
    parafrasi. Se la soglia semantica non è fornita, ricade su `ood_max_strength`.
    """
    if not sources:
        return WEAK_RETRIEVAL
    if embedder is not None:
        strength = semantic_retrieval_strength(question, sources, embedder)
        threshold = (
            ood_max_strength
            if semantic_ood_max_strength is None
            else semantic_ood_max_strength
        )
    else:
        strength = retrieval_strength(question, sources)
        threshold = ood_max_strength
    return classify_by_strength(strength, threshold)


def threshold_accuracy(
    labeled: Sequence[Tuple[float, str]],
    threshold: float,
) -> Optional[float]:
    """Accuratezza della soglia su esempi etichettati `(strength, causa_attesa)`.

    Considera solo le due cause governate dalla soglia (`fuori_dominio` vs
    `evidenza_insufficiente`). Restituisce `None` su insieme vuoto.
    """
    items = [(s, r) for s, r in labeled if r in (OUT_OF_DOMAIN, INSUFFICIENT_EVIDENCE)]
    if not items:
        return None
    correct = sum(1 for s, r in items if classify_by_strength(s, threshold) == r)
    return correct / len(items)


def calibrate_ood_threshold(labeled: Sequence[Tuple[float, str]]) -> float:
    """Sceglie la soglia OOD dai dati, anziché fissarla a mano (Ciclo 2 — FASE 6).

    `labeled` sono coppie `(retrieval_strength, causa_attesa)` con causa in
    {`fuori_dominio`, `evidenza_insufficiente`}. Servono esempi di entrambe le
    classi.

    - **Classi separabili** (ogni `fuori_dominio` ha strength < di ogni
      `evidenza_insufficiente`): si restituisce la soglia a **massimo margine**, il
      punto medio tra la strength più alta dei `fuori_dominio` e la più bassa
      degli `evidenza_insufficiente`. È il criterio con cui era stata fissata a
      mano la soglia 0,37.
    - **Classi sovrapposte**: si sceglie, tra i punti medi dei valori ordinati e
      i due estremi, la soglia che **massimizza l'accuratezza** (a parità, la
      prima in ordine crescente).
    """
    ood = sorted(s for s, r in labeled if r == OUT_OF_DOMAIN)
    insuff = sorted(s for s, r in labeled if r == INSUFFICIENT_EVIDENCE)
    if not ood or not insuff:
        raise ValueError(
            "calibrazione impossibile: servono esempi di entrambe le cause "
            "(fuori_dominio e evidenza_insufficiente)."
        )

    # Caso separabile: soglia a massimo margine fra le due classi.
    if ood[-1] < insuff[0]:
        return (ood[-1] + insuff[0]) / 2.0

    # Caso sovrapposto: soglia che massimizza l'accuratezza fra i candidati.
    values = sorted(s for s, _ in labeled)
    candidates = [values[0] - 1.0]
    candidates += [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
    candidates += [values[-1] + 1.0]

    best_threshold, best_acc = candidates[0], -1.0
    for threshold in candidates:
        acc = threshold_accuracy(labeled, threshold) or 0.0
        if acc > best_acc:
            best_acc, best_threshold = acc, threshold
    return best_threshold


def format_reason(reason: Optional[str]) -> str:
    """Riga di spiegazione, da accodare al messaggio di astensione."""
    if not reason or reason not in REASON_MESSAGES:
        return ""
    return f"\n\n_Astensione — {REASON_LABELS[reason]}: {REASON_MESSAGES[reason]}_"
