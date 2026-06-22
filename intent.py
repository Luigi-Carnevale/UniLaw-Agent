"""Riconoscimento dell'intento e predicati di disambiguazione.

Estratto da `agent.py` in FASE 2 senza modifiche al comportamento. Contiene:
- `infer_query_intent`: rileva corso, argomento, corso non riconosciuto,
  ambiguità ed eventuale uso della memoria a slot;
- helper `extract_unknown_course_label`, `can_use_course_memory`;
- predicati `asks_*` usati anche da retrieval, reranking e regole deterministiche.

La logica è quella originale, basata su liste di parole chiave (fragile verso
formulazioni nuove): la sua evoluzione è prevista nelle fasi successive.
"""

from typing import Any, Optional

from rag_types import QueryIntent


def infer_query_intent(
    question: str,
    memory: dict[str, Any],
    semantic_classifier: Optional[Any] = None,
) -> QueryIntent:
    """Riconosce corso, argomento, ambiguità e uso della memoria a slot.

    `semantic_classifier` (Ciclo 2 — FASE 11, opt-in) è un classificatore d'intento
    per similarità di embedding (`semantic_intent.SemanticIntentClassifier`). Quando
    fornito, **affianca** le keyword: riempie SOLO le caselle (corso/argomento) che le
    keyword hanno lasciato vuote, senza mai sovrascrivere un riconoscimento a keyword,
    e non tocca i corsi fuori dominio (`detected_unknown_course`). Con `None` (default)
    il comportamento è identico al riconoscimento a sole keyword.
    """
    q = question.lower()

    course_tag = None
    topic = None
    detected_unknown_course = None

    if any(k in q for k in ["informatica", "l31", "l-31", "computer science"]):
        course_tag = "informatica"

    elif any(k in q for k in ["scienze dell'educazione", "educazione", "l19", "l-19"]):
        course_tag = "scienze_educazione"

    elif any(k in q for k in ["amministrazione", "organizzazione", "l16", "l-16"]):
        course_tag = "amministrazione"

    elif any(k in q for k in ["economia", "economiche", "statistiche"]):
        course_tag = "economia"

    elif any(k in q for k in ["medicina", "chirurgia", "giurisprudenza", "ingegneria"]):
        detected_unknown_course = extract_unknown_course_label(q)

    if any(
        k in q
        for k in [
            "tolc",
            "ofa",
            "immatricol",
            "accesso",
            "ammissione",
            "iscrizione",
            "punteggio",
        ]
    ):
        topic = "accesso"

    elif any(k in q for k in ["borsa", "isee", "ispe", "beneficio", "idoneo", "idoneità"]):
        topic = "borsa"

    elif any(k in q for k in ["erasmus", "mobilità", "mobilita", "mobilità internazionale"]):
        topic = "erasmus"

    elif any(k in q for k in ["tesi", "prova finale", "elaborato", "laurea", "esame finale"]):
        topic = "tesi"

    elif any(
        k in q
        for k in [
            "piano di studi",
            "piani di studio",
            "insegnamenti",
            "cfu",
            "corso a scelta",
            "corsi a scelta",
            "esami",
        ]
    ):
        topic = "piano_studi"

    # FASE 11 — affiancamento semantico (opt-in): riempie solo le caselle vuote.
    # Non sovrascrive le keyword e non interviene sui corsi fuori dominio; viene
    # eseguito PRIMA della memoria, così la memoria a slot resta l'ultima risorsa
    # per le sole domande ellittiche.
    if semantic_classifier is not None:
        if course_tag is None and detected_unknown_course is None:
            semantic_course = semantic_classifier.classify_course(question)
            if semantic_course is not None:
                course_tag = semantic_course

        if topic is None:
            semantic_topic = semantic_classifier.classify_topic(question)
            if semantic_topic is not None:
                topic = semantic_topic

    used_memory = False

    if course_tag is None and detected_unknown_course is None:
        can_use_memory = can_use_course_memory(q, topic)

        if can_use_memory and memory.get("last_course_tag"):
            course_tag = memory.get("last_course_tag")
            used_memory = True

    ambiguous_topics = {"tesi", "piano_studi", "accesso"}

    is_ambiguous = (
        topic in ambiguous_topics
        and course_tag is None
        and detected_unknown_course is None
    )

    return QueryIntent(
        course_tag=course_tag,
        topic=topic,
        is_ambiguous=is_ambiguous,
        used_memory=used_memory,
        detected_unknown_course=detected_unknown_course,
    )


def extract_unknown_course_label(question_lower: str) -> str:
    if "medicina" in question_lower and "chirurgia" in question_lower:
        return "Medicina e Chirurgia"

    if "medicina" in question_lower:
        return "Medicina"

    if "giurisprudenza" in question_lower:
        return "Giurisprudenza"

    if "ingegneria" in question_lower:
        return "Ingegneria"

    return "corso non riconosciuto"


def can_use_course_memory(question_lower: str, topic: Optional[str]) -> bool:
    """Usa la memoria solo per domande ellittiche."""
    if topic is None:
        return False

    elliptical_markers = [
        "e per",
        "invece",
        "anche",
        "gli stessi",
        "questo corso",
        "il corso",
        "la tesi",
        "il piano",
        "gli ofa",
        "che succede",
        "cosa devo fare",
    ]

    return any(marker in question_lower for marker in elliptical_markers)


def asks_tesi_consultazione(question: str) -> bool:
    """Distingue le domande sulla prova finale da quelle sulla consultabilità,
    deposito, embargo o accessibilità della tesi dopo la laurea.
    """
    q = question.lower()

    keywords = [
        "consultabile",
        "consultazione",
        "consultare",
        "accessibile",
        "accessibilità",
        "embargo",
        "deposito",
        "repository",
        "diritti",
        "lucro",
        "dopo la laurea",
        "dopo laurea",
    ]

    return any(k in q for k in keywords)


def asks_borsa_graduatoria(question: str) -> bool:
    q = question.lower()
    return any(
        k in q
        for k in [
            "graduatoria",
            "graduatorie",
            "provvisoria",
            "definitiva",
            "assestata",
            "idonei",
            "idoneo",
            "beneficiari",
            "posizione",
            "scorrimento",
        ]
    )


def asks_erasmus_end_mobility(question: str) -> bool:
    q = question.lower()
    return any(
        k in q
        for k in [
            "termine della mobilità",
            "termine della mobilita",
            "fine mobilità",
            "fine mobilita",
            "al rientro",
            "dopo la mobilità",
            "dopo la mobilita",
            "documenti devo consegnare",
            "documenti da consegnare",
            "consegnare al termine",
            "attestato di permanenza",
            "relazione finale",
            "giustificativi",
        ]
    )
