"""Test del riconoscimento dell'intento (`_infer_query_intent`).

Verifica il rilevamento di corso e argomento, la gestione dei corsi non
riconosciuti, l'ambiguità (argomento senza corso) e l'uso della memoria a slot
per le domande ellittiche.

In coda (Ciclo 2 — FASE 11) sono presenti i test dell'affiancamento semantico
opt-in: gli helper puri di `semantic_intent`, la meccanica del classificatore con
un embedder finto iniettato (offline, nessun modello scaricato), l'integrazione in
`infer_query_intent` (riempie solo le caselle vuote, non sovrascrive le keyword) e
i fallback di sicurezza. La *qualità semantica* su parafrasi reali è invece
misurata dall'eval con il modello vero; qui si verifica solo il cablaggio.
"""

import config
from intent import infer_query_intent
from semantic_intent import (
    COURSE_ANCHORS,
    TOPIC_ANCHORS,
    SemanticIntentClassifier,
    best_label_by_similarity,
    cosine_similarity,
)

# Spazio semantico finto: ogni etichetta (corso/argomento) ha un proprio asse
# ortogonale. Un embedder controllato mappa anchor e domande su questi assi, così i
# test sono deterministici e offline.
_LABELS = list(COURSE_ANCHORS) + list(TOPIC_ANCHORS)


def _basis(label):
    vec = [0.0] * len(_LABELS)
    vec[_LABELS.index(label)] = 1.0
    return vec


def _make_embedder(query_vectors):
    """Embedder finto: gli anchor vanno sull'asse della loro etichetta; le domande
    sui vettori indicati; tutto il resto sul vettore nullo (lontano da ogni anchor).
    """
    table = {}
    for label, phrases in {**COURSE_ANCHORS, **TOPIC_ANCHORS}.items():
        for phrase in phrases:
            table[phrase] = _basis(label)
    table.update(query_vectors)

    def embed(texts):
        return [table.get(text, [0.0] * len(_LABELS)) for text in texts]

    return embed


# --- Helper puri -----------------------------------------------------------------


def test_cosine_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_best_label_picks_highest_above_threshold():
    anchors = {"a": [[1.0, 0.0]], "b": [[0.0, 1.0]]}
    label, score = best_label_by_similarity([0.9, 0.1], anchors, min_similarity=0.5)
    assert label == "a"
    assert score > 0.5


def test_best_label_none_below_threshold():
    anchors = {"a": [[1.0, 0.0]], "b": [[0.0, 1.0]]}
    assert best_label_by_similarity([1.0, 1.0], anchors, min_similarity=0.9) is None


# --- Meccanica del classificatore (embedder finto iniettato) ---------------------


def test_semantic_classify_topic_paraphrase_without_keyword():
    # Domanda priva di keyword di argomento, mappata vicino agli anchor "borsa".
    question = "posso ricevere un sostegno economico se la mia famiglia ha pochi soldi?"
    classifier = SemanticIntentClassifier(
        embedder=_make_embedder({question: _basis("borsa")})
    )
    assert classifier.classify_topic(question) == "borsa"


def test_semantic_classify_course_paraphrase_without_keyword():
    question = "vorrei studiare programmazione e algoritmi all'università"
    classifier = SemanticIntentClassifier(
        embedder=_make_embedder({question: _basis("informatica")})
    )
    assert classifier.classify_course(question) == "informatica"


def test_semantic_returns_none_below_threshold():
    question = "che tempo fa oggi?"
    classifier = SemanticIntentClassifier(
        embedder=_make_embedder({question: [0.0] * len(_LABELS)})
    )
    assert classifier.classify_topic(question) is None
    assert classifier.classify_course(question) is None


def test_semantic_unavailable_without_embedder():
    classifier = SemanticIntentClassifier(embedder=None)
    assert classifier.available() is False
    assert classifier.classify_topic("qualunque domanda") is None


def test_semantic_embedder_failure_is_safe():
    def boom(_texts):
        raise RuntimeError("modello non disponibile")

    classifier = SemanticIntentClassifier(embedder=boom)
    assert classifier.classify_topic("qualunque domanda") is None
    assert classifier.available() is False


# --- Integrazione con infer_query_intent -----------------------------------------


def test_semantic_fills_topic_gap_left_by_keywords():
    question = "posso ricevere un sostegno economico se ho un reddito basso?"
    classifier = SemanticIntentClassifier(
        embedder=_make_embedder({question: _basis("borsa")})
    )
    intent = infer_query_intent(question, {}, classifier)
    assert intent.topic == "borsa"


def test_semantic_does_not_override_keyword_course():
    # La keyword "Informatica L-31" fissa il corso; il semantico (mappato altrove)
    # non deve sovrascriverlo.
    question = "Quali requisiti per Informatica L-31?"
    classifier = SemanticIntentClassifier(
        embedder=_make_embedder({question: _basis("economia")})
    )
    intent = infer_query_intent(question, {}, classifier)
    assert intent.course_tag == "informatica"


def test_semantic_does_not_touch_unknown_course():
    # I corsi fuori dominio restano gestiti dalle keyword: il semantico non li
    # rimappa su un corso noto.
    question = "Quali sono le regole di accesso a Medicina e Chirurgia?"
    classifier = SemanticIntentClassifier(
        embedder=_make_embedder({question: _basis("informatica")})
    )
    intent = infer_query_intent(question, {}, classifier)
    assert intent.detected_unknown_course == "Medicina e Chirurgia"
    assert intent.course_tag is None


def test_none_classifier_is_identical_to_keywords_only():
    # Senza classificatore (default) il comportamento è quello a sole keyword.
    question = "Requisiti ISEE per la borsa di studio"
    base = infer_query_intent(question, {})
    with_none = infer_query_intent(question, {}, None)
    assert (base.course_tag, base.topic, base.is_ambiguous) == (
        with_none.course_tag,
        with_none.topic,
        with_none.is_ambiguous,
    )


# --- Configurazione e cablaggio nel responder ------------------------------------


def test_semantic_intent_disabled_by_default():
    assert config.SEMANTIC_INTENT_ENABLED is False


def test_responder_default_has_no_semantic_classifier(responder):
    assert responder.use_semantic_intent is False
    assert responder.semantic_intent is None


def test_responder_semantic_intent_safe_without_vector_db():
    # Forzando l'opt-in senza vector store, il responder non costruisce un
    # classificatore (nessun embedder disponibile) e non solleva errori.
    from agent import UniLawResponder

    responder = UniLawResponder(vector_db=None, use_semantic_intent=True)
    assert responder.use_semantic_intent is True
    assert responder.semantic_intent is None
    intent = responder._infer_query_intent("Requisiti ISEE per la borsa", {})
    assert intent.topic == "borsa"


def test_course_and_topic_informatica(responder):
    intent = responder._infer_query_intent(
        "Accesso a Informatica L-31 con il TOLC-I", {}
    )
    assert intent.course_tag == "informatica"
    assert intent.topic == "accesso"
    assert intent.is_ambiguous is False


def test_course_scienze_educazione(responder):
    intent = responder._infer_query_intent(
        "Immatricolazione a Scienze dell'Educazione L-19", {}
    )
    assert intent.course_tag == "scienze_educazione"
    assert intent.topic == "accesso"


def test_topic_erasmus(responder):
    intent = responder._infer_query_intent("Quando scade il bando Erasmus?", {})
    assert intent.topic == "erasmus"


def test_topic_borsa(responder):
    intent = responder._infer_query_intent(
        "Requisiti ISEE per la borsa di studio", {}
    )
    assert intent.topic == "borsa"


def test_unknown_course_detected(responder):
    intent = responder._infer_query_intent(
        "Quali sono le regole di accesso a Medicina e Chirurgia?", {}
    )
    assert intent.detected_unknown_course == "Medicina e Chirurgia"
    assert intent.course_tag is None


def test_ambiguous_topic_without_course(responder):
    intent = responder._infer_query_intent("E per la tesi?", {})
    assert intent.topic == "tesi"
    assert intent.course_tag is None
    assert intent.is_ambiguous is True


def test_memory_resolves_elliptical_question(responder):
    intent = responder._infer_query_intent(
        "E per la tesi?", {"last_course_tag": "informatica"}
    )
    assert intent.course_tag == "informatica"
    assert intent.used_memory is True
    assert intent.is_ambiguous is False


def test_memory_not_used_without_elliptical_marker(responder):
    # Una domanda completa non deve ereditare il corso dalla memoria.
    intent = responder._infer_query_intent(
        "Quali sono i requisiti della borsa di studio?",
        {"last_course_tag": "informatica"},
    )
    assert intent.used_memory is False
