"""Test del profilo di risposta e della mitigazione q14 (Ciclo 2 — FASE 14).

Su una domanda di consultabilità della tesi che nomina un corso preciso (q14,
"La tesi di Informatica L-31 è consultabile dopo la laurea?") la regola si trova
in un regolamento GENERALE di Ateneo (`regolamento-tesi-2023.pdf` → course_tag
"generale"), non in un documento specifico del corso. Il modello 8B tende ad
astenersi cercando un dettaglio dedicato al corso. La FASE 14 aggiunge al profilo
di risposta — solo quando un regolamento generale sulla tesi è effettivamente fra
le fonti recuperate — un hint che autorizza l'uso della regola generale.

Test offline (nessun Ollama): `has_general_tesi_regulation` è puro e
`_build_answer_profile` non chiama il modello.
"""

from agent import QueryIntent, has_general_tesi_regulation

# Frammento stabile dell'hint introdotto dalla FASE 14.
AUTHORIZATION_MARKER = "regolamento generale di Ateneo"

CONSULTAZIONE_Q = "La tesi di Informatica L-31 è consultabile dopo la laurea?"
PROVA_FINALE_Q = (
    "Quanti minuti dura la presentazione della prova finale di Informatica L-31?"
)


# --- predicato puro -------------------------------------------------------


def test_has_general_tesi_regulation_true_for_tesi(source_factory):
    sources = [source_factory(filename="regolamento-tesi-2023.pdf",
                              course_tag="generale", doc_type="tesi")]
    assert has_general_tesi_regulation(sources) is True


def test_has_general_tesi_regulation_true_for_regolamento(source_factory):
    sources = [source_factory(course_tag="generale", doc_type="regolamento")]
    assert has_general_tesi_regulation(sources) is True


def test_has_general_tesi_regulation_false_for_course_specific(source_factory):
    # Regolamento della prova finale di Informatica: è specifico del corso.
    sources = [source_factory(course_tag="informatica", doc_type="tesi")]
    assert has_general_tesi_regulation(sources) is False


def test_has_general_tesi_regulation_false_for_other_general_doc(source_factory):
    # Documento generale ma non sulla tesi: non deve attivare l'hint.
    sources = [source_factory(course_tag="generale", doc_type="accesso")]
    assert has_general_tesi_regulation(sources) is False


def test_has_general_tesi_regulation_false_for_empty_or_none():
    assert has_general_tesi_regulation([]) is False
    assert has_general_tesi_regulation(None) is False


def test_has_general_tesi_regulation_true_if_any_source_matches(source_factory):
    sources = [
        source_factory(index=1, course_tag="informatica", doc_type="tesi"),
        source_factory(index=2, course_tag="generale", doc_type="tesi"),
    ]
    assert has_general_tesi_regulation(sources) is True


# --- profilo di risposta --------------------------------------------------


def test_consultazione_profile_adds_authorization_with_general_reg(
    responder, source_factory
):
    intent = QueryIntent(course_tag="informatica", topic="tesi")
    sources = [source_factory(filename="regolamento-tesi-2023.pdf",
                              course_tag="generale", doc_type="tesi")]
    profile = responder._build_answer_profile(intent, CONSULTAZIONE_Q, sources)
    assert "Profilo consultabilità" in profile
    assert AUTHORIZATION_MARKER in profile


def test_consultazione_profile_no_authorization_without_general_reg(
    responder, source_factory
):
    intent = QueryIntent(course_tag="informatica", topic="tesi")
    sources = [source_factory(course_tag="informatica", doc_type="tesi")]
    profile = responder._build_answer_profile(intent, CONSULTAZIONE_Q, sources)
    # Il profilo consultabilità c'è, ma senza l'autorizzazione alla regola generale.
    assert "Profilo consultabilità" in profile
    assert AUTHORIZATION_MARKER not in profile


def test_non_consultazione_tesi_profile_unaffected(responder, source_factory):
    # Domanda sulla prova finale (non consultabilità): usa il profilo prova
    # finale e non riceve l'hint, anche se è presente un regolamento generale.
    intent = QueryIntent(course_tag="informatica", topic="tesi")
    sources = [source_factory(course_tag="generale", doc_type="tesi")]
    profile = responder._build_answer_profile(intent, PROVA_FINALE_Q, sources)
    assert "Profilo prova finale" in profile
    assert AUTHORIZATION_MARKER not in profile


def test_build_answer_profile_sources_default_none_is_safe(responder):
    # Retrocompatibilità: senza `sources` il profilo consultabilità si costruisce
    # comunque (nessun crash) e l'hint, per assenza di fonti, non viene aggiunto.
    intent = QueryIntent(course_tag="informatica", topic="tesi")
    profile = responder._build_answer_profile(intent, CONSULTAZIONE_Q)
    assert "Profilo consultabilità" in profile
    assert AUTHORIZATION_MARKER not in profile


def test_toggle_off_suppresses_hint(source_factory):
    # Con il toggle FASE 14 spento il profilo torna byte-identico a prima:
    # nessun hint, anche con un regolamento generale fra le fonti.
    from agent import UniLawResponder

    responder = UniLawResponder(vector_db=None, use_general_tesi_hint=False)
    intent = QueryIntent(course_tag="informatica", topic="tesi")
    sources = [source_factory(course_tag="generale", doc_type="tesi")]
    profile = responder._build_answer_profile(intent, CONSULTAZIONE_Q, sources)
    assert "Profilo consultabilità" in profile
    assert AUTHORIZATION_MARKER not in profile
