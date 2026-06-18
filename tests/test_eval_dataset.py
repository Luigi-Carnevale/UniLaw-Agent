"""Test di integrità del dataset di valutazione (Ciclo 2 — FASE 4).

Il dataset `eval/questions_baseline.jsonl` è cresciuto da 20 a 40 domande
(più corsi/argomenti, parafrasi, distrattori e negativi held-out). Questi test
sono una **rete di sicurezza contro l'etichettatura errata**: girano offline
(nessun Ollama, nessun indice) e verificano in modo deterministico che ogni riga
sia ben formata e — soprattutto — che le etichette `expected_course` /
`expected_topic` coincidano con l'output di `infer_query_intent`, che è la stessa
logica deterministica usata dalla pipeline. Così un'etichetta sbagliata fa fallire
i test invece di falsare silenziosamente le metriche.

Per coerenza con lo stile della suite (incrementi di conteggio contenuti), ogni
verifica è un singolo test che itera su tutte le righe e, in caso di errore,
elenca gli `id` colpevoli.
"""

import json
import os

from abstention import (
    AMBIGUOUS,
    INSUFFICIENT_EVIDENCE,
    OUT_OF_DOMAIN,
    OUT_OF_DOMAIN_COURSE,
    WEAK_RETRIEVAL,
)
from intent import infer_query_intent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "eval", "questions_baseline.jsonl")
DOCS_DIR = os.path.join(ROOT, "documenti")

VALID_BEHAVIORS = {"answer", "abstain", "clarify", "unknown_course"}
VALID_CATEGORIES = {
    "easy",
    "hard",
    "ambiguous",
    "out_of_domain",
    "no_answer",
    "synonym",
    "malformed",
}
VALID_COURSES = {"informatica", "scienze_educazione", "amministrazione", "economia", None}
VALID_REASONS = {
    OUT_OF_DOMAIN_COURSE,
    AMBIGUOUS,
    WEAK_RETRIEVAL,
    OUT_OF_DOMAIN,
    INSUFFICIENT_EVIDENCE,
}
NEGATIVE_CATEGORIES = {"out_of_domain", "no_answer", "ambiguous"}


def load_dataset():
    items = []
    with open(DATASET, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


ROWS = load_dataset()


def test_dataset_loads_and_has_grown():
    """Il dataset è valido e ampliato (FASE 4: da 20 a 40 domande)."""
    assert len(ROWS) >= 40


def test_ids_are_unique():
    ids = [r["id"] for r in ROWS]
    assert len(ids) == len(set(ids))


def test_schema_and_value_domains():
    """Ogni riga ha i campi richiesti con valori in dominio."""
    bad = []
    for r in ROWS:
        ok = (
            all(f in r for f in ("id", "question", "category", "expected_behavior", "expected_docs"))
            and bool(str(r.get("question", "")).strip())
            and r.get("category") in VALID_CATEGORIES
            and r.get("expected_behavior") in VALID_BEHAVIORS
            and r.get("expected_course") in VALID_COURSES
            and isinstance(r.get("expected_docs"), list)
        )
        if not ok:
            bad.append(r.get("id"))
    assert not bad, f"righe con schema/valori non validi: {bad}"


def test_intent_labels_match_deterministic_inference():
    """`expected_course`/`expected_topic` coincidono con `infer_query_intent`.

    È il controllo centrale: l'intent è deterministico, quindi le etichette del
    dataset devono riprodurlo esattamente, indipendentemente dal comportamento
    atteso (answer/abstain/clarify/unknown_course).
    """
    mismatches = []
    for r in ROWS:
        intent = infer_query_intent(r["question"], {})
        if intent.course_tag != r.get("expected_course"):
            mismatches.append(
                f"{r['id']}: course atteso {r.get('expected_course')!r} != intent {intent.course_tag!r}"
            )
        if intent.topic != r.get("expected_topic"):
            mismatches.append(
                f"{r['id']}: topic atteso {r.get('expected_topic')!r} != intent {intent.topic!r}"
            )
    assert not mismatches, "etichette intent non allineate:\n" + "\n".join(mismatches)


def test_behavior_consistent_with_intent():
    """I comportamenti senza LLM sono coerenti con i predicati dell'intent."""
    bad = []
    for r in ROWS:
        intent = infer_query_intent(r["question"], {})
        if r["expected_behavior"] == "unknown_course" and intent.detected_unknown_course is None:
            bad.append(f"{r['id']}: atteso unknown_course ma l'intent non rileva un corso fuori dominio")
        if r["expected_behavior"] == "clarify" and not intent.is_ambiguous:
            bad.append(f"{r['id']}: atteso clarify ma l'intent non è ambiguo")
    assert not bad, "\n".join(bad)


def test_negatives_have_valid_reason():
    """Le negative dichiarano una causa valida; le answerable nessuna (per FASE 6)."""
    bad = []
    for r in ROWS:
        reason = r.get("expected_abstention_reason")
        if r["category"] in NEGATIVE_CATEGORIES:
            if reason not in VALID_REASONS:
                bad.append(f"{r['id']}: causa di astensione non valida: {reason!r}")
        elif reason is not None:
            bad.append(f"{r['id']}: answerable non deve dichiarare una causa di astensione")
    assert not bad, "\n".join(bad)


def test_expected_docs_exist_on_disk():
    """I documenti attesi esistono davvero nel corpus (no riferimenti fantasma)."""
    missing = []
    for r in ROWS:
        for doc in r["expected_docs"]:
            if not os.path.exists(os.path.join(DOCS_DIR, doc)):
                missing.append(f"{r['id']}: manca {doc}")
    assert not missing, "\n".join(missing)


def test_answerable_with_docs_have_course_or_topic():
    """Le answerable con documenti attesi hanno sempre corso o argomento risolto."""
    bad = [
        r["id"]
        for r in ROWS
        if r["expected_behavior"] == "answer"
        and r["expected_docs"]
        and not (r.get("expected_course") or r.get("expected_topic"))
    ]
    assert not bad, f"answerable senza corso né argomento: {bad}"


def test_held_out_negative_set_covers_multiple_reasons():
    """FASE 4 introduce negativi held-out (q34–q39) per validare la FASE 6."""
    held_out_ids = {f"q{n}" for n in range(34, 40)}
    present = {r["id"] for r in ROWS if r["id"] in held_out_ids}
    assert present == held_out_ids, f"negativi held-out mancanti: {held_out_ids - present}"
    reasons = {
        r["expected_abstention_reason"] for r in ROWS if r["id"] in held_out_ids
    }
    assert {OUT_OF_DOMAIN_COURSE, OUT_OF_DOMAIN, INSUFFICIENT_EVIDENCE, AMBIGUOUS} <= reasons
