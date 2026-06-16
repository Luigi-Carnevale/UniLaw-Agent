"""Stima euristica dell'affidabilità della risposta.

Estratta da `agent.py` in FASE 2 senza modifiche al comportamento. La funzione
valuta la coerenza tra le fonti selezionate e l'intento (corso/argomento) e
restituisce un livello ("alta"/"media"/"bassa") con motivazione testuale.
"""

from typing import List

from rag_types import QueryIntent, RetrievedSource


def estimate_confidence(
    intent: QueryIntent, sources: List[RetrievedSource]
) -> tuple[str, str]:
    if not sources:
        return "bassa", "Nessuna fonte selezionata."

    score = 0
    topic_match_count = 0
    course_match_count = 0
    off_course_count = 0
    supporting_terms = 0

    combined_text = " ".join(
        f"{source.filename.lower()} {source.content.lower()}"
        for source in sources
    )

    for source in sources:
        filename = source.filename.lower()

        if intent.course_tag:
            if source.course_tag == intent.course_tag:
                course_match_count += 1
                score += 3
            elif source.course_tag != "generale":
                off_course_count += 1
                score -= 4

        if intent.topic and source.doc_type == intent.topic:
            topic_match_count += 1
            score += 3

        if intent.topic == "accesso" and source.doc_type in {"accesso", "regolamento"}:
            score += 1

        if intent.topic == "tesi" and source.doc_type in {"tesi", "regolamento", "guida"}:
            score += 1

        if intent.topic == "tesi" and "regolamento-tesi" in filename:
            score += 3

        if intent.topic == "erasmus" and "erasmus" in filename:
            score += 3

        if intent.topic == "piano_studi" and (
            "piano" in filename
            or "regolamento" in filename
        ):
            score += 2

    if intent.course_tag == "scienze_educazione" and intent.topic == "accesso":
        key_terms = [
            "immatricolazione",
            "scienze dell'educazione",
            "prova di ammissione",
            "80 quesiti",
            "risposta multipla",
            "2 ore",
            "cultura generale",
            "lingua inglese",
            "comprensione del testo",
        ]
        supporting_terms = sum(1 for term in key_terms if term in combined_text)
        score += min(supporting_terms, 5)

    if intent.course_tag == "informatica" and intent.topic == "accesso":
        key_terms = ["tolc", "ofa", "ris_test", "immatricol", "16", "9"]
        supporting_terms = sum(1 for term in key_terms if term in combined_text)
        score += min(supporting_terms, 5)

    if intent.topic and topic_match_count == 0:
        score -= 2

    if intent.course_tag and course_match_count == 0:
        score -= 4

    if off_course_count:
        return (
            "media" if score >= 7 else "bassa",
            "Sono presenti una o più fonti di corsi diversi: risposta da verificare con attenzione.",
        )

    if score >= 9:
        return "alta", "Le fonti selezionate sono coerenti con corso e argomento della domanda."

    if score >= 4:
        return "media", "Sono state trovate fonti pertinenti, ma l'informazione è distribuita o non perfettamente specifica."

    return "bassa", "Le fonti recuperate potrebbero non coprire completamente la domanda."
