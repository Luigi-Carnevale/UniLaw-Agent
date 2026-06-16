"""Reranking euristico e filtro metadata per corso.

Estratti da `agent.py` in FASE 3 senza modifiche al comportamento. Il reranker
applica priori di dominio (corso, tipo documento, parole chiave, nome file) e
resta l'ordinamento finale dei candidati; in FASE 4 sarà affiancato (in modo
opzionale) da un reranker neurale, mantenendo questa euristica come fallback.

`filter_documents_by_course` è il filtro metadata della pipeline ibrida: evita
che una domanda su un corso specifico sia risolta con documenti di altri corsi.
"""

import os

from config import MAX_CONTEXT_DOCUMENTS
from intent import (
    asks_borsa_graduatoria,
    asks_erasmus_end_mobility,
    asks_tesi_consultazione,
)
from rag_types import QueryIntent, RagTrace


def filter_documents_by_course(docs: list, intent: QueryIntent) -> list:
    """
    Evita che una domanda su un corso specifico venga risposta usando
    documenti di altri corsi. Mantiene i documenti generali solo se utili.
    """
    if not intent.course_tag:
        return docs

    if intent.topic in {"erasmus", "borsa"}:
        return docs

    allowed = {intent.course_tag, "generale"}
    filtered = []

    for doc in docs:
        metadata = doc.metadata or {}
        course_tag = metadata.get("course_tag", "generale")

        if course_tag in allowed:
            filtered.append(doc)

    if len(filtered) >= 2:
        return filtered

    # Fallback controllato: se il filtro è troppo aggressivo, non svuotiamo
    # completamente il contesto, ma manteniamo comunque l'ordine del reranking.
    return filtered or docs


def rerank_documents(question: str, docs: list, intent: QueryIntent, trace: RagTrace):
    q = question.lower()
    ranked = []

    tesi_consultazione = asks_tesi_consultazione(question)

    for doc in docs:
        score = 0
        metadata = doc.metadata or {}

        filename = os.path.basename(metadata.get("source", "")).lower()
        course_tag = metadata.get("course_tag", "generale")
        doc_type = metadata.get("doc_type", "altro")
        text = doc.page_content.lower()

        if intent.course_tag:
            if course_tag == intent.course_tag:
                score += 10
            elif course_tag == "generale":
                score += 1
            else:
                score -= 12

        if intent.topic:
            if doc_type == intent.topic:
                score += 12
            elif intent.topic == "accesso" and doc_type == "regolamento":
                score += 4
            elif intent.topic == "tesi" and doc_type == "regolamento":
                score += 3
            elif intent.topic == "piano_studi" and doc_type == "regolamento":
                score += 2

        if intent.course_tag == "informatica" and intent.topic == "accesso":
            if "regolamento-di-accesso" in filename and "informatica" in filename:
                score += 30

            if "accesso" in filename and ("informatica" in filename or "l31" in filename or "l-31" in filename):
                score += 25

            if "tolc" in text:
                score += 8

            if "ofa" in text:
                score += 8

            if "ris_test" in text or "ris test" in text:
                score += 6

            if "immatricol" in text:
                score += 5

            if "tabella" in text:
                score += 5

            if "< 9" in text or "inferiore a 9" in text or "minore di 9" in text:
                score += 5

            if "16" in text:
                score += 3

            if doc_type in {"borsa", "erasmus", "tesi", "guida", "piano_studi"}:
                score -= 30

            if "bando borsa" in filename or "borsa di studio" in filename:
                score -= 40

            if "erasmus" in filename:
                score -= 40

            if "tesi" in filename or "prova" in filename:
                score -= 25

        if intent.course_tag == "scienze_educazione" and intent.topic == "accesso":
            if "immatricolazione" in filename and ("educazione" in filename or "l-19" in filename or "l19" in filename):
                score += 65

            if "scienze" in filename and "educazione" in filename:
                score += 18

            if "l-19" in filename or "l19" in filename:
                score += 12

            if "regolamento" in filename and ("educazione" in filename or "l-19" in filename or "l19" in filename):
                score += 10

            if "prova di ammissione" in text:
                score += 22

            if "80 quesiti" in text or "n. 80 quesiti" in text:
                score += 22

            if "risposta multipla" in text:
                score += 14

            if "2 ore e 30 minuti" in text or "2 ore 30 minuti" in text:
                score += 14

            if "cultura generale" in text:
                score += 8

            if "lingua inglese" in text:
                score += 8

            if "abilità logiche" in text or "abilita logiche" in text:
                score += 8

            if "comprensione del testo" in text:
                score += 8

            if "immatricol" in text:
                score += 10

            if "ammissione" in text or "accesso" in text:
                score += 8

            if "prova finale" in text or "elaborato scritto" in text or "relatore" in text:
                score -= 18

            if "regolamento-di-accesso" in filename and "informatica" in filename:
                score -= 80

            if "informatica" in filename or "l31" in filename or "l-31" in filename:
                score -= 55

            if "amministrazione" in filename or "l16" in filename or "l-16" in filename:
                score -= 45

            if doc_type in {"tesi", "erasmus", "borsa", "piano_studi"}:
                score -= 25

        if intent.course_tag == "scienze_educazione" and intent.topic == "tesi":
            if ("prova" in filename and "finale" in filename and "educazione" in filename) or ("tesi" in filename and "educazione" in filename):
                score += 40

            if "linee-guida" in filename and "tesi" in filename:
                score += 24

            if "regolamento" in filename and ("educazione" in filename or "l-19" in filename or "l19" in filename):
                score += 14

            if "prova finale" in text:
                score += 12

            if "elaborato" in text:
                score += 10

            if "relatore" in text:
                score += 8

            if "prova di ammissione" in text or "80 quesiti" in text:
                score -= 35

            if "immatricolazione" in filename:
                score -= 30

            if "regolamento-di-accesso" in filename and "informatica" in filename:
                score -= 60

        if intent.course_tag == "scienze_educazione" and intent.topic == "piano_studi":
            if "piano" in filename and ("educazione" in filename or "l-19" in filename or "l19" in filename):
                score += 45

            if "regolamento" in filename and ("educazione" in filename or "l-19" in filename or "l19" in filename):
                score += 12

            if "cfu" in text:
                score += 8

            if "insegnamenti" in text or "attività formative" in text or "attivita formative" in text:
                score += 8

            if "regolamento-di-accesso" in filename and "informatica" in filename:
                score -= 60

            if "immatricolazione" in filename:
                score -= 20

            if doc_type in {"accesso", "tesi", "erasmus", "borsa"}:
                score -= 18

        if intent.course_tag == "informatica" and intent.topic == "tesi":
            if tesi_consultazione:
                if "regolamento-tesi" in filename:
                    score += 60

                if "consult" in text:
                    score += 20

                if "embargo" in text:
                    score += 20

                if "deposito" in text:
                    score += 12

                if "lucro" in text:
                    score += 12

                if "guida-3.0-tesi-online" in filename:
                    score -= 10

                if "prova" in filename and "finale" in filename and "informatica" in filename:
                    score -= 8

            else:
                if "regolamento" in filename and "prova" in filename and "finale" in filename and "informatica" in filename:
                    score += 45

                if "regolamento-della" in filename and "prova" in filename and "informatica" in filename:
                    score += 45

                if "guida" in filename and "tesi" in filename:
                    score += 18

                if "tesi-online" in filename:
                    score += 16

                if "elaborato" in text:
                    score += 8

                if "relatore" in text:
                    score += 8

                if "discussione" in text:
                    score += 7

                if "prova finale" in text:
                    score += 8

                if "commissione" in text:
                    score += 4

                if "regolamento-tesi" in filename or ("regolamento" in filename and "tesi-2023" in filename):
                    score -= 28

                if "consultabile" in text or "embargo" in text:
                    score -= 10

            if doc_type in {"accesso", "borsa", "erasmus", "piano_studi"}:
                score -= 25

            if "regolamento-di-accesso" in filename:
                score -= 35

            if "bando borsa" in filename or "erasmus" in filename:
                score -= 35

        if intent.topic == "erasmus":
            asks_end_mobility = asks_erasmus_end_mobility(question)

            if "erasmus" in filename:
                score += 30

            if "mobilità" in text or "mobilita" in text:
                score += 8

            if "internazionale" in text:
                score += 5

            if "graduatoria" in text:
                score += 4

            if "learning agreement" in text:
                score += 4

            if asks_end_mobility:
                if "attestato di permanenza" in text:
                    score += 18
                if "allegato c" in text:
                    score += 12
                if "breve relazione" in text or "relazione" in text:
                    score += 10
                if "giustificativi" in text:
                    score += 10
                if "riconoscimento attività" in text or "riconoscimento attivita" in text:
                    score += 10
                if "10 febbraio" in text or "domanda di partecipazione" in text:
                    score -= 8

            if doc_type in {"accesso", "tesi", "piano_studi", "borsa"}:
                score -= 20

            if "regolamento-di-accesso" in filename:
                score -= 35

            if "prova" in filename or "tesi" in filename:
                score -= 30

        if intent.topic == "borsa":
            asks_graduatoria = asks_borsa_graduatoria(question)

            if "borsa" in filename:
                score += 28

            if asks_graduatoria:
                if "graduatoria" in text:
                    score += 18
                if "provvisoria" in text:
                    score += 10
                if "definitiva" in text:
                    score += 10
                if "assestata" in text:
                    score += 10
                if "idone" in text:
                    score += 6
                if "beneficiar" in text:
                    score += 6

            if doc_type in {"accesso", "tesi", "piano_studi", "erasmus"}:
                score -= 20

        if intent.course_tag == "informatica" and intent.topic == "piano_studi":
            if "piano" in filename and ("studi" in filename or "studio" in filename):
                score += 30

            if "informatica" in filename and "regolamento" in filename:
                score += 8

            if "cfu" in text:
                score += 7

            if "insegnamenti" in text:
                score += 7

            if "attività formative" in text or "attivita formative" in text:
                score += 5

            if doc_type in {"accesso", "tesi", "erasmus", "borsa"}:
                score -= 25

            if "regolamento-di-accesso" in filename:
                score -= 35

            if "prova" in filename or "tesi" in filename:
                score -= 30

            if "erasmus" in filename or "bando borsa" in filename:
                score -= 30

        if "informatica" in q and ("informatica" in filename or "l31" in filename or "l-31" in filename):
            score += 6

        if any(k in q for k in ["tolc", "ofa", "immatricol", "accesso", "ammissione", "punteggio", "prova di ammissione", "test"]):
            if "tolc" in text:
                score += 4
            if "ofa" in text:
                score += 4
            if "immatricol" in text:
                score += 4
            if "ammissione" in text:
                score += 4
            if "prova di ammissione" in text:
                score += 6
            if "accesso" in filename or "immatricolazione" in filename:
                score += 6

        if any(k in q for k in ["borsa", "isee", "ispe"]):
            if "borsa" in text or "isee" in text or "ispe" in text:
                score += 4

        if any(k in q for k in ["tesi", "prova finale", "elaborato"]):
            if "prova finale" in text or "tesi" in text or "elaborato" in text:
                score += 4

        if doc_type == "guida" and intent.topic in {"accesso", "tesi", "piano_studi"}:
            score -= 2

        ranked.append((score, doc))

    ranked.sort(key=lambda x: x[0], reverse=True)

    trace.rejected_hint = [
        os.path.basename(doc.metadata.get("source", "Documento sconosciuto"))
        for score, doc in ranked[MAX_CONTEXT_DOCUMENTS:MAX_CONTEXT_DOCUMENTS + 5]
    ]

    return [doc for _, doc in ranked]
