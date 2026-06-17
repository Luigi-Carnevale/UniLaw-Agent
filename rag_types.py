"""Modello dati condiviso della pipeline RAG di UniLaw Agent.

Raccoglie le dataclass usate dall'orchestratore (`agent.py`) e dai moduli di
supporto (intent, retrieval, citazioni, confidenza), più le etichette leggibili
di corso e argomento. Estratte da `agent.py` in FASE 2 senza modifiche al
comportamento; restano importabili da `agent` per retrocompatibilità.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryIntent:
    course_tag: Optional[str]
    topic: Optional[str]
    is_ambiguous: bool = False
    used_memory: bool = False
    detected_unknown_course: Optional[str] = None


@dataclass
class RetrievedSource:
    index: int
    filename: str
    page: int | None
    content: str
    course_tag: str
    doc_type: str

    @property
    def citation_label(self) -> str:
        if self.page is None:
            return f"[F{self.index}] {self.filename}"

        return f"[F{self.index}] {self.filename}, pag. {self.page + 1}"


@dataclass
class RagTrace:
    question: str = ""
    course_tag: Optional[str] = None
    topic: Optional[str] = None
    used_memory: bool = False
    confidence: str = "bassa"
    confidence_reason: str = ""
    answer_profile: str = ""
    query_variants: list[str] = field(default_factory=list)
    selected_sources: list[str] = field(default_factory=list)
    rejected_hint: list[str] = field(default_factory=list)
    deterministic_rule_used: Optional[str] = None
    retrieval_mode: str = "vettoriale"
    fusion_scores: list[str] = field(default_factory=list)
    reranker: str = "euristico"
    evidence_chars: str = ""
    grounding: str = "n.d."
    abstention_reason: str = ""


COURSE_LABELS = {
    "informatica": "Informatica L-31",
    "scienze_educazione": "Scienze dell'educazione L-19",
    "amministrazione": "Scienze dell'amministrazione L-16",
    "economia": "Area economico-statistica",
}

TOPIC_LABELS = {
    "accesso": "Accesso / TOLC / OFA / immatricolazione",
    "borsa": "Borsa di studio / benefici",
    "erasmus": "Erasmus / mobilità internazionale",
    "tesi": "Prova finale / tesi",
    "piano_studi": "Piano di studi / CFU / insegnamenti",
    "calcolo": "Calcolo numerico",
}
