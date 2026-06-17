"""Esportazione del RagTrace in JSON e Markdown (FASE 8).

Rende fruibile e portabile tutta l'osservabilità raccolta durante una risposta
(domanda, corso/argomento, modalità di retrieval, fusione, reranker, evidence,
grounding, causa di astensione, regola deterministica, query generate, fonti
selezionate e scartate). Funzioni pure: testabili offline, riusabili dalla UI.
"""

import json
from typing import Optional

from rag_types import RagTrace


def trace_to_dict(trace: Optional[RagTrace]) -> dict:
    """Serializza il trace in un dizionario ordinato e completo."""
    if trace is None:
        return {}

    def field(name, default=""):
        return getattr(trace, name, default)

    return {
        "question": trace.question,
        "interpretation": {
            "course_tag": trace.course_tag,
            "topic": trace.topic,
            "used_memory": trace.used_memory,
        },
        "confidence": {
            "level": trace.confidence,
            "reason": trace.confidence_reason,
        },
        "answer_profile": trace.answer_profile,
        "retrieval": {
            "mode": field("retrieval_mode"),
            "query_variants": list(trace.query_variants or []),
            "fusion_scores": list(field("fusion_scores", []) or []),
            "reranker": field("reranker"),
        },
        "evidence": field("evidence_chars"),
        "grounding": field("grounding"),
        "abstention_reason": field("abstention_reason"),
        "deterministic_rule_used": trace.deterministic_rule_used,
        "selected_sources": list(trace.selected_sources or []),
        "rejected_after_rerank": list(trace.rejected_hint or []),
    }


def trace_to_json(trace: Optional[RagTrace]) -> str:
    return json.dumps(trace_to_dict(trace), ensure_ascii=False, indent=2)


def _bullets(items: list) -> list[str]:
    return [f"- `{item}`" for item in items] if items else ["- (nessuno)"]


def trace_to_markdown(trace: Optional[RagTrace]) -> str:
    """Report Markdown leggibile del trace, adatto all'allegato di una relazione."""
    data = trace_to_dict(trace)
    if not data:
        return "# UniLaw Agent — RAG trace\n\n(nessun trace disponibile)\n"

    interp = data["interpretation"]
    conf = data["confidence"]
    retr = data["retrieval"]

    lines = [
        "# UniLaw Agent — RAG trace",
        "",
        f"**Domanda:** {data['question']}",
        "",
        "## Interpretazione",
        f"- Corso: `{interp['course_tag'] or 'non rilevato'}`",
        f"- Argomento: `{interp['topic'] or 'non rilevato'}`",
        f"- Memoria usata: `{interp['used_memory']}`",
        "",
        "## Affidabilità e astensione",
        f"- Confidenza: **{conf['level']}**",
        f"- Motivo: {conf['reason'] or 'n.d.'}",
        f"- Causa di astensione: `{data['abstention_reason'] or 'nessuna'}`",
        f"- Grounding citazioni: `{data['grounding'] or 'n.d.'}`",
        f"- Regola deterministica: `{data['deterministic_rule_used'] or 'nessuna'}`",
        f"- Profilo risposta: `{data['answer_profile'] or 'n.d.'}`",
        "",
        "## Retrieval",
        f"- Modalità: `{retr['mode'] or 'n.d.'}`",
        f"- Reranker: `{retr['reranker'] or 'n.d.'}`",
        f"- Evidence: `{data['evidence'] or 'n.d.'}`",
        "",
        "### Query generate",
        *_bullets(retr["query_variants"]),
        "",
        "### Scoring di fusione (RRF)",
        *_bullets(retr["fusion_scores"]),
        "",
        "### Fonti selezionate",
        *_bullets(data["selected_sources"]),
        "",
        "### Documenti scartati dopo reranking",
        *_bullets(data["rejected_after_rerank"]),
        "",
    ]
    return "\n".join(lines) + "\n"
