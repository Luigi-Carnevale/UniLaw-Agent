"""Evidence selection (FASE 5).

Riduce ogni chunk recuperato ai passaggi più pertinenti alla domanda, così da
fornire al modello evidenze **più brevi e mirate** (meno rumore, meno deriva verso
contenuti tangenziali). La selezione è conservativa: garantisce un numero minimo di
frasi e un tetto di caratteri, e restituisce il chunk intero se è già corto.

Funzioni pure e testabili offline (nessun modello, nessuna rete).
"""

import re

from retrieval import tokenize

# Confine di frase: punteggiatura forte o a-capo. Semplice e trasparente.
_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text or "")]
    return [s for s in parts if s]


def select_passage(
    question: str,
    content: str,
    max_sentences: int = 6,
    min_sentences: int = 3,
    max_chars: int = 700,
) -> str:
    """Estrae dal chunk i passaggi più pertinenti alla domanda.

    Strategia: punteggio per sovrapposizione lessicale con la query; si tengono le
    frasi migliori (con un minimo garantito), **preservandone l'ordine originale**,
    entro un tetto di caratteri. Se il contenuto è già breve, viene restituito
    invariato.
    """
    content = content or ""
    if len(content) <= max_chars:
        return content

    sentences = split_sentences(content)
    if len(sentences) <= min_sentences:
        return content

    query_tokens = set(tokenize(question))

    scored = []
    for i, sentence in enumerate(sentences):
        overlap = len(query_tokens & set(tokenize(sentence)))
        scored.append((overlap, i, sentence))

    # Ordina per pertinenza; tiene almeno `min_sentences`, al più `max_sentences`.
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)

    keep_indices = set()
    for overlap, idx, _ in scored:
        if len(keep_indices) >= max_sentences:
            break
        # oltre il minimo garantito, tieni solo le frasi che toccano la query
        if len(keep_indices) >= min_sentences and overlap == 0:
            break
        keep_indices.add(idx)

    # Ricostruisce in ordine originale, rispettando il tetto di caratteri.
    selected = []
    total = 0
    for idx in sorted(keep_indices):
        sentence = sentences[idx]
        if total + len(sentence) > max_chars and selected:
            break
        selected.append(sentence)
        total += len(sentence) + 1

    return " ".join(selected) if selected else content[:max_chars]
