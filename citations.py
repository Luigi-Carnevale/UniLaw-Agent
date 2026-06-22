"""Gestione e verifica delle citazioni.

- `extract_cited_source_indexes` (FASE 2): mantiene solo i riferimenti [F#] che
  corrispondono a fonti realmente recuperate;
- `format_sources_block` (FASE 2): blocco "Fonti citate" o, in fallback,
  "Fonti utilizzate"; in astensione (Ciclo 2 — FASE 3) il fallback viene
  rietichettato onestamente come "Documenti consultati (nessuno utilizzato)";
- `normalize_citations` (Ciclo 2 — FASE 2): riconduce le varianti `(F#)` e `F#`
  al formato canonico `[F#]`, solo per indici di fonti realmente recuperate;
- `strip_invalid_citations` (FASE 5): rimuove dal testo i riferimenti [F#]
  inventati (che non corrispondono ad alcuna fonte recuperata);
- `grounding_report` (FASE 5): verifica che le frasi che citano abbiano riscontro
  lessicale nella fonte citata, restituendo un rapporto di supporto; dalla
  Ciclo 2 — FASE 12 il supporto può essere riconosciuto anche per **similarità di
  embedding** (opt-in), come rete di recupero per le frasi corrette ma parafrasate.
"""

import logging
import re
from typing import List, Optional

from evidence import split_sentences
from rag_types import RetrievedSource
from retrieval import tokenize
from semantic_intent import Embedder, cosine_similarity

logger = logging.getLogger(__name__)


# Separatori ammessi all'interno di una parentesi di citazioni: virgola, punto e
# virgola, slash, congiunzioni ("e"/"ed"/"o"/"oppure") o semplici spazi.
_CITATION_SEP = r"(?:\s*(?:,|;|/|\be\b|\bed\b|\bo\b|\boppure\b)\s*|\s+)"
# Un gruppo tra parentesi composto SOLO da riferimenti a fonti, es. "(F1)",
# "(F1, F2)", "(F1 e F3)".
_CITATION_PAREN_GROUP = re.compile(
    r"\(\s*(F\d+(?:" + _CITATION_SEP + r"F\d+)*)\s*\)"
)
_CITATION_REF = re.compile(r"F(\d+)")
# Riferimento "nudo" F#, non già tra parentesi quadre/tonde e non parte di una
# parola o di un codice (lookbehind: non preceduto da [, ( o da un carattere di
# parola; lookahead `\b`: non seguito da lettere/cifre come in "F2a").
_CITATION_BARE = re.compile(r"(?<![\[\(\w])F(\d+)\b")


def normalize_citations(answer: str, sources: List[RetrievedSource]) -> str:
    """Riconduce le citazioni al formato canonico `[F#]`.

    Il modello a volte produce `(F1)` o un `F1` "nudo" invece di `[F1]`: queste
    varianti sfuggono all'estrazione, alla verifica del grounding e al blocco
    fonti (le regex riconoscono solo `[F#]`), innescando il fallback fuorviante
    "Fonti utilizzate". La normalizzazione avviene **prima** della verifica.

    Per non alterare la prosa oltre il necessario, vengono ricondotte a `[F#]`
    solo le occorrenze il cui indice corrisponde a una fonte realmente
    recuperata; le parentesi che contengono indici inesistenti restano intatte.
    """
    valid = {source.index for source in sources}

    def _wrap_if_valid(num: str) -> Optional[str]:
        return f"[F{num}]" if int(num) in valid else None

    def _replace_paren_group(match: re.Match) -> str:
        wrapped = [_wrap_if_valid(n) for n in _CITATION_REF.findall(match.group(1))]
        if any(w is None for w in wrapped):
            # Almeno un riferimento non è una fonte valida: non tocco il gruppo.
            return match.group(0)
        return " ".join(wrapped)

    out = _CITATION_PAREN_GROUP.sub(_replace_paren_group, answer)

    def _replace_bare(match: re.Match) -> str:
        wrapped = _wrap_if_valid(match.group(1))
        return wrapped if wrapped is not None else match.group(0)

    return _CITATION_BARE.sub(_replace_bare, out)


def extract_cited_source_indexes(
    answer: str, sources: List[RetrievedSource]
) -> set[int]:
    cited = {
        int(match)
        for match in re.findall(r"\[F(\d+)\]", answer)
    }

    valid = {
        source.index
        for source in sources
    }

    return cited & valid


def strip_invalid_citations(answer: str, sources: List[RetrievedSource]) -> str:
    """Rimuove i riferimenti [F#] che non corrispondono a una fonte recuperata.

    Previene le "citazioni inventate": se il modello produce [F9] ma esistono solo
    F1..F5, il riferimento viene tolto (e gli spazi doppi ricomposti).
    """
    valid = {source.index for source in sources}

    def _repl(match: re.Match) -> str:
        return match.group(0) if int(match.group(1)) in valid else ""

    cleaned = re.sub(r"\[F(\d+)\]", _repl, answer)
    # ricompone spazi lasciati dalla rimozione
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned


def _semantic_support(
    sentence: str,
    cited: List[int],
    by_index: dict,
    embedder: Embedder,
    min_semantic: float,
) -> bool:
    """True se la frase citante è semanticamente vicina a una frase delle fonti citate.

    Rete di recupero (Ciclo 2 — FASE 12) per le frasi corrette ma **parafrasate**,
    che il solo overlap lessicale boccia: confronta la frase con ciascuna frase
    delle fonti citate per similarità di embedding e accetta se la più vicina supera
    `min_semantic`. Il confronto è frase↔frase (non frase↔chunk intero) per non
    diluire la similarità su contenuti lunghi. Fallback sicuro: se l'embedder non è
    disponibile o solleva un'eccezione, restituisce `False` (resta il solo lessicale).
    """
    source_sentences: List[str] = []
    for i in cited:
        source_sentences.extend(
            s for s in split_sentences(by_index[i].content) if s.strip()
        )
    if not source_sentences:
        return False

    try:
        vectors = embedder([sentence, *source_sentences])
    except Exception as exc:  # embedder non disponibile o errore a runtime
        logger.warning(
            "Grounding semantico non disponibile (%s): uso il solo riscontro lessicale.",
            exc,
        )
        return False

    if not vectors or len(vectors) != len(source_sentences) + 1:
        return False

    query_vec = vectors[0]
    return any(cosine_similarity(query_vec, v) >= min_semantic for v in vectors[1:])


def grounding_report(
    answer: str,
    sources: List[RetrievedSource],
    min_overlap: float = 0.18,
    embedder: Optional[Embedder] = None,
    min_semantic: float = 0.45,
) -> tuple[Optional[float], List[str]]:
    """Verifica il supporto delle frasi che contengono una citazione.

    Per ogni frase che cita una o più fonti [F#], misura la sovrapposizione
    lessicale (overlap coefficient) tra la frase e il contenuto delle fonti citate.
    Restituisce (rapporto_di_supporto, frasi_non_supportate). Il rapporto è None se
    non ci sono frasi con citazioni (impossibile valutare il supporto).

    Ciclo 2 — FASE 12 (opt-in): se è fornito un `embedder`, le frasi che NON superano
    la soglia lessicale ricevono un secondo controllo per **similarità di embedding**
    (soglia `min_semantic`); è una rete di recupero per le parafrasi e si aggiunge al
    lessicale (non lo sostituisce e non può togliere un supporto già riconosciuto). Con
    `embedder=None` (default) il comportamento è byte-identico al solo lessicale.
    """
    by_index = {source.index: source for source in sources}

    supported = 0
    total = 0
    unsupported: List[str] = []

    for sentence in split_sentences(answer):
        cited = [int(m) for m in re.findall(r"\[F(\d+)\]", sentence)]
        cited = [i for i in cited if i in by_index]
        if not cited:
            continue

        total += 1

        sentence_tokens = set(tokenize(sentence))
        if not sentence_tokens:
            supported += 1  # nessun contenuto da verificare (solo simboli)
            continue

        source_tokens: set = set()
        for i in cited:
            source_tokens |= set(tokenize(by_index[i].content))

        overlap = len(sentence_tokens & source_tokens) / len(sentence_tokens)
        if overlap >= min_overlap:
            supported += 1
        elif embedder is not None and _semantic_support(
            sentence, cited, by_index, embedder, min_semantic
        ):
            supported += 1
        else:
            unsupported.append(sentence)

    if total == 0:
        return None, []

    return supported / total, unsupported


def format_sources_block(
    answer: str,
    sources: List[RetrievedSource],
    abstaining: bool = False,
) -> str:
    """Blocco delle fonti da accodare alla risposta.

    Se la risposta cita davvero delle fonti (`[F#]` validi), queste vengono
    sempre elencate come "Fonti citate". In assenza di citazioni si ripiega
    sulle prime fonti recuperate; ma quando il sistema si è **astenuto**
    (`abstaining=True`) etichettarle "Fonti utilizzate" sarebbe una falsa
    attribuzione (nessuna fonte è stata usata per rispondere): in quel caso il
    blocco viene rietichettato onestamente come "Documenti consultati (nessuno
    utilizzato)" (Ciclo 2 — FASE 3). Le citazioni reali restano comunque mostrate.
    """
    if not sources:
        return ""

    cited_indexes = extract_cited_source_indexes(answer, sources)

    selected_sources = [
        s
        for s in sources
        if s.index in cited_indexes
    ]

    if selected_sources:
        title = "Fonti citate:"
    elif abstaining:
        selected_sources = sources[:3]
        title = "Documenti consultati (nessuno utilizzato per la risposta):"
    else:
        selected_sources = sources[:3]
        title = "Fonti utilizzate:"

    lines = ["\n\n---", title]

    for source in selected_sources:
        lines.append(f"- {source.citation_label}")

    return "\n".join(lines)
