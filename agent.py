import logging
import os
from typing import Any, List, Optional

import langchain
import redis
import streamlit as st
from langchain_community.cache import RedisCache
from langchain_community.chat_models import ChatOllama

from citations import (
    extract_cited_source_indexes,
    format_sources_block,
    grounding_report,
    normalize_citations,
    strip_invalid_citations,
)
from abstention import (
    AMBIGUOUS,
    OUT_OF_DOMAIN_COURSE,
    WEAK_RETRIEVAL,
    classify_llm_abstention,
    format_reason,
    is_abstention,
)
from config import (
    ABSTENTION_OOD_MAX_STRENGTH,
    ABSTENTION_OOD_SEMANTIC_MAX_STRENGTH,
    ABSTENTION_SEMANTIC_STRENGTH_ENABLED,
    ANSWER_STYLE_GUIDE,
    CITATION_GROUNDING_ENABLED,
    CITATION_GROUNDING_MIN_RATIO,
    CITATION_GROUNDING_SEMANTIC_ENABLED,
    CITATION_GROUNDING_SEMANTIC_MIN_SIMILARITY,
    DETERMINISTIC_RULES_ENABLED,
    GENERAL_TESI_HINT_ENABLED,
    PROSE_TEMPLATES_ENABLED,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    EVIDENCE_MAX_CHARS,
    EVIDENCE_MAX_SENTENCES,
    EVIDENCE_MIN_SENTENCES,
    EVIDENCE_SELECTION_ENABLED,
    MAX_CONTEXT_DOCUMENTS,
    QA_PROMPT,
    RERANKER_ENABLED,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
    SEMANTIC_INTENT_COURSE_MIN_SIMILARITY,
    SEMANTIC_INTENT_ENABLED,
    SEMANTIC_INTENT_TOPIC_MIN_SIMILARITY,
)
from confidence import estimate_confidence
from evidence import select_passage
from intent import (
    asks_borsa_graduatoria,
    asks_erasmus_end_mobility,
    asks_tesi_consultazione,
    infer_query_intent,
)
from knowledge import l19_test_table_markdown, tolc_bands_table
from neural_reranker import CrossEncoderReranker
from semantic_intent import SemanticIntentClassifier
from rag_types import (  # re-export: mantiene `from agent import QueryIntent, ...`
    COURSE_LABELS,
    TOPIC_LABELS,
    QueryIntent,
    RagTrace,
    RetrievedSource,
)
from reranking import filter_documents_by_course, rerank_documents
from retrieval import build_bm25_index, hybrid_retrieve
from rules_tolc import classify_tolc_score, extract_tolc_score
from tools import prova_calcolo_sicuro


logger = logging.getLogger(__name__)


try:
    from langchain.globals import set_llm_cache
except ImportError:
    def set_llm_cache(cache):
        langchain.llm_cache = cache


def setup_redis_cache():
    """
    Attiva Redis come cache LLM solo se il servizio è disponibile.
    Se Redis non è disponibile, l'app continua a funzionare normalmente.
    """
    try:
        r = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            socket_connect_timeout=0.3,
        )

        if r.ping():
            set_llm_cache(RedisCache(redis_=r))
            logger.info("Redis cache attivata.")
            return True

    except Exception:
        logger.info("Cache LLM disattivata: Redis non disponibile.")

    return False


# QueryIntent, RetrievedSource, RagTrace, COURSE_LABELS e TOPIC_LABELS sono
# definiti in rag_types.py (estratti in FASE 2) e importati/riesportati sopra.


# Ciclo 2 — FASE 14 — mitigazione della falsa astensione su regolamento generale.
# Le regole su consultabilità, deposito ed embargo della tesi vivono in un
# regolamento di Ateneo NON specifico per corso (`regolamento-tesi-2023.pdf` →
# course_tag "generale", doc_type "tesi"). Su una domanda di consultabilità che
# nomina un corso preciso (es. q14, "La tesi di Informatica L-31 è consultabile
# dopo la laurea?") il modello 8B tende ad astenersi cercando un dettaglio
# "specifico per quel corso" che il regolamento generale, per sua natura, non
# riporta — pur contenendo la regola. Questo predicato puro riconosce quando fra
# le fonti recuperate è presente un regolamento generale sulla tesi, così che il
# profilo di risposta possa autorizzare esplicitamente l'uso della regola
# generale invece di lasciare che il modello si astenga.
_GENERAL_TESI_DOC_TYPES = {"tesi", "regolamento"}


def has_general_tesi_regulation(sources: List[RetrievedSource]) -> bool:
    """True se fra le fonti c'è un regolamento generale sulla tesi (di Ateneo).

    Criterio: `course_tag == "generale"` e `doc_type` tra {tesi, regolamento}.
    Funzione pura, testabile offline.
    """
    return any(
        getattr(s, "course_tag", None) == "generale"
        and getattr(s, "doc_type", None) in _GENERAL_TESI_DOC_TYPES
        for s in (sources or [])
    )


class UniLawResponder:
    def __init__(
        self,
        vector_db,
        use_bm25: bool = True,
        use_neural_reranker: bool | None = None,
        use_evidence: bool | None = None,
        use_deterministic: bool | None = None,
        use_prose_templates: bool | None = None,
        use_semantic_intent: bool | None = None,
        use_semantic_grounding: bool | None = None,
        use_semantic_abstention: bool | None = None,
        use_general_tesi_hint: bool | None = None,
    ):
        self.vector_db = vector_db
        self.use_bm25 = use_bm25

        # Mitigazione q14 (FASE 14, default ON): su una domanda di consultabilità
        # della tesi con un regolamento GENERALE fra le fonti, il profilo di risposta
        # autorizza l'uso della regola generale (riduce la falsa astensione). Toggle
        # per l'A/B; con False il profilo torna byte-identico a prima della FASE 14.
        self.use_general_tesi_hint = (
            GENERAL_TESI_HINT_ENABLED if use_general_tesi_hint is None else use_general_tesi_hint
        )

        # Regole deterministiche. `use_deterministic` è il master (se False: RAG puro,
        # nessun template). `use_prose_templates` abilita i 5 template di prosa
        # (default OFF: ridondanti col RAG); il guard numerico TOLC resta attivo col
        # solo master, perché l'esattezza delle soglie è critica.
        self.use_deterministic = (
            DETERMINISTIC_RULES_ENABLED if use_deterministic is None else use_deterministic
        )
        self.use_prose_templates = (
            PROSE_TEMPLATES_ENABLED if use_prose_templates is None else use_prose_templates
        )

        # Evidence selection (FASE 5): default da config, sovrascrivibile per l'A/B.
        self.use_evidence = (
            EVIDENCE_SELECTION_ENABLED if use_evidence is None else use_evidence
        )

        # Reranker neurale opzionale (default da config, sovrascrivibile da UI/eval).
        # L'oggetto è leggero: il modello viene caricato solo al primo uso effettivo.
        self.use_neural_reranker = (
            RERANKER_ENABLED if use_neural_reranker is None else use_neural_reranker
        )
        self.neural_reranker = CrossEncoderReranker(RERANKER_MODEL_NAME)

        # Intent detection semantica opzionale (FASE 11, default OFF). Quando è attiva
        # affianca le keyword; riusa il modello di embedding già caricato nel vector
        # store (nessun nuovo download/caricamento). Se disattivata, il classificatore
        # non viene costruito: il riconoscimento resta byte-identico a quello a keyword.
        self.use_semantic_intent = (
            SEMANTIC_INTENT_ENABLED if use_semantic_intent is None else use_semantic_intent
        )
        self.semantic_intent = (
            self._build_semantic_intent_classifier()
            if self.use_semantic_intent
            else None
        )

        # Grounding semantico delle citazioni (FASE 12, default OFF). Quando è attivo
        # le frasi citanti bocciate dal solo lessicale ricevono un secondo controllo
        # per similarità di embedding (rete di recupero per le parafrasi). Riusa lo
        # stesso embedder del vector store; se disattivato l'embedder non viene
        # nemmeno recuperato e `grounding_report` resta byte-identico al solo lessicale.
        self.use_semantic_grounding = (
            CITATION_GROUNDING_SEMANTIC_ENABLED
            if use_semantic_grounding is None
            else use_semantic_grounding
        )
        self.semantic_grounding_embedder = (
            self._embedder_from_vector_db() if self.use_semantic_grounding else None
        )

        # Retrieval strength semantica per l'astensione (FASE 13, default OFF). Quando è
        # attiva la distinzione fuori_dominio vs evidenza_insufficiente usa la similarità
        # di embedding query↔fonte (e una soglia ricalibrata) invece dell'overlap di token,
        # più robusta verso le parafrasi. Riusa lo stesso embedder del vector store; se
        # disattivata l'embedder non viene recuperato e la classificazione resta
        # byte-identica a quella lessicale.
        self.use_semantic_abstention = (
            ABSTENTION_SEMANTIC_STRENGTH_ENABLED
            if use_semantic_abstention is None
            else use_semantic_abstention
        )
        self.semantic_abstention_embedder = (
            self._embedder_from_vector_db() if self.use_semantic_abstention else None
        )

        # Indice lessicale BM25 costruito una volta dai chunk già in ChromaDB
        # (None se il vector store è assente/vuoto: i test e i casi senza indice
        # restano validi).
        self.bm25_index = build_bm25_index(vector_db)

        self.llm = ChatOllama(
            model=DEFAULT_MODEL_NAME,
            temperature=DEFAULT_TEMPERATURE,
            num_ctx=DEFAULT_NUM_CTX,
        )

        self.last_trace = RagTrace()

    def answer(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        memory: dict[str, Any] | None = None,
        show_interpretation: bool = True,
        show_confidence: bool = True,
    ) -> str:
        question = (question or "").strip()
        self.last_trace = RagTrace(question=question)

        if not question:
            return "Inserisci una domanda."

        calcolo = prova_calcolo_sicuro(question)

        if calcolo is not None:
            self.last_trace.topic = "calcolo"
            self.last_trace.confidence = "alta"
            self.last_trace.confidence_reason = (
                "Richiesta aritmetica riconosciuta e gestita dal modulo di calcolo sicuro."
            )
            return self._format_calculation_answer(
                calcolo,
                show_interpretation,
                show_confidence,
            )

        intent = self._infer_query_intent(question, memory or {})
        self.last_trace.course_tag = intent.course_tag
        self.last_trace.topic = intent.topic
        self.last_trace.used_memory = intent.used_memory

        if intent.detected_unknown_course:
            self.last_trace.confidence = "bassa"
            self.last_trace.abstention_reason = OUT_OF_DOMAIN_COURSE
            self.last_trace.confidence_reason = (
                "La domanda cita un corso non presente tra quelli riconosciuti dal corpus."
            )
            return self._format_unknown_course_answer(
                intent.detected_unknown_course,
                show_interpretation,
                show_confidence,
            )

        if intent.is_ambiguous:
            self.last_trace.confidence = "bassa"
            self.last_trace.abstention_reason = AMBIGUOUS
            self.last_trace.confidence_reason = (
                "Domanda ambigua: manca il corso di laurea necessario per selezionare fonti affidabili."
            )
            return self._format_clarification_answer(
                show_interpretation,
                show_confidence,
            )

        docs = self._retrieve_documents(question, intent)

        if not docs:
            self.last_trace.confidence = "bassa"
            self.last_trace.abstention_reason = WEAK_RETRIEVAL
            self.last_trace.confidence_reason = "Nessun documento pertinente recuperato."
            return self._format_no_evidence_answer(
                show_interpretation,
                show_confidence,
            )

        sources = self._prepare_sources(docs)

        if not sources:
            self.last_trace.confidence = "bassa"
            self.last_trace.abstention_reason = WEAK_RETRIEVAL
            self.last_trace.confidence_reason = (
                "Documenti recuperati ma nessuna fonte utilizzabile dopo il filtraggio."
            )
            return self._format_no_evidence_answer(
                show_interpretation,
                show_confidence,
            )

        self.last_trace.selected_sources = [s.citation_label for s in sources]

        confidence, reason = self._estimate_confidence(intent, sources)
        self.last_trace.confidence = confidence
        self.last_trace.confidence_reason = reason

        if self.use_deterministic:
            kwargs = dict(
                question=question,
                intent=intent,
                sources=sources,
                show_interpretation=show_interpretation,
                show_confidence=show_confidence,
            )

            # Guard numerico TOLC-I: sempre attivo (esattezza del verdetto sulle
            # soglie 9/16 e citazione della fonte canonica di accesso).
            deterministic_answer = self._try_deterministic_accesso_informatica_answer(**kwargs)
            if deterministic_answer is not None:
                return deterministic_answer

            # Template di prosa: opzionali, default OFF (ridondanti col RAG puro,
            # cfr. ESP-07). Short-circuitano sul primo applicabile.
            if self.use_prose_templates:
                deterministic_answer = (
                    self._try_deterministic_ofa_informatica_answer(**kwargs)
                    or self._try_deterministic_tesi_informatica_answer(**kwargs)
                    or self._try_deterministic_erasmus_answer(**kwargs)
                    or self._try_deterministic_borsa_answer(**kwargs)
                    or self._try_deterministic_accesso_scienze_educazione_answer(**kwargs)
                )
                if deterministic_answer is not None:
                    return deterministic_answer

        context = self._build_context(sources, question)
        answer_profile = self._build_answer_profile(intent, question, sources)
        self.last_trace.answer_profile = answer_profile

        prompt = QA_PROMPT.format(
            context=context,
            question=question,
            style_guide=ANSWER_STYLE_GUIDE,
            answer_profile=answer_profile,
        )

        try:
            llm_response = self.llm.invoke(prompt)
            raw_answer = getattr(llm_response, "content", str(llm_response)).strip()

        except Exception as exc:
            return self._format_ollama_error(exc)

        final_answer = self._postprocess_answer(raw_answer)

        # Ciclo 2 — FASE 2 — normalizza le citazioni al formato canonico [F#]
        # (il modello a volte scrive (F1) o F1): si esegue PRIMA della verifica e
        # del blocco fonti, così le citazioni vengono riconosciute e validate.
        final_answer = normalize_citations(final_answer, sources)

        # FASE 5 — verifica delle citazioni: rimuove i riferimenti [F#] inventati e
        # valuta il supporto delle frasi che citano. Politica "reduce": sotto soglia
        # si abbassa la confidenza e si aggiunge una nota, senza bloccare la risposta.
        final_answer = strip_invalid_citations(final_answer, sources)
        caveat = ""

        if CITATION_GROUNDING_ENABLED:
            # FASE 12: con `semantic_grounding_embedder` non-None, le frasi parafrasate
            # bocciate dal lessicale vengono recuperate per similarità di embedding;
            # con None (default) il risultato è quello del solo lessicale.
            ratio, unsupported = grounding_report(
                final_answer,
                sources,
                embedder=self.semantic_grounding_embedder,
                min_semantic=CITATION_GROUNDING_SEMANTIC_MIN_SIMILARITY,
            )

            if ratio is None:
                self.last_trace.grounding = "n.d. (nessuna citazione verificabile)"
            else:
                self.last_trace.grounding = f"{ratio:.0%} frasi citanti supportate"

                if ratio < CITATION_GROUNDING_MIN_RATIO:
                    if self.last_trace.confidence == "alta":
                        self.last_trace.confidence = "media"
                    elif self.last_trace.confidence == "media":
                        self.last_trace.confidence = "bassa"
                    self.last_trace.confidence_reason += (
                        " Alcune affermazioni citate hanno riscontro debole nelle fonti."
                    )
                    caveat = (
                        "\n\n> Nota: alcune affermazioni potrebbero non essere "
                        "pienamente supportate dalle fonti citate; verifica i riferimenti."
                    )

        # FASE 6 — astensione affidabile: se il modello si è astenuto, classifica la
        # causa (fuori dominio vs fonte presente ma insufficiente) e la rende esplicita.
        abstaining = is_abstention(final_answer)
        abstention_note = ""
        if abstaining:
            # FASE 13: con `semantic_abstention_embedder` non-None la causa
            # (fuori dominio vs evidenza insufficiente) è decisa sulla forza
            # SEMANTICA query↔fonte (soglia ricalibrata); con None (default) sulla
            # forza lessicale (soglia 0,37), byte-identica al comportamento storico.
            reason = classify_llm_abstention(
                question,
                sources,
                ABSTENTION_OOD_MAX_STRENGTH,
                embedder=self.semantic_abstention_embedder,
                semantic_ood_max_strength=ABSTENTION_OOD_SEMANTIC_MAX_STRENGTH,
            )
            self.last_trace.abstention_reason = reason
            abstention_note = format_reason(reason)

        prefix = ""

        if show_interpretation:
            prefix += self._format_interpretation_block(intent)

        if show_confidence:
            prefix += self._format_confidence_block()

        final_answer = prefix + final_answer
        # Ciclo 2 — FASE 3: in astensione il blocco fonti non rivendica fonti
        # "utilizzate" (sarebbe una falsa attribuzione su domande come q19).
        final_answer += self._format_sources_block(final_answer, sources, abstaining=abstaining)
        final_answer += caveat
        final_answer += abstention_note

        return final_answer

    def update_memory_from_trace(self, memory: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Aggiorna una memoria a slot, evitando di salvare tutta la conversazione.
        """
        memory = dict(memory or {})

        if self.last_trace.course_tag:
            memory["last_course_tag"] = self.last_trace.course_tag

        if self.last_trace.topic:
            memory["last_topic"] = self.last_trace.topic

        return memory

    def _format_calculation_answer(
        self,
        result: str,
        show_interpretation: bool,
        show_confidence: bool,
    ) -> str:
        blocks = []

        if show_interpretation:
            blocks.append(
                "### Interpretazione della richiesta\n"
                "- Tipo: calcolo numerico\n"
                "- Metodo: modulo di calcolo sicuro\n"
            )

        if show_confidence:
            blocks.append(
                "### Affidabilità\n"
                "- Livello: alta\n"
                "- Motivo: il risultato è prodotto da una funzione aritmetica controllata, "
                "non dal modello linguistico.\n"
            )

        blocks.append(f"Risultato: **{result}**")
        return "\n".join(blocks)

    def _format_ollama_error(self, exc: Exception) -> str:
        text = str(exc)

        if "Connection" in text or "Failed to establish" in text or "Max retries" in text:
            return (
                "Errore: Ollama non è raggiungibile su localhost:11434.\n\n"
                "Avvia Ollama con:\n\n"
                "```powershell\n"
                "ollama serve\n"
                "```\n\n"
                "Poi, in un altro terminale, riavvia l'app Streamlit."
            )

        if "404" in text or "model is not found" in text or "not found" in text.lower():
            return (
                f"Errore: il modello Ollama '{DEFAULT_MODEL_NAME}' non è stato trovato.\n\n"
                "Scaricalo con:\n\n"
                "```powershell\n"
                f"ollama pull {DEFAULT_MODEL_NAME}\n"
                "```\n\n"
                "Poi verifica con:\n\n"
                "```powershell\n"
                "ollama list\n"
                "```"
            )

        return f"Errore durante la chiamata a Ollama: {exc}"

    def _build_semantic_intent_classifier(self) -> Optional[SemanticIntentClassifier]:
        """Costruisce il classificatore semantico riusando l'embedder del vector store.

        Restituisce `None` se non è possibile recuperare un embedder utilizzabile
        (es. vector store assente nei test): in tal caso il riconoscimento ricade
        sulle sole keyword, senza errori.
        """
        embedder = self._embedder_from_vector_db()
        if embedder is None:
            return None
        return SemanticIntentClassifier(
            embedder=embedder,
            course_min_similarity=SEMANTIC_INTENT_COURSE_MIN_SIMILARITY,
            topic_min_similarity=SEMANTIC_INTENT_TOPIC_MIN_SIMILARITY,
        )

    def _embedder_from_vector_db(self):
        """Estrae dal vector store la funzione di embedding già caricata (o `None`)."""
        embedding_function = getattr(self.vector_db, "_embedding_function", None) or getattr(
            self.vector_db, "embeddings", None
        )
        if embedding_function is None:
            return None
        embed_documents = getattr(embedding_function, "embed_documents", None)
        return embed_documents if callable(embed_documents) else None

    def _infer_query_intent(self, question: str, memory: dict[str, Any]) -> QueryIntent:
        return infer_query_intent(question, memory, self.semantic_intent)

    def _asks_tesi_consultazione(self, question: str) -> bool:
        return asks_tesi_consultazione(question)

    def _format_unknown_course_answer(
        self,
        unknown_course: str,
        show_interpretation: bool,
        show_confidence: bool,
    ) -> str:
        blocks = []

        if show_interpretation:
            blocks.append(
                "### Interpretazione della richiesta\n"
                f"- Corso citato: {unknown_course}\n"
                "- Esito: corso non riconosciuto tra quelli disponibili nel corpus indicizzato\n"
            )

        if show_confidence:
            blocks.append(
                "### Affidabilità\n"
                "- Livello: bassa\n"
                "- Motivo: il corso indicato non risulta tra quelli riconosciuti dal sistema.\n"
            )

        blocks.append(
            "Non posso rispondere in modo affidabile su questo corso usando i documenti attualmente indicizzati.\n\n"
            "I corsi riconosciuti dal sistema sono:\n"
            "- Informatica L-31\n"
            "- Scienze dell'educazione L-19\n"
            "- Scienze dell'amministrazione L-16\n\n"
            "Se hai aggiunto documenti relativi a questo corso nella cartella `documenti/`, "
            "ricostruisci la knowledge base dalla sidebar."
        )

        return "\n".join(blocks)

    def _format_clarification_answer(
        self,
        show_interpretation: bool,
        show_confidence: bool,
    ) -> str:
        blocks = []

        if show_interpretation:
            topic_label = TOPIC_LABELS.get(
                self.last_trace.topic or "",
                "Argomento non determinato",
            )
            blocks.append(
                "### Interpretazione della richiesta\n"
                f"- Argomento rilevato: {topic_label}\n"
                "- Corso: non specificato\n"
            )

        if show_confidence:
            blocks.append(
                "### Affidabilità\n"
                "- Livello: bassa\n"
                "- Motivo: manca il corso di laurea necessario per selezionare il regolamento corretto.\n"
            )

        blocks.append(
            "La domanda è ambigua: indica il corso di laurea a cui ti riferisci.\n\n"
            "Posso rispondere, ad esempio, per:\n"
            "- Informatica L-31\n"
            "- Scienze dell'educazione L-19\n"
            "- Scienze dell'amministrazione L-16"
        )

        return "\n".join(blocks)

    def _format_no_evidence_answer(
        self,
        show_interpretation: bool,
        show_confidence: bool,
    ) -> str:
        blocks = []

        if show_interpretation:
            blocks.append(self._format_interpretation_block_from_trace())

        if show_confidence:
            blocks.append(self._format_confidence_block())

        blocks.append("Non lo so in base ai documenti disponibili.")
        return "\n".join(blocks)

    def _retrieve_documents(self, question: str, intent: QueryIntent):
        # FASE 3: candidati ibridi (vettoriale + BM25, fusi con RRF), poi
        # ordinamento euristico e filtro metadata per corso. Con use_bm25=False
        # la pipeline riproduce esattamente il comportamento pre-FASE 3.
        docs = hybrid_retrieve(
            self.vector_db,
            self.bm25_index,
            question,
            intent,
            self.last_trace,
            use_bm25=self.use_bm25,
        )
        docs = rerank_documents(question, docs, intent, self.last_trace)
        docs = filter_documents_by_course(docs, intent)

        # FASE 4: reranking neurale opzionale sui top-N candidati già filtrati.
        # Se disattivato o non disponibile, resta l'ordinamento euristico.
        if self.use_neural_reranker and self.neural_reranker.available():
            docs = self.neural_reranker.rerank(question, docs, RERANKER_TOP_N)
            self.last_trace.reranker = "euristico + cross-encoder"
        elif self.use_neural_reranker:
            self.last_trace.reranker = "euristico (reranker neurale non disponibile)"
        else:
            self.last_trace.reranker = "euristico"

        return docs[:MAX_CONTEXT_DOCUMENTS]

    def _prepare_sources(self, docs: list) -> List[RetrievedSource]:
        prepared = []
        seen_pages = set()
        seen_content = set()

        for doc in docs:
            metadata = doc.metadata or {}

            filename = metadata.get("filename") or os.path.basename(
                metadata.get("source", "Documento sconosciuto")
            )

            page = metadata.get("page", None)
            course_tag = metadata.get("course_tag", "generale")
            doc_type = metadata.get("doc_type", "altro")
            content = " ".join(doc.page_content.strip().split())

            page_key = (filename, page)
            content_key = (filename, content[:250])

            if page_key in seen_pages:
                continue

            if content_key in seen_content:
                continue

            seen_pages.add(page_key)
            seen_content.add(content_key)

            prepared.append(
                RetrievedSource(
                    index=len(prepared) + 1,
                    filename=filename,
                    page=page,
                    content=content,
                    course_tag=course_tag,
                    doc_type=doc_type,
                )
            )

            if len(prepared) >= MAX_CONTEXT_DOCUMENTS:
                break

        return prepared

    def _build_context(self, sources: List[RetrievedSource], question: str) -> str:
        blocks = []
        chars_before = 0
        chars_after = 0

        for source in sources:
            chars_before += len(source.content)

            if self.use_evidence:
                evidence = select_passage(
                    question,
                    source.content,
                    max_sentences=EVIDENCE_MAX_SENTENCES,
                    min_sentences=EVIDENCE_MIN_SENTENCES,
                    max_chars=EVIDENCE_MAX_CHARS,
                )
            else:
                evidence = source.content

            chars_after += len(evidence)

            blocks.append(
                f"{source.citation_label}\n"
                f"Tipo documento: {source.doc_type}\n"
                f"Corso: {source.course_tag}\n"
                f"Contenuto:\n{evidence}"
            )

        self.last_trace.evidence_chars = (
            f"{chars_before} → {chars_after} caratteri"
            + ("" if self.use_evidence else " (selezione disattivata)")
        )

        return "\n\n".join(blocks)

    def _build_answer_profile(
        self,
        intent: QueryIntent,
        question: str,
        sources: Optional[List[RetrievedSource]] = None,
    ) -> str:
        topic = intent.topic

        if topic == "accesso":
            if intent.course_tag == "scienze_educazione":
                return (
                    "Profilo accesso/ammissione Scienze dell'Educazione L-19:\n"
                    "- Rispondi sulla prova di ammissione o sull'immatricolazione, non sulla prova finale di laurea.\n"
                    "- Se il contesto contiene durata, numero di quesiti, materie o criteri di valutazione, riportali chiaramente.\n"
                    "- Se possibile, organizza la risposta in: Risposta breve, Struttura della prova, Valutazione, Cosa fare.\n"
                    "- Non usare fonti di Informatica L-31 o di altri corsi.\n"
                )

            return (
                "Profilo accesso/TOLC/OFA:\n"
                "- Inizia con una risposta breve.\n"
                "- Se ci sono soglie o fasce di punteggio, usa una tabella Markdown.\n"
                "- Spiega la conseguenza pratica per l'immatricolazione.\n"
                "- Aggiungi una sezione 'Cosa fare' se utile.\n"
            )

        if topic == "tesi":
            if self._asks_tesi_consultazione(question):
                profile = (
                    "Profilo consultabilità/deposito tesi:\n"
                    "- Rispondi alla domanda sulla consultabilità della tesi dopo la laurea.\n"
                    "- Distingui tra tesi consultabile, consultabile dopo embargo e non consultabile, se presente nel contesto.\n"
                    "- Non parlare della procedura di prenotazione della seduta se non richiesta.\n"
                    "- Cita il regolamento sulla tesi o consultabilità se recuperato.\n"
                )

                # Ciclo 2 — FASE 14: se è presente un regolamento GENERALE sulla
                # tesi, autorizza esplicitamente l'uso della regola generale. Le
                # regole di consultabilità/deposito/embargo valgono per l'intero
                # Ateneo: senza questo hint il modello 8B tende ad astenersi perché
                # la fonte non nomina il corso indicato nella domanda (caso q14).
                if self.use_general_tesi_hint and has_general_tesi_regulation(sources):
                    profile += (
                        "- IMPORTANTE: le regole di consultabilità, deposito ed embargo della "
                        "tesi sono stabilite da un regolamento generale di Ateneo, unico e valido "
                        "per TUTTI i corsi, compreso quello indicato nella domanda. Il fatto che "
                        "il regolamento non nomini il singolo corso NON significa che manchi "
                        "l'informazione: la regola generale è quella applicabile. Rispondi quindi "
                        "in base al contesto e non dire \"Non lo so\" se il contesto riporta le "
                        "condizioni di consultabilità (consultabile, consultabile dopo embargo, "
                        "non consultabile).\n"
                    )

                return profile

            return (
                "Profilo prova finale/tesi:\n"
                "- Distingui requisiti, modalità e procedura pratica.\n"
                "- Se le fonti includono una guida online, separa gli adempimenti amministrativi dalle regole accademiche.\n"
                "- Non concentrarti su consultabilità, embargo o deposito se l'utente non lo chiede esplicitamente.\n"
                "- Usa punti elenco chiari.\n"
            )

        if topic == "erasmus":
            return (
                "Profilo Erasmus/mobilità:\n"
                "- Organizza la risposta per aree: domanda, graduatorie, documenti, contributi, adempimenti finali.\n"
                "- Non usare fonti non Erasmus.\n"
                "- Usa punti elenco pratici.\n"
            )

        if topic == "piano_studi":
            return (
                "Profilo piano di studi:\n"
                "- Spiega regole generali, corsi coerenti, CFU e fonti regolamentari.\n"
                "- Se sono presenti corsi a scelta o attività formative, elencali chiaramente.\n"
            )

        if topic == "borsa":
            return (
                "Profilo borsa di studio:\n"
                "- Evidenzia requisiti, soglie, importi, scadenze e documenti richiesti, se presenti.\n"
                "- Usa tabelle per soglie ISEE/ISPE o importi quando il contesto lo consente.\n"
            )

        return (
            "Profilo generale:\n"
            "- Rispondi in modo sintetico e verificabile.\n"
            "- Usa le fonti più pertinenti.\n"
        )

    def _estimate_confidence(self, intent: QueryIntent, sources: List[RetrievedSource]) -> tuple[str, str]:
        return estimate_confidence(intent, sources)

    def _try_deterministic_ofa_informatica_answer(
        self,
        question: str,
        intent: QueryIntent,
        sources: List[RetrievedSource],
        show_interpretation: bool,
        show_confidence: bool,
    ) -> Optional[str]:
        """
        Risposta controllata e prudente per domande operative sugli OFA
        di Informatica L-31. Evita formulazioni troppo nette quando le fonti
        indicano soglie e rinvii regolamentari ma non descrivono una procedura
        amministrativa completa.
        """
        if not (intent.course_tag == "informatica" and intent.topic == "accesso"):
            return None

        q = question.lower()
        asks_ofa_procedure = (
            "ofa" in q
            and any(
                k in q
                for k in [
                    "assolver",
                    "assolvo",
                    "assolvimento",
                    "recuper",
                    "come posso",
                    "cosa devo fare",
                    "procedura",
                    "modalità",
                    "modalita",
                ]
            )
        )

        if not asks_ofa_procedure:
            return None

        context_text = " ".join(s.content.lower() for s in sources)

        has_ofa_context = (
            "ofa" in context_text
            or "obblighi formativi aggiuntivi" in context_text
            or "ris_test" in context_text
            or "ris test" in context_text
        )

        if not has_ofa_context:
            return None

        self.last_trace.deterministic_rule_used = "ofa_informatica_prudente"
        self.last_trace.answer_profile = (
            "Template prudenziale per assolvimento OFA Informatica L-31."
        )
        self.last_trace.confidence = "media"
        self.last_trace.confidence_reason = (
            "Le fonti recuperate collegano gli OFA alle soglie del TOLC-I/Ris_Test, "
            "ma non descrivono in modo completo una procedura amministrativa autonoma."
        )

        prefix = ""

        if show_interpretation:
            prefix += self._format_interpretation_block(intent)

        if show_confidence:
            prefix += self._format_confidence_block()

        main_source_label = self._best_access_source_label(sources)

        body = (
            "Risposta breve: secondo le fonti recuperate, per Informatica L-31 gli OFA "
            "sono collegati al punteggio TOLC-I/Ris\\_Test: con un punteggio "
            "**inferiore a 16 e non inferiore a 9** l'immatricolazione è consentita "
            f"ma con OFA; con un punteggio **>= 16** si risulta senza OFA. {main_source_label}\n\n"
            "Dettaglio: le fonti indicano le fasce di accesso e rinviano alle modalità "
            "previste dal regolamento per l'assolvimento degli Obblighi Formativi Aggiuntivi. "
            "Non uso quindi una procedura inventata: se il documento non specifica tutti i passaggi "
            "operativi, la risposta deve restare prudente.\n\n"
            + tolc_bands_table("Esito", "OFA")
            + "Cosa fare:\n"
            "- verifica il tuo punteggio TOLC-I/Ris\\_Test;\n"
            "- se sei nella fascia **>= 9 e < 16**, considera che l'immatricolazione è con OFA;\n"
            "- consulta l'articolo del regolamento richiamato dalle fonti per le modalità operative "
            "di assolvimento;\n"
            "- evita di assumere procedure non riportate esplicitamente nei documenti recuperati.\n"
        )

        body += self._format_sources_block(body, sources)
        return prefix + body

    def _best_tesi_academic_source_label(self, sources: List[RetrievedSource]) -> str:
        for source in sources:
            filename = source.filename.lower()
            if "prova" in filename and "finale" in filename and "informatica" in filename:
                return source.citation_label

        for source in sources:
            filename = source.filename.lower()
            if "regolamento" in filename and "informatica" in filename:
                return source.citation_label

        return sources[0].citation_label if sources else ""

    def _best_tesi_guide_source_label(self, sources: List[RetrievedSource]) -> str:
        for source in sources:
            filename = source.filename.lower()
            if "guida" in filename and "tesi" in filename:
                return source.citation_label

        return ""

    def _try_deterministic_tesi_informatica_answer(
        self,
        question: str,
        intent: QueryIntent,
        sources: List[RetrievedSource],
        show_interpretation: bool,
        show_confidence: bool,
    ) -> Optional[str]:
        """
        Risposta più ordinata per domande generiche/ellittiche sulla tesi di
        Informatica L-31. Prima spiega la prova finale, poi la procedura online.
        """
        if not (intent.course_tag == "informatica" and intent.topic == "tesi"):
            return None

        if self._asks_tesi_consultazione(question):
            return None

        q = question.lower().strip()
        is_generic_tesi_question = (
            q in {"e per la tesi?", "per la tesi?", "tesi?"}
            or any(k in q for k in ["prova finale", "come funziona la tesi", "regole della tesi"])
        )

        if not (is_generic_tesi_question or intent.used_memory):
            return None

        context_text = " ".join(s.content.lower() for s in sources)
        has_tesi_context = any(
            k in context_text
            for k in ["prova finale", "elaborato", "relatore", "discussione", "domanda di laurea"]
        )

        if not has_tesi_context:
            return None

        self.last_trace.deterministic_rule_used = "tesi_informatica_strutturata"
        self.last_trace.answer_profile = (
            "Template strutturato per prova finale/tesi Informatica L-31."
        )
        self.last_trace.confidence = "alta"
        self.last_trace.confidence_reason = (
            "Le fonti selezionate includono regolamento della prova finale e/o guida operativa tesi."
        )

        prefix = ""

        if show_interpretation:
            prefix += self._format_interpretation_block(intent)

        if show_confidence:
            prefix += self._format_confidence_block()

        academic_source = self._best_tesi_academic_source_label(sources)
        guide_source = self._best_tesi_guide_source_label(sources)

        guide_sentence = (
            f" Per gli adempimenti online è utile anche la guida operativa. {guide_source}"
            if guide_source
            else ""
        )

        body = (
            "Risposta breve: per Informatica L-31 la tesi/prova finale consiste nella "
            "preparazione, stesura e discussione di un elaborato concordato con un docente "
            f"relatore. {academic_source}\n\n"
            "Dettaglio accademico:\n"
            "- l'elaborato riguarda un argomento concordato con il relatore;\n"
            "- il lavoro deve essere sviluppato dallo studente con autonomia;\n"
            "- può includere contributi teorici, metodologici, progettuali o implementativi;\n"
            "- può anche essere collegato a un progetto svolto presso aziende o enti esterni, "
            "se previsto dalle fonti recuperate.\n\n"
            "Procedura amministrativa:\n"
            "- la guida tesi online disciplina il caricamento della domanda di laurea e degli allegati;\n"
            "- tra i documenti operativi possono comparire domanda di laurea, modulo di prenotazione "
            "della prova finale, documenti firmati e ricevuta del versamento previsto;\n"
            "- le scadenze operative vanno verificate nella guida e nella procedura Esse3/tesi online."
            f"{guide_sentence}\n\n"
            "Cosa fare:\n"
            "- scegli un argomento e concordalo con un docente relatore;\n"
            "- prepara l'elaborato secondo le indicazioni del corso;\n"
            "- verifica sulla guida online i documenti da caricare e le scadenze prima dell'appello di laurea.\n"
        )

        body += self._format_sources_block(body, sources)
        return prefix + body

    def _asks_borsa_graduatoria(self, question: str) -> bool:
        return asks_borsa_graduatoria(question)

    def _asks_erasmus_end_mobility(self, question: str) -> bool:
        return asks_erasmus_end_mobility(question)

    def _best_erasmus_source_label(self, sources: List[RetrievedSource]) -> str:
        for source in sources:
            if "erasmus" in source.filename.lower():
                return source.citation_label
        return sources[0].citation_label if sources else ""

    def _try_deterministic_erasmus_answer(
        self,
        question: str,
        intent: QueryIntent,
        sources: List[RetrievedSource],
        show_interpretation: bool,
        show_confidence: bool,
    ) -> Optional[str]:
        """
        Risposta mirata per domande sugli adempimenti di fine mobilità Erasmus.
        Evita di mischiare la candidatura iniziale con i documenti da consegnare al rientro.
        """
        if intent.topic != "erasmus":
            return None

        if not self._asks_erasmus_end_mobility(question):
            return None

        erasmus_sources = [
            source
            for source in sources
            if source.doc_type == "erasmus" or "erasmus" in source.filename.lower()
        ]

        if not erasmus_sources:
            return None

        context_text = " ".join(source.content.lower() for source in erasmus_sources)
        has_end_mobility_context = any(
            k in context_text
            for k in [
                "attestato di permanenza",
                "learning agreement",
                "breve relazione",
                "giustificativi",
                "riconoscimento attività",
                "riconoscimento attivita",
            ]
        )

        if not has_end_mobility_context:
            return None

        self.last_trace.deterministic_rule_used = "erasmus_fine_mobilita_documenti"
        self.last_trace.answer_profile = (
            "Template mirato per documenti da consegnare al termine della mobilità Erasmus."
        )
        self.last_trace.confidence = "alta"
        self.last_trace.confidence_reason = (
            "Le fonti Erasmus recuperate contengono gli adempimenti finali della mobilità."
        )

        prefix = ""

        if show_interpretation:
            prefix += self._format_interpretation_block(intent)

        if show_confidence:
            prefix += self._format_confidence_block()

        main_source = self._best_erasmus_source_label(erasmus_sources)

        body = (
            f"Risposta breve: al termine della mobilità Erasmus devi consegnare la documentazione finale prevista dal bando. {main_source}\n\n"
            "Documenti finali indicati dalle fonti recuperate:\n"
            "- attestato di permanenza, indicato come Allegato C, con date di arrivo e partenza;\n"
            "- Learning Agreement nella versione definitiva, completo nelle sezioni richieste e con firme/approvazioni previste;\n"
            "- breve relazione sull'esperienza di mobilità, se richiesta dal bando;\n"
            "- giustificativi di viaggio nominativi, in particolare nei casi previsti per il viaggio green;\n"
            "- modulo di riconoscimento dell'attività didattica svolta in mobilità, con esami da convalidare e attività sostenute.\n\n"
            "Cosa fare:\n"
            "- prepara i documenti finali prima del rientro o subito dopo la conclusione della mobilità;\n"
            "- verifica che Learning Agreement e attestato di permanenza siano completi e firmati/approvati;\n"
            "- conserva eventuali giustificativi di viaggio;\n"
            "- non confondere questi adempimenti finali con la domanda di candidatura iniziale.\n"
        )

        body += self._format_sources_block(body, erasmus_sources)
        return prefix + body

    def _try_deterministic_borsa_graduatoria_answer(
        self,
        question: str,
        intent: QueryIntent,
        sources: List[RetrievedSource],
        show_interpretation: bool,
        show_confidence: bool,
    ) -> Optional[str]:
        """Risposta mirata per domande sulla graduatoria della borsa di studio."""
        if intent.topic != "borsa":
            return None

        if not self._asks_borsa_graduatoria(question):
            return None

        borsa_sources = [
            source
            for source in sources
            if source.doc_type == "borsa" or "borsa" in source.filename.lower()
        ]

        if not borsa_sources:
            return None

        context_text = " ".join(source.content.lower() for source in borsa_sources)
        has_graduatoria_context = "graduatoria" in context_text or "graduatorie" in context_text

        if not has_graduatoria_context:
            return None

        self.last_trace.deterministic_rule_used = "borsa_graduatoria_strutturata"
        self.last_trace.answer_profile = "Template mirato per graduatorie della borsa di studio."
        self.last_trace.confidence = "alta"
        self.last_trace.confidence_reason = (
            "Le fonti recuperate sono pertinenti al bando borsa di studio e contengono riferimenti alle graduatorie."
        )

        prefix = ""

        if show_interpretation:
            prefix += self._format_interpretation_block(intent)

        if show_confidence:
            prefix += self._format_confidence_block()

        main_source = borsa_sources[0].citation_label

        body = (
            f"Risposta breve: il bando disciplina le graduatorie della borsa di studio e la posizione dello studente rispetto all'idoneità o al beneficio. {main_source}\n\n"
            "In base alle fonti recuperate, quando si parla di graduatoria occorre controllare questi aspetti:\n\n"
            "| Aspetto | Significato pratico |\n"
            "|---|---|\n"
            "| Graduatoria provvisoria | Prima pubblicazione/esito iniziale, da controllare per verificare posizione ed eventuali anomalie |\n"
            "| Graduatoria definitiva | Esito consolidato dopo le verifiche previste dal bando |\n"
            "| Graduatoria assestata | Aggiornamento successivo, se previsto, legato a scorrimenti o assestamenti delle posizioni |\n"
            "| Idoneità | Indica che lo studente possiede i requisiti previsti, ma va distinta dall'effettiva assegnazione del beneficio quando le risorse sono limitate |\n"
            "| Posizione in graduatoria | Serve a capire priorità, eventuale beneficio e possibili scorrimenti |\n\n"
            "Cosa fare:\n"
            "- controlla la graduatoria pubblicata e la tua posizione;\n"
            "- verifica se risulti idoneo, beneficiario o in attesa di eventuale scorrimento;\n"
            "- consulta le versioni successive della graduatoria se il bando prevede aggiornamenti;\n"
            "- non dedurre importi o assegnazioni se non sono esplicitamente riportati nella fonte.\n"
        )

        body += self._format_sources_block(body, borsa_sources)
        return prefix + body

    def _try_deterministic_borsa_answer(
        self,
        question: str,
        intent: QueryIntent,
        sources: List[RetrievedSource],
        show_interpretation: bool,
        show_confidence: bool,
    ) -> Optional[str]:
        """
        Risposta strutturata per domande generali sui requisiti della borsa di studio.
        Non inventa soglie/importi: organizza solo le categorie che tipicamente emergono
        dal bando e rimanda alle fonti per valori puntuali.
        """
        if intent.topic != "borsa":
            return None

        grad_answer = self._try_deterministic_borsa_graduatoria_answer(
            question=question,
            intent=intent,
            sources=sources,
            show_interpretation=show_interpretation,
            show_confidence=show_confidence,
        )

        if grad_answer is not None:
            return grad_answer

        q = question.lower()
        asks_requirements = any(
            k in q
            for k in ["requisiti", "cosa serve", "come funziona", "principali"]
        )

        if not asks_requirements:
            return None

        borsa_sources = [
            source
            for source in sources
            if source.doc_type == "borsa" or "borsa" in source.filename.lower()
        ]

        if not borsa_sources:
            return None

        self.last_trace.deterministic_rule_used = "borsa_requisiti_strutturata"
        self.last_trace.answer_profile = (
            "Template strutturato per requisiti principali della borsa di studio."
        )
        self.last_trace.confidence = "media"
        self.last_trace.confidence_reason = (
            "Le fonti recuperate sono pertinenti al bando borsa di studio; la risposta organizza "
            "le categorie principali senza inventare soglie o importi non verificati."
        )

        prefix = ""

        if show_interpretation:
            prefix += self._format_interpretation_block(intent)

        if show_confidence:
            prefix += self._format_confidence_block()

        main_source = borsa_sources[0].citation_label

        body = (
            f"Risposta breve: i requisiti principali per la borsa di studio vanno verificati nel bando. {main_source}\n\n"
            "In base alle fonti recuperate, la valutazione ruota soprattutto intorno a queste aree:\n\n"
            "| Area | Cosa controllare |\n"
            "|---|---|\n"
            "| Requisiti economici | Indicatori come ISEE/ISPE o altri parametri economici previsti dal bando |\n"
            "| Requisiti di iscrizione | Anno di corso, iscrizione e condizioni amministrative previste dal bando |\n"
            "| Requisiti di merito | Crediti, anno di iscrizione o condizioni di merito se previste per la categoria di studente |\n"
            "| Graduatorie | Idoneità, posizione in graduatoria provvisoria/definitiva/assestata |\n"
            "| Benefici e rimborsi | Eventuali rimborsi, importi o benefici indicati dal bando |\n\n"
            "Cosa fare:\n"
            "- consulta l'articolo del bando relativo ai requisiti economici e di merito;\n"
            "- verifica la tua posizione nelle graduatorie pubblicate;\n"
            "- controlla eventuali incompatibilità, scadenze e documentazione richiesta;\n"
            "- non assumere soglie o importi se non sono esplicitamente riportati nella fonte.\n"
        )

        body += self._format_sources_block(body, sources)
        return prefix + body

    def _try_deterministic_accesso_informatica_answer(
        self,
        question: str,
        intent: QueryIntent,
        sources: List[RetrievedSource],
        show_interpretation: bool,
        show_confidence: bool,
    ) -> Optional[str]:
        """
        Risposta controllata per TOLC-I / OFA / Accesso Informatica L-31.
        Serve a evitare errori del modello su soglie numeriche e fasce normative.
        """
        if not (intent.course_tag == "informatica" and intent.topic == "accesso"):
            return None

        context_text = " ".join(s.content.lower() for s in sources)

        has_access_context = (
            ("tolc" in context_text or "ris_test" in context_text or "ris test" in context_text)
            and "ofa" in context_text
            and ("immatricol" in context_text or "accesso" in context_text)
        )

        if not has_access_context:
            return None

        q = question.lower()

        asks_general_access = any(
            k in q
            for k in [
                "come funziona",
                "possibili esiti",
                "in base al punteggio",
                "punteggio ottenuto",
                "casistiche",
                "fasce",
                "soglie",
                "valutazioni",
            ]
        )

        score = self._extract_tolc_score(question)

        if score is None and not asks_general_access:
            return None

        self.last_trace.deterministic_rule_used = "accesso_informatica_tolc"
        self.last_trace.answer_profile = (
            "Template deterministico per accesso Informatica L-31 / TOLC-I / OFA."
        )

        prefix = ""

        if show_interpretation:
            prefix += self._format_interpretation_block(intent)

        if show_confidence:
            prefix += self._format_confidence_block()

        main_source_label = self._best_access_source_label(sources)

        if score is not None:
            outcome = self._classify_tolc_score(score)

            body = (
                f"Risposta breve: **{outcome['short']}**\n\n"
                f"Dettaglio: il punteggio TOLC-I/Ris\\_Test pari a **{score:g}** "
                f"rientra nella fascia **{outcome['range']}**. "
                f"La conseguenza indicata è: **{outcome['consequence']}**. {main_source_label}\n\n"
                + tolc_bands_table("Esito per l'immatricolazione", "Conseguenza")
                + "Cosa fare:\n"
                f"- Considera il tuo punteggio: **{score:g}**.\n"
                f"- Segui la conseguenza della fascia: **{outcome['consequence']}**.\n"
                "- Verifica nel regolamento le modalità operative e le eventuali procedure per assolvere gli OFA.\n"
            )

        else:
            body = (
                "Risposta breve: l'accesso a Informatica L-31 dipende dal punteggio "
                "TOLC-I/Ris\\_Test. Le fonti distinguono tre fasce principali. "
                f"{main_source_label}\n\n"
                + tolc_bands_table("Esito per l'immatricolazione", "Conseguenza")
                + "Cosa fare:\n"
                "- Individua il tuo punteggio TOLC-I/Ris\\_Test.\n"
                "- Confrontalo con la tabella.\n"
                "- Se rientri nella fascia con OFA, verifica nel regolamento le modalità di assolvimento.\n"
            )

        body += self._format_sources_block(body, sources)
        return prefix + body

    def _extract_tolc_score(self, question: str) -> Optional[float]:
        return extract_tolc_score(question)

    def _classify_tolc_score(self, score: float) -> dict[str, str]:
        return classify_tolc_score(score)

    def _best_access_source_label(self, sources: List[RetrievedSource]) -> str:
        for source in sources:
            filename = source.filename.lower()

            if "regolamento-di-accesso" in filename and "informatica" in filename:
                return f"{source.citation_label}"

        if sources:
            return sources[0].citation_label

        return ""

    def _best_l19_access_source_label(self, sources: List[RetrievedSource]) -> str:
        for source in sources:
            filename = source.filename.lower()

            if "immatricolazione" in filename and (
                "educazione" in filename or "l-19" in filename or "l19" in filename
            ):
                return f"{source.citation_label}"

        for source in sources:
            filename = source.filename.lower()

            if "regolamento" in filename and (
                "educazione" in filename or "l-19" in filename or "l19" in filename
            ):
                return f"{source.citation_label}"

        if sources:
            return sources[0].citation_label

        return ""

    def _try_deterministic_accesso_scienze_educazione_answer(
        self,
        question: str,
        intent: QueryIntent,
        sources: List[RetrievedSource],
        show_interpretation: bool,
        show_confidence: bool,
    ) -> Optional[str]:
        """
        Risposta controllata per accesso / ammissione / immatricolazione
        a Scienze dell'Educazione L-19.

        Serve a evitare due errori ricorrenti:
        - confondere la prova di ammissione con la prova finale;
        - inventare un TOLC o una soglia minima quando le fonti recuperate
          descrivono invece una prova di ammissione interna.
        """
        if not (intent.course_tag == "scienze_educazione" and intent.topic == "accesso"):
            return None

        context_text = " ".join(s.content.lower() for s in sources)
        q = question.lower()

        has_l19_access_context = (
            (
                "scienze dell'educazione" in context_text
                or "scienze dell’educazione" in context_text
                or "l-19" in context_text
                or "l19" in context_text
            )
            and (
                "prova di ammissione" in context_text
                or "80 quesiti" in context_text
                or "risposta multipla" in context_text
                or "immatricolazione" in context_text
            )
        )

        if not has_l19_access_context:
            return None

        asks_tolc = "tolc" in q
        asks_admission_test = any(
            k in q
            for k in [
                "prova di ammissione",
                "test di ammissione",
                "com'è strutturata",
                "come è strutturata",
                "strutturata",
                "quesiti",
                "risposta multipla",
            ]
        )
        asks_immatricolazione = any(
            k in q
            for k in [
                "immatricol",
                "iscrizione",
                "iscrivermi",
                "che cosa mi serve",
                "cosa mi serve",
                "requisiti",
                "ammissione al corso",
            ]
        )

        if not (asks_tolc or asks_admission_test or asks_immatricolazione):
            return None

        self.last_trace.deterministic_rule_used = "accesso_scienze_educazione_l19"
        self.last_trace.answer_profile = (
            "Template deterministico per accesso / ammissione / immatricolazione "
            "a Scienze dell'Educazione L-19."
        )

        main_source_label = self._best_l19_access_source_label(sources)

        if asks_tolc:
            self.last_trace.confidence = "media"
            self.last_trace.confidence_reason = (
                "Le fonti pertinenti su Scienze dell'Educazione L-19 descrivono la prova di ammissione, "
                "ma non riportano un TOLC specifico o una soglia minima TOLC."
            )
        else:
            self.last_trace.confidence = "alta"
            self.last_trace.confidence_reason = (
                "Le fonti selezionate sono specifiche per Scienze dell'Educazione L-19 e descrivono "
                "accesso, prova di ammissione e criteri di valutazione."
            )

        prefix = ""

        if show_interpretation:
            prefix += self._format_interpretation_block(intent)

        if show_confidence:
            prefix += self._format_confidence_block()

        # FASE 7: tabella generata dal layer di conoscenza (dati con provenienza),
        # output identico alla versione prima codificata qui.
        test_table = l19_test_table_markdown()

        if asks_tolc:
            body = (
                "Risposta breve: **nei documenti recuperati non risulta un TOLC specifico "
                "per Scienze dell'Educazione L-19**. Le fonti descrivono invece una "
                f"**prova di ammissione** per il corso. {main_source_label}\n\n"
                "Dettaglio: la prova indicata dalle fonti dura **2 ore e 30 minuti** "
                "e consiste in **80 quesiti a risposta multipla**, articolati così:\n\n"
                f"{test_table}\n"
                "Valutazione:\n"
                "- **1 punto** per ogni risposta esatta;\n"
                "- **0 punti** per ogni risposta omessa o errata;\n"
                "- in caso di parità, prevale prima il voto dell'esame di Stato conclusivo "
                "della scuola secondaria superiore e poi lo studente anagraficamente più giovane.\n\n"
                "Cosa fare:\n"
                "- Non assumere l'esistenza di un TOLC se non è indicato dal bando o dal regolamento.\n"
                "- Usa il documento di immatricolazione L-19 come riferimento operativo.\n"
                "- Verifica eventuali scadenze e modalità di domanda nel bando ufficiale aggiornato.\n"
            )

        elif asks_admission_test:
            body = (
                "Risposta breve: la **prova di ammissione a Scienze dell'Educazione L-19** "
                "è composta da **80 quesiti a risposta multipla** e dura **2 ore e 30 minuti**. "
                f"{main_source_label}\n\n"
                "Struttura della prova:\n\n"
                f"{test_table}\n"
                "Valutazione:\n"
                "- **1 punto** per ogni risposta esatta;\n"
                "- **0 punti** per ogni risposta omessa o errata;\n"
                "- in caso di parità, prevale il voto dell'esame di Stato conclusivo della scuola "
                "secondaria superiore; in caso di ulteriore parità, prevale lo studente "
                "anagraficamente più giovane.\n\n"
                "Cosa fare:\n"
                "- Preparati sulle quattro aree indicate nella tabella.\n"
                "- Controlla nel bando o nel documento di immatricolazione le date, le modalità "
                "di partecipazione e gli eventuali adempimenti amministrativi.\n"
            )

        else:
            body = (
                "Risposta breve: per l'immatricolazione a **Scienze dell'Educazione L-19** "
                "devi fare riferimento alla procedura di accesso/ammissione prevista dal documento "
                f"di immatricolazione del corso. {main_source_label}\n\n"
                "Dettaglio: dalle fonti recuperate emerge che il corso prevede una **prova di ammissione** "
                "della durata di **2 ore e 30 minuti**, composta da **80 quesiti a risposta multipla**. "
                "La prova verifica cultura generale, lingua inglese, abilità logiche e analitiche "
                "e comprensione del testo.\n\n"
                f"{test_table}\n"
                "Valutazione:\n"
                "- **1 punto** per ogni risposta esatta;\n"
                "- **0 punti** per ogni risposta omessa o errata;\n"
                "- in caso di parità, prevale il voto dell'esame di Stato conclusivo della scuola "
                "secondaria superiore e poi il candidato anagraficamente più giovane.\n\n"
                "Cosa fare:\n"
                "- Verifica di possedere i requisiti generali richiesti per l'accesso al corso.\n"
                "- Consulta il documento di immatricolazione L-19 per scadenze, posti disponibili "
                "e modalità operative.\n"
                "- Preparati alla prova sulle aree indicate.\n"
            )

        body += self._format_sources_block(body, sources)
        return prefix + body

    def _postprocess_answer(self, answer: str) -> str:
        cleaned = answer.strip()

        if not cleaned:
            self.last_trace.confidence = "bassa"
            self.last_trace.confidence_reason = "La risposta generata è vuota."
            return "Non lo so in base ai documenti disponibili."

        # Ciclo 2 — FASE 10 — pulizia del caso speciale di astensione.
        # Se il modello segnala incertezza, qui ci si limita ad abbassare la
        # confidenza: la *classificazione* della causa e il messaggio esplicito
        # sono prodotti a valle dal layer di astensione (FASE 6:
        # `is_abstention` / `classify_llm_abstention` / `format_reason`).
        # In precedenza, per `doc_type ∈ {accesso, regolamento, altro}` e
        # argomento "accesso", l'astensione veniva *riscritta* con un testo
        # generico: condizione troppo ampia ("altro") che, riscrivendo la
        # risposta, sfuggiva ai marcatori di `is_abstention` e quindi
        # mascherava l'astensione al layer FASE 6 (causa non classificata,
        # blocco fonti non onesto). Il ramo è ridondante con quel layer ed è
        # stato rimosso, lasciando l'astensione riconoscibile e coerente con
        # la causa. `is_abstention` è l'unica fonte di verità sui marcatori.
        if is_abstention(cleaned):
            self.last_trace.confidence = "bassa"
            self.last_trace.confidence_reason = (
                "La risposta generata segnala insufficienza informativa nonostante le fonti recuperate."
            )

        return cleaned

    def _format_interpretation_block(self, intent: QueryIntent) -> str:
        course = COURSE_LABELS.get(intent.course_tag or "", "non specificato")
        topic = TOPIC_LABELS.get(intent.topic or "", "generale / non specificato")
        memory_note = "sì" if intent.used_memory else "no"

        return (
            "### Interpretazione della richiesta\n"
            f"- Corso rilevato: {course}\n"
            f"- Argomento rilevato: {topic}\n"
            f"- Memoria conversazionale usata: {memory_note}\n\n"
        )

    def _format_interpretation_block_from_trace(self) -> str:
        course = COURSE_LABELS.get(self.last_trace.course_tag or "", "non specificato")
        topic = TOPIC_LABELS.get(self.last_trace.topic or "", "generale / non specificato")
        memory_note = "sì" if self.last_trace.used_memory else "no"

        return (
            "### Interpretazione della richiesta\n"
            f"- Corso rilevato: {course}\n"
            f"- Argomento rilevato: {topic}\n"
            f"- Memoria conversazionale usata: {memory_note}\n"
        )

    def _format_confidence_block(self) -> str:
        confidence = self.last_trace.confidence
        reason = self.last_trace.confidence_reason

        return (
            "### Affidabilità della risposta\n"
            f"- Livello: {confidence}\n"
            f"- Motivo: {reason}\n\n"
        )

    def _extract_cited_source_indexes(
        self,
        answer: str,
        sources: List[RetrievedSource],
    ) -> set[int]:
        return extract_cited_source_indexes(answer, sources)

    def _format_sources_block(
        self, answer: str, sources: List[RetrievedSource], abstaining: bool = False
    ) -> str:
        return format_sources_block(answer, sources, abstaining=abstaining)


@st.cache_resource(show_spinner=False)
def get_cached_responder(_vector_db):
    return UniLawResponder(_vector_db)
