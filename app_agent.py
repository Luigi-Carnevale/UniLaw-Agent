import os

import streamlit as st

import trace_export
from agent import get_cached_responder, setup_redis_cache
from config import (
    CITATION_GROUNDING_ENABLED,
    CSS_STYLES,
    DEFAULT_MODEL_NAME,
    DOCUMENTS_FOLDER,
    EVIDENCE_SELECTION_ENABLED,
    RERANKER_ENABLED,
    setup_environment,
)
from database import calcola_firma_documenti, inizializza_conoscenza


# ============================================================
# Environment / page setup
# ============================================================

setup_environment()

st.set_page_config(
    page_title="UniLaw Agent | Legal Intelligence Console",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS_STYLES, unsafe_allow_html=True)


# ============================================================
# Session state
# ============================================================

WELCOME_MESSAGE = (
    "Sono UniLaw Agent. Rispondo consultando i documenti caricati "
    "e cito le fonti utilizzate."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]

if "force_rebuild" not in st.session_state:
    st.session_state.force_rebuild = False

if "rag_memory" not in st.session_state:
    st.session_state.rag_memory = {}

if "last_rag_trace" not in st.session_state:
    st.session_state.last_rag_trace = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# Backend initialization
# ============================================================

redis_enabled = setup_redis_cache()

docs_signature = calcola_firma_documenti()
documents = docs_signature.get("documents", [])
documents_count = len(documents)

vector_db, msg = inizializza_conoscenza(
    docs_signature=docs_signature,
    force_rebuild=st.session_state.force_rebuild,
)

st.session_state.force_rebuild = False


# ============================================================
# Small UI helpers
# ============================================================

def status_label(ok: bool, active_text: str = "ONLINE", inactive_text: str = "OFFLINE") -> str:
    return active_text if ok else inactive_text


def status_class(ok: bool) -> str:
    return "status-ok" if ok else "status-ko"


def render_terminal_kpi(label: str, value: str, detail: str, state: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="terminal-kpi {state}">
            <div class="terminal-kpi-label">{label}</div>
            <div class="terminal-kpi-value">{value}</div>
            <div class="terminal-kpi-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trace(trace, title="TRACE // LAST RETRIEVAL"):
    if trace is None:
        st.info("Nessun debug disponibile.")
        return

    with st.expander(title, expanded=True):
        st.markdown('<div class="trace-grid">', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="trace-card">
                <div class="trace-label">QUESTION</div>
                <div class="trace-value">{trace.question}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="trace-card">
                <div class="trace-label">COURSE</div>
                <div class="trace-value">{trace.course_tag or "non rilevato"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="trace-card">
                <div class="trace-label">TOPIC</div>
                <div class="trace-value">{trace.topic or "non rilevato"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="trace-card">
                <div class="trace-label">MEMORY</div>
                <div class="trace-value">{trace.used_memory}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="trace-card">
                <div class="trace-label">CONFIDENCE</div>
                <div class="trace-value">{trace.confidence or "n.d."}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Confidence reason")
        st.code(trace.confidence_reason or "Nessun motivo disponibile.", language="text")

        if getattr(trace, "deterministic_rule_used", None):
            st.markdown("#### Deterministic rule")
            st.code(trace.deterministic_rule_used, language="text")

        st.markdown("#### Answer profile")
        st.code(trace.answer_profile or "Nessun profilo disponibile.", language="text")

        st.markdown("#### Query generate")
        if trace.query_variants:
            for query in trace.query_variants:
                st.markdown(f"- `{query}`")
        else:
            st.write("Nessuna query generata.")

        st.markdown("#### Retrieval ibrido (RRF)")
        st.markdown(f"- Modalità: `{getattr(trace, 'retrieval_mode', 'vettoriale')}`")
        st.markdown(f"- Reranker: `{getattr(trace, 'reranker', 'euristico')}`")
        st.markdown(f"- Evidence: `{getattr(trace, 'evidence_chars', '') or 'n.d.'}`")
        st.markdown(f"- Grounding citazioni: `{getattr(trace, 'grounding', 'n.d.')}`")
        st.markdown(f"- Astensione: `{getattr(trace, 'abstention_reason', '') or 'nessuna'}`")
        fusion_scores = getattr(trace, "fusion_scores", None)
        if fusion_scores:
            for line in fusion_scores:
                st.markdown(f"- `{line}`")
        else:
            st.write("Nessuno scoring di fusione disponibile.")

        st.markdown("#### Fonti selezionate")
        if trace.selected_sources:
            for source in trace.selected_sources:
                st.markdown(f"- `{source}`")
        else:
            st.write("Nessuna fonte selezionata.")

        st.markdown("#### Documenti scartati dopo reranking")
        if trace.rejected_hint:
            for source in trace.rejected_hint:
                st.markdown(f"- `{source}`")
        else:
            st.write("Nessun documento scartato mostrato.")

        st.markdown("#### Esporta trace")
        export_cols = st.columns(2)
        with export_cols[0]:
            st.download_button(
                "⬇ JSON",
                data=trace_export.trace_to_json(trace),
                file_name="rag_trace.json",
                mime="application/json",
                use_container_width=True,
                key=f"dl_json_{title}",
            )
        with export_cols[1]:
            st.download_button(
                "⬇ Markdown",
                data=trace_export.trace_to_markdown(trace),
                file_name="rag_trace.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"dl_md_{title}",
            )


def reset_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]
    st.session_state.rag_memory = {}
    st.session_state.last_rag_trace = None


def ask_from_button(question: str):
    st.session_state.pending_prompt = question


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="side-terminal">
            <div class="side-terminal-dotline">
                <span class="dot red"></span>
                <span class="dot yellow"></span>
                <span class="dot green"></span>
            </div>
            <div class="side-terminal-title">RAG CONTROL PANEL</div>
            <div class="side-terminal-subtitle">UniLaw Agent runtime</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### SYSTEM STATUS")

    if vector_db:
        st.success(msg)
    else:
        st.error(msg)

    st.markdown(
        f"""
        <div class="sidebar-status-row">
            <span>Redis cache</span>
            <strong class="{status_class(redis_enabled)}">{status_label(redis_enabled, "ACTIVE", "DISABLED")}</strong>
        </div>
        <div class="sidebar-status-row">
            <span>Knowledge base</span>
            <strong class="{status_class(bool(vector_db))}">{status_label(bool(vector_db), "READY", "ERROR")}</strong>
        </div>
        <div class="sidebar-status-row">
            <span>Documents</span>
            <strong>{documents_count}</strong>
        </div>
        <div class="sidebar-status-row">
            <span>LLM</span>
            <strong>{DEFAULT_MODEL_NAME}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### PIPELINE")
    st.markdown(
        f"""
        <div class="sidebar-status-row">
            <span>Retrieval</span>
            <strong>HYBRID (vector + BM25)</strong>
        </div>
        <div class="sidebar-status-row">
            <span>Reranker neurale</span>
            <strong class="{status_class(RERANKER_ENABLED)}">{status_label(RERANKER_ENABLED, "ON", "OFF")}</strong>
        </div>
        <div class="sidebar-status-row">
            <span>Evidence selection</span>
            <strong class="{status_class(EVIDENCE_SELECTION_ENABLED)}">{status_label(EVIDENCE_SELECTION_ENABLED, "ON", "OFF")}</strong>
        </div>
        <div class="sidebar-status-row">
            <span>Verifica citazioni</span>
            <strong class="{status_class(CITATION_GROUNDING_ENABLED)}">{status_label(CITATION_GROUNDING_ENABLED, "ON", "OFF")}</strong>
        </div>
        <div class="sidebar-status-row">
            <span>Astensione</span>
            <strong>CLASSIFICATA</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### RESPONSE OPTIONS")

    show_interpretation = st.toggle(
        "Interpretazione richiesta",
        value=True,
        help="Mostra corso, argomento e uso della memoria controllata.",
    )

    show_confidence = st.toggle(
        "Affidabilità risposta",
        value=True,
        help="Mostra una stima euristica di affidabilità basata sulle fonti recuperate.",
    )

    show_debug = st.toggle(
        "Debug RAG globale",
        value=False,
        help="Mostra query generate, fonti selezionate e memoria a slot.",
    )

    show_inline_debug = st.toggle(
        "Debug sotto risposta",
        value=False,
        help="Mostra il debug della risposta appena generata direttamente nella chat.",
    )

    use_reranker = st.toggle(
        "Reranker neurale (cross-encoder)",
        value=RERANKER_ENABLED,
        help=(
            "Riordina i candidati con un cross-encoder multilingua. Opzionale: al "
            "primo uso carica il modello (~80 s una tantum su CPU). Se non disponibile, "
            "resta il reranking euristico."
        ),
    )

    st.divider()

    st.markdown("### OPERATIONS")

    with st.expander("AGGIUNGI DOCUMENTI (PDF)"):
        uploaded_files = st.file_uploader(
            "Carica PDF da aggiungere al corpus",
            type=["pdf"],
            accept_multiple_files=True,
            help="I file vengono salvati nella cartella 'documenti/' e l'indice viene ricostruito.",
        )
        if uploaded_files and st.button("SALVA E RICOSTRUISCI", use_container_width=True):
            os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
            saved = 0
            for uploaded in uploaded_files:
                destination = os.path.join(DOCUMENTS_FOLDER, os.path.basename(uploaded.name))
                with open(destination, "wb") as out_file:
                    out_file.write(uploaded.getbuffer())
                saved += 1
            inizializza_conoscenza.clear()
            get_cached_responder.clear()
            st.session_state.force_rebuild = True
            st.success(f"Salvati {saved} PDF. Ricostruzione della knowledge base in corso...")
            st.rerun()

    if st.button("REBUILD KNOWLEDGE BASE", use_container_width=True):
        inizializza_conoscenza.clear()
        get_cached_responder.clear()
        st.session_state.force_rebuild = True
        st.rerun()

    if st.button("CLEAR CHAT", use_container_width=True):
        reset_chat()
        st.rerun()

    if st.button("RESET RAG MEMORY", use_container_width=True):
        st.session_state.rag_memory = {}
        st.session_state.last_rag_trace = None
        st.rerun()

    st.divider()

    with st.expander("MEMORY SLOT"):
        if not st.session_state.rag_memory:
            st.caption("Nessuna memoria attiva.")
        else:
            st.json(st.session_state.rag_memory)

    with st.expander("INDEXED DOCUMENTS"):
        if not documents:
            st.caption("Nessun PDF trovato.")
        else:
            for doc in documents:
                st.markdown(f"- `{doc['filename']}`")

    if show_debug and st.session_state.last_rag_trace is not None:
        render_trace(st.session_state.last_rag_trace, title="TRACE // LAST RETRIEVAL")


# ============================================================
# Main terminal interface
# ============================================================

st.markdown(
    """
    <section class="terminal-hero">
        <div class="terminal-window-bar">
            <span class="dot red"></span>
            <span class="dot yellow"></span>
            <span class="dot green"></span>
            <span class="terminal-window-title">/unilaw/agent/console</span>
        </div>

        <div class="terminal-command-line">
            <span class="terminal-prompt">luigi@unilaw-agent</span>
            <span class="terminal-symbol">:</span>
            <span class="terminal-path">~/documenti</span>
            <span class="terminal-dollar">$</span>
            <span class="terminal-command">run rag --grounded --cite-sources --local</span>
        </div>

        <h1>Legal Intelligence Console</h1>
        <p>
            Interroga regolamenti, bandi, piani di studio e guide universitarie
            con un assistente RAG locale, tracciabile e orientato alle fonti.
        </p>

        <div class="terminal-badge-row">
            <span>LOCAL-FIRST</span>
            <span>OLLAMA</span>
            <span>CHROMADB</span>
            <span>RAG TRACE</span>
            <span>CITED SOURCES</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    render_terminal_kpi(
        label="CORPUS",
        value=str(documents_count),
        detail="PDF indicizzati",
        state="cyan",
    )

with kpi_col2:
    render_terminal_kpi(
        label="VECTOR DB",
        value="READY" if vector_db else "ERROR",
        detail="ChromaDB persistent index",
        state="green" if vector_db else "red",
    )

with kpi_col3:
    render_terminal_kpi(
        label="MODEL",
        value=DEFAULT_MODEL_NAME,
        detail="Ollama local runtime",
        state="violet",
    )

with kpi_col4:
    render_terminal_kpi(
        label="CACHE",
        value="ACTIVE" if redis_enabled else "DISABLED",
        detail="Redis LLM cache",
        state="green" if redis_enabled else "neutral",
    )


st.markdown(
    """
    <div class="terminal-section-title">
        <span>DEMO COMMANDS</span>
        <small>Seleziona una query pronta o scrivi un comando nella chat.</small>
    </div>
    """,
    unsafe_allow_html=True,
)

quick_questions = [
    (
        "ACCESSO",
        "Ho preso 11 al TOLC-I per Informatica: posso immatricolarmi?",
    ),
    (
        "TOLC TABLE",
        "Come funziona l'accesso a Informatica L-31 in base al punteggio del TOLC-I?",
    ),
    (
        "TESI",
        "E per la tesi?",
    ),
    (
        "CONSULTABILITÀ",
        "La tesi è consultabile dopo la laurea?",
    ),
    (
        "ERASMUS",
        "Quali informazioni forniscono i documenti disponibili sul bando Erasmus e sulla mobilità internazionale?",
    ),
    (
        "OUT OF DOMAIN",
        "Quali sono le regole di accesso al corso di Medicina e Chirurgia?",
    ),
]

q_cols = st.columns(3)

for index, (label, question) in enumerate(quick_questions):
    with q_cols[index % 3]:
        st.markdown(
            f"""
            <div class="quick-command-card">
                <div class="quick-command-label">{label}</div>
                <div class="quick-command-text">{question}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(f"RUN // {label}", key=f"quick_{index}", use_container_width=True):
            ask_from_button(question)


st.markdown(
    """
    <div class="terminal-section-title chat-title">
        <span>CONVERSATION STREAM</span>
        <small>Le risposte sono generate solo sui documenti disponibili nel corpus.</small>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Chat rendering
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# Input utente
# ============================================================

typed_prompt = st.chat_input(
    "ask> Scrivi una domanda da porre a UniLaw Agent..."
)

pending_prompt = st.session_state.pending_prompt

if pending_prompt:
    prompt = pending_prompt
    st.session_state.pending_prompt = None
elif typed_prompt:
    prompt = typed_prompt
else:
    prompt = None


# ============================================================
# Prompt execution
# ============================================================

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not vector_db:
            st.error("Database non disponibile. Controlla la knowledge base.")

        else:
            with st.status(
                "RAG pipeline in esecuzione: retrieval → reranking → grounding → risposta",
                expanded=True,
            ) as status:
                try:
                    responder = get_cached_responder(vector_db)
                    responder.use_neural_reranker = use_reranker

                    response = responder.answer(
                        prompt,
                        chat_history=st.session_state.messages[:-1],
                        memory=st.session_state.rag_memory,
                        show_interpretation=show_interpretation,
                        show_confidence=show_confidence,
                    )

                    st.session_state.rag_memory = responder.update_memory_from_trace(
                        st.session_state.rag_memory
                    )

                    st.session_state.last_rag_trace = responder.last_trace

                    status.update(
                        label="Pipeline completata. Risposta grounded generata.",
                        state="complete",
                        expanded=False,
                    )

                    st.markdown(response)

                    if show_inline_debug:
                        render_trace(
                            responder.last_trace,
                            title="TRACE // CURRENT RESPONSE",
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                        }
                    )

                except Exception as exc:
                    status.update(
                        label="Errore nella pipeline RAG",
                        state="error",
                        expanded=True,
                    )

                    error_message = (
                        f"**Si è verificato un errore durante l'elaborazione.**\n\n"
                        f"Dettaglio tecnico: `{exc}`\n\n"
                        "Possibili cause comuni:\n"
                        "- Ollama non in esecuzione o modello non scaricato "
                        "(`ollama serve`, `ollama pull llama3.1:8b`);\n"
                        "- knowledge base non disponibile: prova *REBUILD KNOWLEDGE BASE* nella sidebar;\n"
                        "- documento PDF problematico appena aggiunto.\n\n"
                        "Riprova oppure riformula la domanda."
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )