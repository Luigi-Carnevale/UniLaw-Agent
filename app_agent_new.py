"""UniLaw Agent — interfaccia 'accademica' (versione in valutazione).

Variante dell'interfaccia con tono caldo e istituzionale al posto del tema
'terminale'. Riusa **identica** la pipeline RAG (`agent.py`, `database.py`, ...):
cambia solo la presentazione. Vive in un file separato per poter essere provata
affiancata a quella attuale senza rischi:

    streamlit run app_agent.py        # interfaccia attuale (terminale)
    streamlit run app_agent_new.py    # nuova interfaccia (accademica)

Se non viene approvata, eliminare questo file e `theme_light.py`: l'app originale
resta intatta.
"""

import html
import os
import re

import streamlit as st

import trace_export
from agent import get_cached_responder, setup_redis_cache
from config import (
    DEFAULT_MODEL_NAME,
    DOCUMENTS_FOLDER,
    RERANKER_ENABLED,
    setup_environment,
)
from database import calcola_firma_documenti, inizializza_conoscenza
from rag_types import COURSE_LABELS, TOPIC_LABELS
from theme_light import CSS_STYLES_LIGHT


# ============================================================
# Environment / page setup
# ============================================================

setup_environment()

st.set_page_config(
    page_title="UniLaw Agent — Assistente accademico",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS_STYLES_LIGHT, unsafe_allow_html=True)


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "force_rebuild" not in st.session_state:
    st.session_state.force_rebuild = False

if "rag_memory" not in st.session_state:
    st.session_state.rag_memory = {}

if "last_rag_trace" not in st.session_state:
    st.session_state.last_rag_trace = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# Backend initialization (identico all'app originale)
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
# Helpers di presentazione
# ============================================================

CONFIDENCE_LABEL = {
    "alta": "Affidabilità alta",
    "media": "Affidabilità media",
    "bassa": "Affidabilità bassa",
}

# Avatar custom solo per l'assistente: l'utente resta sull'avatar di default così
# il suo messaggio mantiene il testid 'chatAvatarIcon-user' usato dal CSS per la bolla.
# (Le icone di marchio e fonti sono disegnate via CSS background-image in theme_light.py,
# perché Streamlit strippa l'SVG inline dentro st.markdown.)
ASSISTANT_AVATAR = "⚖️"

SOURCE_TITLES = ("Fonti citate:", "Fonti utilizzate:")
_SOURCE_LINE = re.compile(r"^\[F\d+\]\s+(.+?)(?:,\s*pag\.\s*(\d+))?$")


def split_body_and_sources(answer: str):
    """Separa il corpo della risposta dal blocco fonti accodato dall'agente.

    L'agente termina sempre la risposta con un blocco tipo
    ``--- / Fonti citate: / - [F1] file, pag. 3``. Qui lo isoliamo per renderlo
    come schede, mostrando il corpo pulito nella bolla di chat.
    """
    body_lines: list[str] = []
    source_lines: list[str] = []
    in_block = False

    for line in answer.split("\n"):
        stripped = line.strip()
        is_title = stripped in SOURCE_TITLES or stripped.startswith("Documenti consultati")

        if is_title:
            in_block = True
            while body_lines and body_lines[-1].strip() in ("", "---"):
                body_lines.pop()
            continue

        if in_block:
            if stripped.startswith("- "):
                source_lines.append(stripped[2:].strip())
                continue
            if stripped == "":
                continue
            in_block = False  # un paragrafo (caveat/nota) chiude il blocco fonti
            if body_lines and body_lines[-1].strip() != "":
                body_lines.append("")  # mantiene il caveat come paragrafo a sé

        body_lines.append(line)

    body = "\n".join(body_lines).strip()

    parsed: list[tuple[str, str | None]] = []
    for raw in source_lines:
        match = _SOURCE_LINE.match(raw)
        if match:
            parsed.append((match.group(1).strip(), match.group(2)))
        else:
            parsed.append((raw, None))

    return body, parsed


def confidence_pill_html(level: str) -> str:
    level = level if level in CONFIDENCE_LABEL else "bassa"
    return (
        f'<span class="ul-pill {level}"><span class="ul-dot"></span>'
        f"{CONFIDENCE_LABEL[level]}</span>"
    )


def source_cards_html(parsed: list[tuple[str, str | None]]) -> str:
    cards = []
    for name, page in parsed:
        safe_name = html.escape(name)
        page_text = f"pag. {page}" if page else "documento"
        cards.append(
            '<div class="ul-source-card">'
            '<span class="ul-source-ico"></span>'
            f'<div><div class="ul-source-name">{safe_name}</div>'
            f'<div class="ul-source-page">{page_text}</div></div></div>'
        )
    return '<div class="ul-source-grid">' + "".join(cards) + "</div>"


def interp_from_trace(trace) -> dict:
    return {
        "corso": COURSE_LABELS.get(trace.course_tag or "", "non rilevato"),
        "argomento": TOPIC_LABELS.get(trace.topic or "", "non rilevato"),
        "memoria": "sì" if trace.used_memory else "no",
        "recupero": getattr(trace, "retrieval_mode", "vettoriale"),
        "reranker": getattr(trace, "reranker", "euristico"),
        "grounding": getattr(trace, "grounding", "n.d."),
        "motivo": trace.confidence_reason or "—",
        "astensione": trace.abstention_reason or "nessuna",
    }


def render_come_ho_risposto(interp: dict) -> None:
    rows = [
        ("Corso individuato", interp["corso"]),
        ("Argomento", interp["argomento"]),
        ("Memoria conversazionale", interp["memoria"]),
        ("Recupero documenti", f"{interp['recupero']} · reranker {interp['reranker']}"),
        ("Verifica citazioni", interp["grounding"]),
        ("Perché questa affidabilità", interp["motivo"]),
        ("Astensione", interp["astensione"]),
    ]
    body = "".join(
        f'<div class="ul-explain-row"><span>{html.escape(str(label))}</span>'
        f"<strong>{html.escape(str(value))}</strong></div>"
        for label, value in rows
    )
    with st.expander("Come ho risposto"):
        st.markdown(f'<div class="ul-explain">{body}</div>', unsafe_allow_html=True)


def render_advanced_trace(trace, key_prefix: str = "adv") -> None:
    if trace is None:
        st.info("Nessun dettaglio disponibile.")
        return

    cols = st.columns(3)
    cols[0].metric("Corso", trace.course_tag or "—")
    cols[1].metric("Argomento", trace.topic or "—")
    cols[2].metric("Affidabilità", trace.confidence or "—")

    st.markdown("**Query generate**")
    if trace.query_variants:
        for query in trace.query_variants:
            st.markdown(f"- `{query}`")
    else:
        st.caption("Nessuna query generata.")

    st.markdown("**Retrieval**")
    st.markdown(f"- Modalità: `{getattr(trace, 'retrieval_mode', 'vettoriale')}`")
    st.markdown(f"- Reranker: `{getattr(trace, 'reranker', 'euristico')}`")
    st.markdown(f"- Verifica citazioni: `{getattr(trace, 'grounding', 'n.d.')}`")
    st.markdown(f"- Astensione: `{trace.abstention_reason or 'nessuna'}`")
    for line in getattr(trace, "fusion_scores", None) or []:
        st.markdown(f"- `{line}`")

    st.markdown("**Fonti selezionate**")
    if trace.selected_sources:
        for source in trace.selected_sources:
            st.markdown(f"- `{source}`")
    else:
        st.caption("Nessuna fonte selezionata.")

    if trace.rejected_hint:
        st.markdown("**Scartati dopo reranking**")
        for source in trace.rejected_hint:
            st.markdown(f"- `{source}`")

    col_json, col_md = st.columns(2)
    col_json.download_button(
        "⬇ JSON",
        data=trace_export.trace_to_json(trace),
        file_name="rag_trace.json",
        mime="application/json",
        use_container_width=True,
        key=f"{key_prefix}_json",
    )
    col_md.download_button(
        "⬇ Markdown",
        data=trace_export.trace_to_markdown(trace),
        file_name="rag_trace.md",
        mime="text/markdown",
        use_container_width=True,
        key=f"{key_prefix}_md",
    )


def render_assistant(message: dict, show_confidence: bool, show_interpretation: bool) -> None:
    confidence = message.get("confidence")
    if show_confidence and confidence:
        st.markdown(confidence_pill_html(confidence), unsafe_allow_html=True)

    st.markdown(message["content"])

    parsed = message.get("sources") or []
    if parsed:
        st.markdown('<div class="ul-section-label">Fonti citate</div>', unsafe_allow_html=True)
        st.markdown(source_cards_html(parsed), unsafe_allow_html=True)

    interp = message.get("interp")
    if show_interpretation and interp:
        render_come_ho_risposto(interp)


def reset_chat():
    st.session_state.messages = []
    st.session_state.rag_memory = {}
    st.session_state.last_rag_trace = None


def ask_from_button(question: str):
    st.session_state.pending_prompt = question


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div class="ul-brand">
            <div class="ul-brand-mark"></div>
            <div>
                <div class="ul-brand-name">UniLaw Agent</div>
                <div class="ul-brand-sub">Assistente accademico</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kb_ok = bool(vector_db)
    head_class = "ok" if kb_ok else "ko"
    head_text = "Base di conoscenza pronta" if kb_ok else "Base di conoscenza non disponibile"

    st.markdown(
        f"""
        <div class="ul-status-card">
            <div class="ul-status-head {head_class}"><span class="ul-dot"></span>{head_text}</div>
            <div class="ul-status-row"><span>Documenti indicizzati</span><strong>{documents_count}</strong></div>
            <div class="ul-status-row"><span>Modello</span><strong class="mono">{DEFAULT_MODEL_NAME}</strong></div>
            <div class="ul-status-row"><span>Cache Redis</span><strong>{"attiva" if redis_enabled else "non attiva"}</strong></div>
            <div class="ul-status-row"><span>Privacy</span><strong>tutto in locale</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not kb_ok:
        st.error(msg)

    st.markdown('<div class="ul-section-label">Opzioni risposta</div>', unsafe_allow_html=True)

    show_interpretation = st.toggle(
        "Mostra «Come ho risposto»",
        value=True,
        help="Mostra corso, argomento, recupero e motivo dell'affidabilità sotto la risposta.",
    )
    show_confidence = st.toggle(
        "Mostra affidabilità",
        value=True,
        help="Mostra la stima di affidabilità (alta / media / bassa).",
    )
    use_reranker = st.toggle(
        "Reranker neurale (cross-encoder)",
        value=RERANKER_ENABLED,
        help=(
            "Riordina i candidati con un cross-encoder multilingua. Al primo uso "
            "carica il modello (~80 s una tantum su CPU)."
        ),
    )
    show_debug = st.toggle(
        "Modalità avanzata (debug RAG)",
        value=False,
        help="Mostra query generate, scoring di fusione, fonti e trace esportabile.",
    )

    st.markdown('<div class="ul-section-label">Strumenti</div>', unsafe_allow_html=True)

    with st.expander("Aggiungi documenti (PDF)"):
        uploaded_files = st.file_uploader(
            "Carica PDF da aggiungere al corpus",
            type=["pdf"],
            accept_multiple_files=True,
            help="I file vengono salvati in 'documenti/' e l'indice viene ricostruito.",
        )
        if uploaded_files and st.button("Salva e ricostruisci", use_container_width=True):
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
            st.success(f"Salvati {saved} PDF. Ricostruzione in corso...")
            st.rerun()

    if st.button("Ricostruisci base di conoscenza", use_container_width=True):
        inizializza_conoscenza.clear()
        get_cached_responder.clear()
        st.session_state.force_rebuild = True
        st.rerun()

    col_a, col_b = st.columns(2)
    if col_a.button("Nuova chat", use_container_width=True):
        reset_chat()
        st.rerun()
    if col_b.button("Reset memoria", use_container_width=True):
        st.session_state.rag_memory = {}
        st.session_state.last_rag_trace = None
        st.rerun()

    with st.expander("Documenti indicizzati"):
        if not documents:
            st.caption("Nessun PDF trovato.")
        else:
            for doc in documents:
                st.markdown(f"- `{doc['filename']}`")

    if show_debug:
        with st.expander("Memoria conversazionale"):
            if not st.session_state.rag_memory:
                st.caption("Nessuna memoria attiva.")
            else:
                st.json(st.session_state.rag_memory)

        with st.expander("Debug — ultimo retrieval"):
            render_advanced_trace(st.session_state.last_rag_trace, key_prefix="side")


# ============================================================
# Header — welcome ricco a conversazione vuota, compatto durante la chat
# ============================================================

SUGGESTIONS = [
    "Posso immatricolarmi con 11 al TOLC-I per Informatica?",
    "Come funziona l'accesso a Informatica L-31 in base al TOLC-I?",
    "Come funziona la prova finale e la tesi?",
    "La tesi è consultabile dopo la laurea?",
    "Cosa dicono i documenti sul bando Erasmus?",
    "Quali regole seguo per il piano di studi di Informatica L-31?",
]

if not st.session_state.messages:
    st.markdown(
        f"""
        <section class="ul-hero">
            <div class="ul-brand">
                <div class="ul-brand-mark"></div>
                <div>
                    <div class="ul-brand-name">UniLaw Agent</div>
                    <div class="ul-brand-sub">Università degli Studi di Salerno · assistente accademico</div>
                </div>
            </div>
            <h1>Ciao, sono UniLaw.</h1>
            <p>
                Rispondo a domande su regolamenti, bandi, piani di studio e tesi
                consultando <span class="ul-accent">solo i documenti ufficiali</span>
                indicizzati e citando sempre la fonte.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ul-eyebrow">Prova a chiedere</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for index, question in enumerate(SUGGESTIONS):
        if cols[index % 3].button(question, key=f"suggest_{index}", use_container_width=True):
            ask_from_button(question)
else:
    st.markdown(
        f"""
        <div class="ul-topbar">
            <div class="ul-brand-mark"></div>
            <div class="ul-topbar-name">UniLaw Agent</div>
            <div class="ul-topbar-sub">risponde solo dai documenti ufficiali, citando le fonti</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Chat stream
# ============================================================

for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            render_assistant(message, show_confidence, show_interpretation)
    else:
        with st.chat_message("user"):
            st.markdown(message["content"])


# ============================================================
# Input utente
# ============================================================

typed_prompt = st.chat_input("Scrivi la tua domanda a UniLaw Agent…")

pending_prompt = st.session_state.pending_prompt
if pending_prompt:
    prompt = pending_prompt
    st.session_state.pending_prompt = None
elif typed_prompt:
    prompt = typed_prompt
else:
    prompt = None


# ============================================================
# Esecuzione
# ============================================================

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if not vector_db:
            st.error("Base di conoscenza non disponibile. Prova a ricostruirla dalla barra laterale.")
        else:
            assistant_message = None
            error_message = None
            trace = None

            # Lo status avvolge SOLO il lavoro della pipeline: il rendering (che usa
            # expander) va fatto fuori, perché st.status è a sua volta un expander e
            # Streamlit non consente expander annidati.
            with st.status("Sto consultando i documenti…", expanded=False) as status:
                try:
                    responder = get_cached_responder(vector_db)
                    responder.use_neural_reranker = use_reranker

                    raw_response = responder.answer(
                        prompt,
                        chat_history=st.session_state.messages[:-1],
                        memory=st.session_state.rag_memory,
                        show_interpretation=False,
                        show_confidence=False,
                    )

                    st.session_state.rag_memory = responder.update_memory_from_trace(
                        st.session_state.rag_memory
                    )
                    trace = responder.last_trace
                    st.session_state.last_rag_trace = trace

                    body, parsed_sources = split_body_and_sources(raw_response)
                    assistant_message = {
                        "role": "assistant",
                        "content": body,
                        "sources": parsed_sources,
                        "confidence": trace.confidence,
                        "interp": interp_from_trace(trace),
                    }

                    status.update(label="Risposta pronta.", state="complete", expanded=False)

                except Exception as exc:
                    status.update(label="Errore durante l'elaborazione", state="error", expanded=False)
                    error_message = (
                        "**Si è verificato un errore durante l'elaborazione.**\n\n"
                        f"Dettaglio tecnico: `{exc}`\n\n"
                        "Possibili cause comuni:\n"
                        "- Ollama non in esecuzione o modello non scaricato "
                        "(`ollama serve`, `ollama pull llama3.1:8b`);\n"
                        "- base di conoscenza non disponibile: prova *Ricostruisci base di conoscenza*;\n"
                        "- un PDF appena aggiunto è problematico.\n\n"
                        "Riprova oppure riformula la domanda."
                    )

            if assistant_message is not None:
                render_assistant(assistant_message, show_confidence, show_interpretation)

                if show_debug:
                    with st.expander("Dettagli tecnici (questa risposta)"):
                        render_advanced_trace(trace, key_prefix="inline")

                st.session_state.messages.append(assistant_message)

            elif error_message is not None:
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
