import logging
import os
import warnings

from langchain.prompts import PromptTemplate


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_environment():
    """
    Configura ambiente, warning e logging dell'applicazione.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = "false"

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message="`resume_download` is deprecated",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="ARC4 has been moved",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Since Chroma 0.4.x the manual persistence method is no longer supported",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message="The method `BaseChatModel.predict` was deprecated",
        category=Warning,
    )


CSS_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --terminal-bg: #030712;
        --terminal-bg-2: #07111f;
        --terminal-card: rgba(8, 18, 34, 0.86);
        --terminal-card-2: rgba(15, 23, 42, 0.88);
        --terminal-border: rgba(125, 211, 252, 0.28);
        --terminal-border-soft: rgba(148, 163, 184, 0.16);
        --terminal-text: #e5f4ff;
        --terminal-muted: #8ea4b8;
        --terminal-cyan: #22d3ee;
        --terminal-green: #34d399;
        --terminal-violet: #a78bfa;
        --terminal-red: #fb7185;
        --terminal-yellow: #facc15;
        --terminal-blue: #60a5fa;
        --terminal-shadow: rgba(34, 211, 238, 0.24);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(34, 211, 238, 0.12), transparent 34%),
            radial-gradient(circle at 80% 10%, rgba(167, 139, 250, 0.15), transparent 30%),
            radial-gradient(circle at 50% 100%, rgba(52, 211, 153, 0.08), transparent 30%),
            linear-gradient(135deg, #020617 0%, #030712 42%, #07111f 100%);
        color: var(--terminal-text);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(3, 7, 18, 0.98), rgba(7, 17, 31, 0.98));
        border-right: 1px solid rgba(125, 211, 252, 0.18);
    }

    section[data-testid="stSidebar"] * {
        color: #dbeafe;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(15, 23, 42, 0.92);
        color: #e0f2fe;
        border: 1px solid rgba(34, 211, 238, 0.28);
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        transition: all 0.18s ease-in-out;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(34, 211, 238, 0.72);
        box-shadow: 0 0 22px rgba(34, 211, 238, 0.18);
        transform: translateY(-1px);
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(125, 211, 252, 0.16);
        border-radius: 14px;
    }

    [data-testid="stStatusWidget"] {
        visibility: hidden;
    }

    .stStatus {
        background: rgba(8, 18, 34, 0.82);
        border: 1px solid rgba(34, 211, 238, 0.24);
        border-radius: 16px;
    }

    .stChatMessage {
        background: rgba(8, 18, 34, 0.70);
        border: 1px solid rgba(125, 211, 252, 0.18);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.22);
    }

    .stChatMessage [data-testid="stMarkdownContainer"] {
        color: #e5f4ff;
    }

    .stChatInputContainer {
        border-top: 1px solid rgba(125, 211, 252, 0.16);
        background: rgba(3, 7, 18, 0.86);
    }

    textarea {
        background: rgba(15, 23, 42, 0.95) !important;
        color: #e0f2fe !important;
        border: 1px solid rgba(34, 211, 238, 0.30) !important;
        border-radius: 16px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    div[data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(125, 211, 252, 0.20);
        background: rgba(8, 18, 34, 0.70);
    }

    div[data-testid="stMarkdownContainer"] th {
        background: rgba(14, 165, 233, 0.16);
        color: #e0f2fe;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    div[data-testid="stMarkdownContainer"] td {
        border-top: 1px solid rgba(125, 211, 252, 0.12);
        color: #dbeafe;
    }

    code {
        color: #67e8f9 !important;
        background: rgba(34, 211, 238, 0.08) !important;
        border: 1px solid rgba(34, 211, 238, 0.14);
        border-radius: 6px;
        padding: 0.08rem 0.28rem;
        font-family: 'JetBrains Mono', monospace !important;
    }

    pre {
        background: rgba(2, 6, 23, 0.92) !important;
        border: 1px solid rgba(34, 211, 238, 0.18);
        border-radius: 14px;
    }

    .terminal-hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2rem 2.2rem 2rem;
        margin-bottom: 1.4rem;
        border-radius: 28px;
        border: 1px solid rgba(34, 211, 238, 0.32);
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.94)),
            radial-gradient(circle at top right, rgba(34, 211, 238, 0.20), transparent 30%);
        box-shadow:
            0 0 0 1px rgba(255, 255, 255, 0.03) inset,
            0 34px 90px rgba(0, 0, 0, 0.38),
            0 0 60px rgba(34, 211, 238, 0.10);
    }

    .terminal-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(125, 211, 252, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(125, 211, 252, 0.05) 1px, transparent 1px);
        background-size: 28px 28px;
        mask-image: linear-gradient(to bottom, rgba(0,0,0,0.9), transparent);
        pointer-events: none;
    }

    .terminal-hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -60px;
        top: -80px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.26), transparent 68%);
        filter: blur(2px);
        pointer-events: none;
    }

    .terminal-window-bar {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.3rem;
    }

    .dot {
        width: 12px;
        height: 12px;
        display: inline-block;
        border-radius: 999px;
        box-shadow: 0 0 12px rgba(255,255,255,0.15);
    }

    .dot.red {
        background: #fb7185;
    }

    .dot.yellow {
        background: #facc15;
    }

    .dot.green {
        background: #34d399;
    }

    .terminal-window-title {
        margin-left: 0.4rem;
        color: #8ea4b8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }

    .terminal-command-line {
        position: relative;
        z-index: 2;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: #dbeafe;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(125, 211, 252, 0.18);
        background: rgba(2, 6, 23, 0.62);
        width: fit-content;
        max-width: 100%;
        margin-bottom: 1.2rem;
        box-shadow: 0 14px 40px rgba(0,0,0,0.20);
    }

    .terminal-prompt {
        color: #34d399;
        font-weight: 700;
    }

    .terminal-symbol {
        color: #94a3b8;
    }

    .terminal-path {
        color: #67e8f9;
    }

    .terminal-dollar {
        color: #facc15;
        font-weight: 800;
        margin-left: 0.3rem;
    }

    .terminal-command {
        color: #e0f2fe;
    }

    .terminal-hero h1 {
        position: relative;
        z-index: 2;
        margin: 0;
        font-size: clamp(2.4rem, 5.2vw, 5.4rem);
        line-height: 0.94;
        font-weight: 900;
        letter-spacing: -0.075em;
        color: #f8fafc;
        text-shadow:
            0 0 24px rgba(34, 211, 238, 0.18),
            0 0 60px rgba(167, 139, 250, 0.16);
    }

    .terminal-hero p {
        position: relative;
        z-index: 2;
        max-width: 840px;
        margin-top: 1.1rem;
        color: #a8c5da;
        font-size: 1.08rem;
        line-height: 1.7;
    }

    .terminal-badge-row {
        position: relative;
        z-index: 2;
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.5rem;
    }

    .terminal-badge-row span {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #cffafe;
        padding: 0.55rem 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(34, 211, 238, 0.32);
        background: rgba(34, 211, 238, 0.08);
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.08);
    }

    .terminal-kpi {
        min-height: 138px;
        padding: 1.1rem;
        border-radius: 22px;
        border: 1px solid rgba(125, 211, 252, 0.18);
        background:
            linear-gradient(135deg, rgba(8, 18, 34, 0.86), rgba(15, 23, 42, 0.72));
        box-shadow:
            0 20px 50px rgba(0, 0, 0, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
        margin-bottom: 1rem;
    }

    .terminal-kpi.cyan {
        border-color: rgba(34, 211, 238, 0.30);
    }

    .terminal-kpi.green {
        border-color: rgba(52, 211, 153, 0.32);
    }

    .terminal-kpi.violet {
        border-color: rgba(167, 139, 250, 0.34);
    }

    .terminal-kpi.red {
        border-color: rgba(251, 113, 133, 0.36);
    }

    .terminal-kpi-label {
        color: #8ea4b8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
    }

    .terminal-kpi-value {
        margin-top: 0.55rem;
        color: #f8fafc;
        font-size: 1.72rem;
        line-height: 1.05;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .terminal-kpi-detail {
        margin-top: 0.6rem;
        color: #8ea4b8;
        font-size: 0.86rem;
        line-height: 1.45;
    }

    .terminal-section-title {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin: 1.2rem 0 0.75rem 0;
        padding: 0.95rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(125, 211, 252, 0.16);
        background: rgba(8, 18, 34, 0.58);
    }

    .terminal-section-title span {
        color: #e0f2fe;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 900;
        letter-spacing: 0.08em;
        font-size: 0.9rem;
    }

    .terminal-section-title small {
        color: #8ea4b8;
        font-size: 0.85rem;
    }

    .chat-title {
        margin-top: 1.4rem;
    }

    .quick-command-card {
        min-height: 132px;
        padding: 1rem;
        margin-bottom: 0.65rem;
        border-radius: 20px;
        border: 1px solid rgba(125, 211, 252, 0.16);
        background:
            linear-gradient(135deg, rgba(2, 6, 23, 0.72), rgba(15, 23, 42, 0.70));
        box-shadow: 0 18px 50px rgba(0,0,0,0.22);
    }

    .quick-command-label {
        color: #67e8f9;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.11em;
        margin-bottom: 0.7rem;
    }

    .quick-command-text {
        color: #dbeafe;
        font-size: 0.93rem;
        line-height: 1.45;
    }

    div[data-testid="column"] .stButton > button {
        background:
            linear-gradient(135deg, rgba(34, 211, 238, 0.14), rgba(167, 139, 250, 0.14));
        color: #e0f2fe;
        border: 1px solid rgba(34, 211, 238, 0.28);
        border-radius: 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        transition: all 0.18s ease-in-out;
    }

    div[data-testid="column"] .stButton > button:hover {
        border-color: rgba(34, 211, 238, 0.78);
        box-shadow: 0 0 28px rgba(34, 211, 238, 0.18);
        transform: translateY(-1px);
    }

    .side-terminal {
        padding: 1rem;
        border-radius: 20px;
        border: 1px solid rgba(34, 211, 238, 0.22);
        background:
            linear-gradient(135deg, rgba(2, 6, 23, 0.90), rgba(15, 23, 42, 0.78));
        margin-bottom: 1rem;
        box-shadow: 0 18px 44px rgba(0,0,0,0.28);
    }

    .side-terminal-dotline {
        display: flex;
        gap: 0.45rem;
        margin-bottom: 1rem;
    }

    .side-terminal-title {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 900;
        color: #e0f2fe;
        letter-spacing: 0.06em;
        font-size: 0.92rem;
    }

    .side-terminal-subtitle {
        color: #8ea4b8;
        margin-top: 0.25rem;
        font-size: 0.82rem;
    }

    .sidebar-status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.7rem;
        padding: 0.72rem 0;
        border-bottom: 1px solid rgba(125, 211, 252, 0.10);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }

    .sidebar-status-row span {
        color: #8ea4b8;
    }

    .sidebar-status-row strong {
        color: #e0f2fe;
        text-align: right;
        font-size: 0.75rem;
    }

    .status-ok {
        color: #34d399 !important;
    }

    .status-ko {
        color: #fb7185 !important;
    }

    .trace-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.7rem;
        margin-bottom: 1rem;
    }

    .trace-card {
        padding: 0.85rem;
        border-radius: 14px;
        border: 1px solid rgba(125, 211, 252, 0.16);
        background: rgba(2, 6, 23, 0.62);
    }

    .trace-label {
        color: #67e8f9;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }

    .trace-value {
        color: #dbeafe;
        font-size: 0.82rem;
        line-height: 1.35;
    }

    @media (max-width: 900px) {
        .terminal-hero {
            padding: 1.4rem;
        }

        .terminal-command-line {
            font-size: 0.78rem;
            width: 100%;
        }

        .terminal-section-title {
            display: block;
        }

        .terminal-section-title small {
            display: block;
            margin-top: 0.35rem;
        }

        .trace-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
"""


DOCUMENTS_FOLDER = "documenti"
# L'indice ChromaDB è tenuto FUORI dalla cartella di progetto quando questa è in
# ~/Desktop (sincronizzata da iCloud): durante le scritture iCloud può bloccare il
# file SQLite ("attempt to write a readonly database") e corrompere l'indice. La
# posizione predefinita è in ~/Library/Application Support (non sincronizzata).
# Sovrascrivibile con la variabile d'ambiente UNILAW_CHROMA_DIR.
CHROMA_PERSIST_DIRECTORY = os.getenv(
    "UNILAW_CHROMA_DIR",
    os.path.join(
        os.path.expanduser("~"),
        "Library",
        "Application Support",
        "UniLawAgent",
        "chroma_db",
    ),
)
INDEX_MANIFEST_FILE = "index_manifest.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_MODEL_NAME = "llama3.1:8b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_CTX = 4096

DEFAULT_K_RETRIEVAL = 12
MAX_CONTEXT_DOCUMENTS = 5

# FASE 4 — reranker neurale (cross-encoder multilingua) OPZIONALE.
# Disattivato di default: si abilita via env (UNILAW_RERANKER=1) o dal toggle in
# sidebar. Se il modello non è disponibile, la pipeline torna al reranking
# euristico. Costi indicativi su CPU: ~77 s di caricamento una tantum (lazy),
# ~4 ms per coppia query-documento, ~800 MB RAM di picco, ~458 MB su disco.
RERANKER_ENABLED = os.getenv("UNILAW_RERANKER", "0").strip() in {"1", "true", "True"}
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_TOP_N = 15

# FASE 5 — evidence selection + verifica delle citazioni.
# Evidence selection: al modello vengono passati passaggi più brevi e mirati
# (le frasi più pertinenti alla domanda), con un minimo garantito per non perdere
# informazione. Verifica citazioni: si rimuovono i riferimenti [F#] inventati e si
# controlla che le frasi che citano abbiano riscontro lessicale nella fonte; sotto
# soglia si applica la politica "reduce" (confidenza più bassa + nota), non il blocco.
# Regole deterministiche. Due livelli, in base a quanto è "rischioso" lasciare
# decidere al modello:
#  - DETERMINISTIC_RULES_ENABLED (master): se 0 (UNILAW_DETERMINISTIC=0) nessun
#    template, ogni domanda passa al RAG generativo puro (utile per la valutazione).
#  - PROSE_TEMPLATES_ENABLED: i 5 template "di prosa" (OFA, tesi, Erasmus, borsa,
#    accesso L-19). L'esperimento (docs/esperimenti_rag.md, ESP-07) li ha mostrati
#    RIDONDANTI col RAG generativo, quindi sono DISATTIVATI di default. Resta sempre
#    attivo (quando il master è on) solo il guard numerico sulle soglie TOLC-I, dove
#    l'esattezza del verdetto e la fonte canonica sono critiche.
DETERMINISTIC_RULES_ENABLED = os.getenv("UNILAW_DETERMINISTIC", "1").strip() in {"1", "true", "True"}
PROSE_TEMPLATES_ENABLED = os.getenv("UNILAW_PROSE_TEMPLATES", "0").strip() in {"1", "true", "True"}

EVIDENCE_SELECTION_ENABLED = True
EVIDENCE_MAX_SENTENCES = 6
EVIDENCE_MIN_SENTENCES = 3
EVIDENCE_MAX_CHARS = 700
CITATION_GROUNDING_ENABLED = True
CITATION_GROUNDING_MIN_RATIO = 0.5

# Ciclo 2 — FASE 12 — grounding semantico delle citazioni OPZIONALE (opt-in).
# Il grounding lessicale (overlap di token) boccia le frasi corrette ma PARAFRASATE,
# abbassando ingiustamente la confidenza. Quando questo flag è attivo, le frasi che
# non superano la soglia lessicale ricevono un secondo controllo per similarità di
# embedding (frase citante ↔ frasi della fonte citata), come rete di recupero che si
# AGGIUNGE al lessicale (non lo sostituisce). Riusa il modello di embedding già
# caricato per il retrieval (nessuna nuova dipendenza). DISATTIVATO di default: con il
# flag spento `grounding_report` resta byte-identico al solo lessicale (nessun costo,
# nessun embedder costruito).
# La soglia del coseno è stata CALIBRATA su una piccola sonda di parafrasi reali
# (modello `paraphrase-multilingual-MiniLM-L12-v2`): le coppie parafrasi hanno dato
# similarità 0,19–0,61 e le coppie estranee ≤0,13; il punto di massimo margine è la
# mediana fra le due classi (≈0,16). La separazione è però sottile e la similarità di
# frase di questo modello sulle parafrasi normative è debole: per questo il grounding
# semantico resta DISATTIVATO di default (rischio alto dichiarato in roadmap), in
# attesa di validazione su un insieme più ampio (sinergia con Ciclo 2 — FASE 4/13).
CITATION_GROUNDING_SEMANTIC_ENABLED = os.getenv("UNILAW_SEMANTIC_GROUNDING", "0").strip() in {"1", "true", "True"}
CITATION_GROUNDING_SEMANTIC_MIN_SIMILARITY = 0.16

# FASE 6 — astensione affidabile.
# Soglia di "retrieval strength" (quota di token della domanda coperti dalla
# migliore fonte) sotto la quale un'astensione del modello è classificata come
# "fuori dominio" anziché "fonte presente ma insufficiente". Calibrata sui casi
# reali: q19 (capitale Francia)=0,33 fuori dominio; q17/q18/q14=0,40–0,67 in dominio
# ma risposta assente.
ABSTENTION_OOD_MAX_STRENGTH = 0.37

# Ciclo 2 — FASE 13 — retrieval strength SEMANTICA per l'astensione (opt-in).
# La distinzione fuori_dominio vs evidenza_insufficiente usa, di default, la
# `retrieval_strength` LESSICALE (overlap di token, soglia ABSTENTION_OOD_MAX_STRENGTH):
# penalizza le parafrasi, che non condividono token con la fonte pur essendo pertinenti.
# Con questa opzione attiva la forza è misurata per SIMILARITÀ DI EMBEDDING query↔fonte
# (più robusta verso le parafrasi), con una soglia RICALIBRATA dai dati sullo stesso
# split held-out della FASE 6 (harness `eval/abstention_threshold_validation.py`, modalità
# semantica). Riusa l'embedder del vector store (nessuna nuova dipendenza/modello).
# DISATTIVATO di default: con l'opzione spenta l'embedder non viene nemmeno recuperato e
# la classificazione resta byte-identica a quella lessicale. La separazione semantica fra
# le due cause è risultata sottile (similarità di frase del MiniLM moderata anche su query
# fuori dominio): come per il reranker (FASE 4), l'intent semantico (FASE 11) e il
# grounding semantico (FASE 12), la funzione è implementata ma resta OFF in attesa di
# validazione su un insieme più ampio (rischio alto dichiarato in roadmap).
ABSTENTION_SEMANTIC_STRENGTH_ENABLED = os.getenv("UNILAW_SEMANTIC_ABSTENTION", "0").strip() in {"1", "true", "True"}
# Soglia semantica ricalibrata dai dati (calibrazione a massimo margine sui soli negativi
# storici q17/q18/q19 → ≈0,5286; cfr. report abstention_threshold_validation, sezione
# semantica): q19 (fuori dominio) ha forza semantica 0,46, q17/q18 0,60/0,67 → punto medio
# 0,53. Validata 3/3 sugli held-out q35/q36/q37 (mai visti). NB: in valore assoluto le
# forze semantiche fuori dominio restano alte (0,32–0,46: il MiniLM dà similarità moderata
# anche a query off-topic), quindi la banda è più compressa di quella lessicale.
ABSTENTION_OOD_SEMANTIC_MAX_STRENGTH = 0.53

# Ciclo 2 — FASE 11 — intent detection semantica OPZIONALE (opt-in).
# Affianca il riconoscimento a parole chiave di `intent.py` con un classificatore
# per similarità di embedding (modulo `semantic_intent.py`), che riusa il modello
# di embedding già caricato per il retrieval. Riempie SOLO le caselle (corso/
# argomento) che le keyword lasciano vuote, senza mai sovrascrivere un
# riconoscimento a keyword; i corsi fuori dominio restano gestiti dalle keyword.
# DISATTIVATO di default: il comportamento predefinito resta quello a keyword e il
# classificatore non viene nemmeno costruito (nessun costo). Le soglie di similarità
# del coseno sono PROVVISORIE, da validare su parafrasi prima di abilitarlo di default
# (rischio medio-alto dichiarato in roadmap).
SEMANTIC_INTENT_ENABLED = os.getenv("UNILAW_SEMANTIC_INTENT", "0").strip() in {"1", "true", "True"}
SEMANTIC_INTENT_COURSE_MIN_SIMILARITY = 0.5
SEMANTIC_INTENT_TOPIC_MIN_SIMILARITY = 0.45

# Ciclo 2 — FASE 14 — mitigazione della falsa astensione su regolamento generale (q14).
# Le regole di consultabilità/deposito/embargo della tesi vivono in un regolamento
# GENERALE di Ateneo (`regolamento-tesi-2023.pdf` → course_tag "generale"), non in un
# documento specifico del corso. Su una domanda di consultabilità che nomina un corso
# preciso (q14) il modello 8B tende ad astenersi cercando un dettaglio "per quel corso"
# che il regolamento generale, per natura, non riporta — pur contenendo la regola.
# Quando questo flag è attivo e fra le fonti recuperate è presente un regolamento generale
# sulla tesi, il profilo di risposta riceve un hint che AUTORIZZA esplicitamente l'uso
# della regola generale (cfr. `agent.has_general_tesi_regulation` e `_build_answer_profile`).
# A differenza delle reti semantiche opt-in (FASE 11–13), questa è una correzione mirata e
# a basso rischio (gated sulla situazione di retrieval reale, niente nuove dipendenze):
# resta quindi ATTIVA di default. Il toggle serve all'A/B riproducibile (flag eval
# `--no-general-tesi-hint`): con l'opzione spenta il profilo torna byte-identico a prima.
GENERAL_TESI_HINT_ENABLED = os.getenv("UNILAW_GENERAL_TESI_HINT", "1").strip() in {"1", "true", "True"}


# Ciclo 2 — FASE 8: le fasce numeriche TOLC-I (soglie 9/16) NON sono più ripetute in
# questa guida. Vivono come unica fonte di verità in `knowledge.py` (con provenienza
# sul PDF) e sono applicate dal guard numerico deterministico (sempre attivo).
#
# Ciclo 2 — FASE 9 (RISULTATO NEGATIVO / trade-off): si è VALUTATO di rimuovere anche le
# cinque sezioni "Regole specifiche per <corso/topic>" qui sotto, perché l'instradamento
# *dei documenti* è già fatto, sui metadata, dal reranker euristico e dal filtro per corso
# (`reranking.py`: `rerank_documents` + `filter_documents_by_course`) e le regole di
# contenuto per topic sono emesse dal profilo dinamico `_build_answer_profile`. L'A/B
# nella stessa sessione (config predefinita, `llama3.1:8b`, `temperature=0`, 40 domande,
# within-session σ=0) ha però MISURATO una regressione: il behavior scende da 0,90 (con
# routing) a 0,875 (senza + clausola di chiarimento) / 0,85 (rimozione pura). course/topic/
# retrieval restano invarianti (1,0) — il routing dei documenti È ridondante — ma la prosa
# faceva anche da *framing di commit* di cui il modello 8B si avvale: senza, aumenta le
# false astensioni su domande answerable borderline (q28/q30/q31 borsa/economia) e compare
# over-answering su q18. Come per il reranker neurale (Ciclo 1 — FASE 4), si riporta il
# risultato negativo e si MANTIENE il routing nel prompt. Dettagli: ESP-09 in
# `docs/esperimenti_rag.md` e changelog Ciclo 2 — FASE 9.
ANSWER_STYLE_GUIDE = """
Regole generali:
- Rispondi in italiano.
- Rispondi solo in base al contesto fornito.
- Se il contesto non basta, scrivi esattamente: "Non lo so in base ai documenti disponibili."
- Non inventare dettagli, scadenze, importi, soglie o requisiti non presenti nel contesto.
- Non usare tono promozionale.
- Quando fai un'affermazione sostanziale, cita uno o più riferimenti nel formato [F1], [F2], ecc.
- Non citare fonti che non sono presenti nel contesto.
- Non citare fonti non pertinenti all'argomento della domanda.
- Evita risposte inutilmente lunghe: sii completo ma operativo.

Regole di struttura:
- Se la domanda è puntuale, inizia con "Risposta breve:".
- Se la domanda richiede istruzioni pratiche, aggiungi una sezione "Cosa fare:".
- Se la domanda riguarda requisiti, soglie, punteggi, CFU, importi o casistiche, usa una tabella Markdown quando utile.
- Quando produci una tabella, usa obbligatoriamente il formato Markdown con pipe `|` e riga separatrice `|---|---|`.
- Se la domanda riguarda una procedura, organizza la risposta in: "Risposta breve", "Dettaglio", "Cosa fare", "Fonti".
- Se la domanda riguarda un elenco di documenti o adempimenti, usa punti elenco.

Regole specifiche per accesso/TOLC/OFA:
- Per domande su TOLC-I, OFA, immatricolazione e accesso a Informatica L-31, usa prioritariamente il documento "regolamento-di-accesso-informatical-31-.pdf".
- Per domande su TOLC-I, OFA, immatricolazione e accesso, non usare fonti su borse di studio, Erasmus o tesi se sono disponibili fonti di accesso più pertinenti.

Regole specifiche per accesso/ammissione Scienze dell'Educazione L-19:
- Per domande su immatricolazione, accesso, ammissione o prova di ammissione a Scienze dell'Educazione L-19, privilegia il documento di immatricolazione L-19 e il regolamento L-19.
- "Prova di ammissione" non significa "prova finale": non rispondere parlando di tesi, elaborato finale o seduta di laurea se l'utente chiede l'ammissione al corso.
- Se il contesto contiene durata della prova, numero di quesiti, argomenti, criteri di valutazione o criteri di parità, riportali nella risposta.
- Non usare il regolamento di accesso di Informatica L-31 per rispondere a domande su Scienze dell'Educazione L-19.
- Se il contesto non contiene una soglia minima o un TOLC, non inventare un TOLC o una soglia minima.

Regole specifiche per prova finale:
- Per domande su prova finale, tesi, elaborato o seduta di laurea, privilegia regolamenti della prova finale e guide operative tesi.
- Distingui, quando possibile, tra regole accademiche e procedura amministrativa online.

Regole specifiche per Erasmus:
- Per domande su Erasmus o mobilità internazionale, privilegia bandi Erasmus e documenti di mobilità.
- Non usare regolamenti di accesso o prova finale per rispondere a domande Erasmus.

Regole specifiche per piano di studi:
- Per domande su piano di studi, CFU, insegnamenti e corsi a scelta, privilegia piani di studio e regolamenti del corso.
- Se la domanda non specifica il corso, chiedi chiarimento invece di usare documenti casuali.
"""


qa_template = """
Sei UniLaw Agent, un assistente universitario specializzato nella consultazione di documenti accademici ufficiali.

Devi rispondere esclusivamente in base al CONTESTO qui sotto.

{style_guide}

TIPO DI RISPOSTA CONSIGLIATO:
{answer_profile}

CONTESTO:
{context}

DOMANDA:
{question}

Risposta:
"""


QA_PROMPT = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question", "style_guide", "answer_profile"],
)
