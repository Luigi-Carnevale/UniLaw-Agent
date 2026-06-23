"""Tema chiaro 'accademico' per UniLaw Agent (versione in valutazione).

Foglio di stile alternativo a `config.CSS_STYLES` (tema 'terminale'): tono caldo,
istituzionale e leggibile, pensato per studenti. Vive in un file separato per non
toccare la UI attuale: lo usa solo `app_agent_new.py`. Se la nuova interfaccia non
viene approvata, basta eliminare questo file e `app_agent_new.py` — niente è cambiato.
"""

CSS_STYLES_LIGHT = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --ul-paper: #fbf8f2;
        --ul-paper-2: #f5f0e6;
        --ul-card: #ffffff;
        --ul-ink: #222a3f;
        --ul-soft: #5c6478;
        --ul-faint: #8a8270;
        --ul-navy: #283659;
        --ul-navy-ink: #f3ecdb;
        --ul-gold: #9c6f1f;
        --ul-gold-bg: #f8efd8;
        --ul-teal: #1d7a5c;
        --ul-teal-bg: #e3f3ec;
        --ul-amber: #9a6a12;
        --ul-amber-bg: #f8edd5;
        --ul-red: #a8442f;
        --ul-red-bg: #f7e7e2;
        --ul-line: #ece4d4;
        --ul-line-2: #e2d9c7;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(156, 111, 31, 0.06), transparent 36%),
            radial-gradient(circle at 88% 6%, rgba(40, 54, 89, 0.06), transparent 30%),
            var(--ul-paper);
        color: var(--ul-ink);
    }

    .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
    .stMarkdown, [data-testid="stMarkdownContainer"] {
        color: var(--ul-ink);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 5rem;
        max-width: 900px;
        margin: 0 auto;
    }

    /* Pulizia del chrome di Streamlit per un look 'prodotto finito' */
    #MainMenu, [data-testid="stMainMenu"] { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stHeader"] { background: transparent; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: var(--ul-line-2); border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: #cbbf9f; }
    ::-webkit-scrollbar-track { background: transparent; }

    h1, h2, h3, h4 {
        font-family: 'Fraunces', serif !important;
        font-weight: 600 !important;
        color: var(--ul-navy) !important;
        letter-spacing: -0.01em;
    }

    [data-testid="stStatusWidget"] { visibility: hidden; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--ul-paper-2);
        border-right: 1px solid var(--ul-line);
    }

    section[data-testid="stSidebar"] * { color: var(--ul-ink); }

    section[data-testid="stSidebar"] .stButton > button {
        background: var(--ul-card);
        color: var(--ul-navy);
        border: 1px solid var(--ul-line-2);
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
        font-size: 0.84rem;
        font-weight: 600;
        transition: all 0.16s ease;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--ul-gold);
        background: #fffdf8;
        transform: translateY(-1px);
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        background: var(--ul-card);
        border: 1px solid var(--ul-line);
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary {
        color: var(--ul-navy) !important;
        font-weight: 600;
    }

    /* ---------- Chat ---------- */
    .stChatMessage {
        background: var(--ul-card);
        border: 1px solid var(--ul-line);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        box-shadow: 0 6px 20px rgba(40, 40, 60, 0.05);
    }

    /* Bolla utente: il messaggio utente ha l'avatar con testid chatAvatarIcon-user
       (l'assistente usa un avatar custom → chatAvatarIcon-custom). */
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--ul-navy);
        border-color: var(--ul-navy);
        flex-direction: row-reverse;
        margin-left: 14%;
    }

    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) * {
        color: var(--ul-navy-ink) !important;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] { color: var(--ul-ink); }

    /* Barra di input ancorata in basso: la barra full-width E il blocco centrato
       devono essere color carta, altrimenti agli angoli resta lo sfondo scuro. */
    [data-testid="stBottom"] {
        background: var(--ul-paper);
        border-top: 1px solid var(--ul-line);
    }
    [data-testid="stBottom"] > div { background: var(--ul-paper); }
    [data-testid="stBottomBlockContainer"] {
        background: var(--ul-paper);
        max-width: 900px;
        margin: 0 auto;
    }

    textarea, [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] {
        background: var(--ul-card) !important;
        color: var(--ul-ink) !important;
        border: 1px solid var(--ul-line-2) !important;
        border-radius: 14px !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--ul-gold) !important;
        box-shadow: 0 0 0 3px rgba(156, 111, 31, 0.10) !important;
    }

    [data-testid="stChatInput"] button {
        color: var(--ul-navy) !important;
    }

    /* File uploader: la dropzone di default è scura → portala a tema chiaro */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--ul-paper) !important;
        border: 1px dashed var(--ul-line-2) !important;
        border-radius: 12px;
    }
    [data-testid="stFileUploaderDropzone"] * { color: var(--ul-ink) !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--ul-soft) !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--ul-card) !important;
        color: var(--ul-navy) !important;
        border: 1px solid var(--ul-line-2) !important;
        border-radius: 10px;
    }
    [data-testid="stFileUploaderFile"] { color: var(--ul-ink) !important; }

    /* Pulsanti di download (JSON / Markdown del trace) */
    [data-testid="stDownloadButton"] button {
        background: var(--ul-card) !important;
        color: var(--ul-navy) !important;
        border: 1px solid var(--ul-line-2) !important;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.16s ease;
    }
    [data-testid="stDownloadButton"] button:hover {
        border-color: var(--ul-gold) !important;
        background: #fffdf8 !important;
    }

    code {
        color: var(--ul-gold) !important;
        background: var(--ul-gold-bg) !important;
        border: 1px solid #efe2c4;
        border-radius: 6px;
        padding: 0.05rem 0.3rem;
        font-family: 'JetBrains Mono', monospace !important;
    }

    pre {
        background: #fcfaf5 !important;
        border: 1px solid var(--ul-line);
        border-radius: 12px;
    }

    div[data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--ul-line-2);
        background: var(--ul-card);
    }

    div[data-testid="stMarkdownContainer"] th {
        background: var(--ul-paper-2);
        color: var(--ul-navy);
        font-size: 0.84rem;
        text-transform: none;
    }

    div[data-testid="stMarkdownContainer"] td {
        border-top: 1px solid var(--ul-line);
        color: var(--ul-ink);
    }

    /* ---------- Suggerimenti (chip) ---------- */
    div[data-testid="column"] .stButton > button {
        background: var(--ul-card);
        color: var(--ul-navy);
        border: 1px solid var(--ul-line-2);
        border-radius: 999px;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        text-align: left;
        transition: all 0.16s ease;
    }

    div[data-testid="column"] .stButton > button:hover {
        border-color: var(--ul-gold);
        background: #fffdf8;
        transform: translateY(-1px);
    }

    /* ---------- Componenti su misura ---------- */
    .ul-hero {
        padding: 0.4rem 0 0.2rem 0;
        margin-bottom: 0.4rem;
    }

    .ul-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 1rem;
    }

    /* Icona disegnata via CSS (data-URI): evita che Streamlit strippi l'SVG inline. */
    .ul-brand-mark {
        width: 38px; height: 38px;
        border-radius: 10px;
        flex: 0 0 auto;
        background-color: var(--ul-navy);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 21px 21px;
        background-image: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='%23f4e9cf'%20stroke-width='1.8'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M12%203%2021%208H3z'/%3E%3Cpath%20d='M5%2021V10'/%3E%3Cpath%20d='M9.5%2021V10'/%3E%3Cpath%20d='M14.5%2021V10'/%3E%3Cpath%20d='M19%2021V10'/%3E%3Cpath%20d='M3%2021h18'/%3E%3C/svg%3E");
    }

    .ul-brand-name {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.15rem;
        color: var(--ul-navy);
        line-height: 1;
    }

    .ul-brand-sub {
        font-size: 0.74rem;
        color: var(--ul-soft);
        margin-top: 2px;
    }

    .ul-hero h1 {
        font-size: clamp(1.8rem, 3.4vw, 2.6rem) !important;
        line-height: 1.12;
        margin: 0;
    }

    .ul-hero p {
        max-width: 620px;
        margin-top: 0.6rem;
        color: var(--ul-soft);
        font-size: 1.02rem;
        line-height: 1.6;
    }

    .ul-hero .ul-accent { color: var(--ul-gold); font-weight: 600; }

    .ul-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ul-faint);
        margin: 0.4rem 0 0.5rem 0;
    }

    .ul-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
    }
    .ul-pill .ul-dot { width: 7px; height: 7px; border-radius: 50%; }
    .ul-pill.alta  { background: var(--ul-teal-bg);  color: var(--ul-teal); }
    .ul-pill.alta  .ul-dot { background: var(--ul-teal); }
    .ul-pill.media { background: var(--ul-amber-bg); color: var(--ul-amber); }
    .ul-pill.media .ul-dot { background: var(--ul-amber); }
    .ul-pill.bassa { background: var(--ul-red-bg);   color: var(--ul-red); }
    .ul-pill.bassa .ul-dot { background: var(--ul-red); }

    .ul-source-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.55rem;
        margin-top: 0.4rem;
    }

    .ul-source-card {
        display: flex;
        gap: 0.6rem;
        align-items: flex-start;
        border: 1px solid var(--ul-line);
        border-radius: 11px;
        padding: 0.6rem 0.7rem;
        background: var(--ul-paper);
        transition: border-color 0.16s ease, background 0.16s ease;
    }
    .ul-source-card:hover { border-color: var(--ul-gold); background: #fffdf8; }

    .ul-source-ico {
        width: 17px; height: 17px; flex: 0 0 auto; margin-top: 1px;
        background-repeat: no-repeat; background-position: center; background-size: contain;
        background-image: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='%23283659'%20stroke-width='1.8'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M14%203v4a1%201%200%200%200%201%201h4'/%3E%3Cpath%20d='M17%2021H7a2%202%200%200%201-2-2V5a2%202%200%200%201%202-2h7l5%205v11a2%202%200%200%201-2%202z'/%3E%3Cpath%20d='M9%2013h6'/%3E%3Cpath%20d='M9%2017h4'/%3E%3C/svg%3E");
    }
    .ul-source-name { font-size: 0.82rem; font-weight: 600; color: var(--ul-navy); line-height: 1.3; word-break: break-word; }
    .ul-source-page { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--ul-soft); margin-top: 2px; }

    .ul-status-card {
        background: var(--ul-card);
        border: 1px solid var(--ul-line);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.6rem;
    }

    .ul-status-head {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 0.84rem; font-weight: 600;
    }
    .ul-status-head .ul-dot { width: 8px; height: 8px; border-radius: 50%; }
    .ul-status-head.ok  { color: var(--ul-teal); }
    .ul-status-head.ok  .ul-dot { background: var(--ul-teal); }
    .ul-status-head.ko  { color: var(--ul-red); }
    .ul-status-head.ko  .ul-dot { background: var(--ul-red); }

    .ul-status-row {
        display: flex; justify-content: space-between; align-items: center;
        gap: 0.7rem; font-size: 0.8rem; color: var(--ul-soft);
        margin-top: 0.5rem;
    }
    .ul-status-row strong { color: var(--ul-ink); font-weight: 600; }
    .ul-status-row .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }

    .ul-section-label {
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ul-faint);
        margin: 0.4rem 0 0.4rem 0;
    }

    /* Header compatto (in conversazione) */
    .ul-topbar {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.2rem 0 0.9rem 0;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid var(--ul-line);
    }
    .ul-topbar .ul-brand-mark { width: 30px; height: 30px; border-radius: 8px; background-size: 17px 17px; }
    .ul-topbar-name { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1rem; color: var(--ul-navy); }
    .ul-topbar-sub { font-size: 0.78rem; color: var(--ul-soft); margin-left: 0.2rem; }

    /* Riquadro "Come ho risposto" / "Dettagli" */
    .ul-explain { margin-top: 0.2rem; }
    .ul-explain-row {
        display: flex; justify-content: space-between; gap: 1.2rem;
        padding: 0.45rem 0; border-bottom: 1px solid var(--ul-line);
        font-size: 0.84rem;
    }
    .ul-explain-row:last-child { border-bottom: none; }
    .ul-explain-row span { color: var(--ul-soft); flex: 0 0 auto; }
    .ul-explain-row strong { color: var(--ul-ink); font-weight: 600; text-align: right; }
</style>
"""
