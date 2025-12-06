import streamlit as st
import os
import glob
import time
import logging

# Nascondo la GPU a PyTorch perché la 1070 non è compatibile
# (così tutto gira in CPU ed evitiamo errori CUDA con torch)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Disattivo la telemetria anonima di Chroma / client
# (per evitare tentativi di invio analytics e relativi warning)
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# --- ZITTIAMO GLI AVVISI DEI PDF ---
# Questo nasconde i messaggi "Ignoring wrong pointing object" di pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# --- IMPORT STANDARD PER IL RAG ---
import langchain
from langchain_community.cache import RedisCache
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate   # Prompt per il QA
import redis

# --- IMPORT PER L'AGENTE ---
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool


# --- CONFIGURAZIONE CACHE (Con Fallback) ---
# Se la versione di LangChain è nuova, usa langchain.globals.set_llm_cache
# Se è vecchia, usiamo langchain.llm_cache = ...
try:
    from langchain.globals import set_llm_cache
except ImportError:
    def set_llm_cache(cache):
        langchain.llm_cache = cache


# --- PROMPT per il QA sui PDF ---
# Questo prompt viene usato dal RAG manuale per forzare:
# - uso solo dei documenti
# - zero invenzioni
qa_template = """
Sei UniLaw Agent, un assistente universitario che risponde SOLO in base ai documenti forniti.

REGOLE:
- Usa esclusivamente le informazioni nel CONTENUTO riportato sotto.
- Quando rispondi, NON aggiungere interpretazioni personali, rimani aderente al testo.
- Se le informazioni non sono presenti o non sono sufficienti per rispondere con sicurezza,
  rispondi esattamente: "Non lo so in base ai documenti disponibili."
- Non inventare regole, date, importi o procedure.
- Se la domanda è generica, limita la risposta a ciò che è chiaramente scritto nei documenti.

CONTENUTO DEI DOCUMENTI:
{context}

DOMANDA DELL’UTENTE:
{question}

Risposta (chiara, strutturata in punti, e aderente al testo dei documenti):
"""

QA_PROMPT = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"],
)


# --- TENTATIVO DI CONNESSIONE A REDIS ---
# L'Agente può usare Redis come cache LLM (memoria veloce).
try:
    # Cerchiamo il server Redis locale
    r = redis.Redis(host="localhost", port=6379, db=0)
    # Ping! C'è nessuno?
    if r.ping():
        # Se risponde, diciamo a LangChain: "Usa questo per ricordare tutto!"
        set_llm_cache(RedisCache(redis_=r))
        print("✅ Redis Cache attivata! L'Agente ora ha la memoria a breve termine.")
    else:
        print("⚠️ Redis non risponde. L'Agente funzionerà ma sarà smemorato.")
except Exception as e:
    print(f"⚠️ Cache disattivata: {e}")


# --- CONFIGURAZIONE VISIVA DELLA PAGINA ---
st.set_page_config(page_title="UniLaw Agent", page_icon="🤖", layout="wide")

# --- CSS (TRUCCHI GRAFICI) ---
st.markdown(
    """
<style>
    /* Nasconde la scritta 'Running' fastidiosa in alto */
    [data-testid="stStatusWidget"] { visibility: hidden; }
    /* Arrotonda i messaggi della chat per renderli moderni */
    .stChatMessage { border-radius: 15px; }
    /* Titolo rosso IBM style */
    .main-header { font-size: 3rem; color: #b71c1c; text-align: center; font-weight: 800; font-family: 'Helvetica Neue', sans-serif; margin-top: -20px; }
    /* Sottotitolo elegante */
    .sub-header { font-size: 1.3rem; color: white; text-align: center; margin-bottom: 30px; font-style: italic; }
</style>
""",
    unsafe_allow_html=True,
)


# --- 1. IL CERVELLO DOCUMENTALE (RAG) ---
# Questa funzione prepara la "biblioteca" dell'agente: legge i PDF, li spezza,
# crea gli embeddings e li salva in Chroma.
@st.cache_resource(show_spinner=False)
def inizializza_conoscenza():
    folder_path = "documenti"

    # Controllo: cartella esiste?
    if not os.path.exists(folder_path):
        return None, "⚠️ Cartella 'documenti' non trovata."
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        return None, "⚠️ Metti dei PDF nella cartella 'documenti'!"

    all_texts = []

    # Ciclo su ogni PDF trovato
    for pdf_path in pdf_files:
        print(f"👉 Sto analizzando: {os.path.basename(pdf_path)}...")
        try:
            # Carichiamo il PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            # Lo spezzettiamo in chunk da 700 caratteri
            # (chunk più piccoli = contesto più mirato)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=200,
            )
            all_texts.extend(text_splitter.split_documents(documents))
        except Exception as e:
            print(f"Errore leggendo {pdf_path}: {e}")

    # Creiamo gli "Embeddings" (parole -> vettori numerici)
    # Modello multilingua, molto adatto anche all'italiano.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    # Salviamo tutto in ChromaDB (database vettoriale locale)
    db = Chroma.from_documents(all_texts, embeddings)
    return db, f"✅ Agente Operativo. Ho letto {len(pdf_files)} documenti."


# --- 2. GLI STRUMENTI DELL'AGENTE (TOOLS) ---

# STRUMENTO A: La Calcolatrice
# L'Agente capirà da solo QUANDO usare questa funzione.
@tool
def calcolatrice_tasse(espressione: str):
    """Utile SOLO quando devi fare calcoli matematici precisi (somme, percentuali, tasse).
    Input: una espressione matematica scritta come stringa (es: '20000 * 0.05').
    Non usare questo strumento per cercare testo."""
    try:
        return str(eval(espressione))
    except Exception:
        return "Errore nel calcolo."


# --- 3. COSTRUZIONE DELL'AGENTE ---
def get_agent_executor(vector_db):
    """
    Costruisce l'agente ReAct che:
    - usa ChatOllama (Llama 3.1) come LLM
    - usa un retriever su Chroma (RAG manuale)
    - usa anche il tool calcolatrice_tasse
    """

    # 1. Modello Llama 3.1 servito da Ollama
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.01,  # modello "serio", poco creativo (meno allucinazioni)
        num_ctx=4096,      # contesto massimo
    )

    # 2. Retriever su ChromaDB
    #    Uso MMR per avere passaggi diversi tra loro (meno ridondanza)
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )

    # 3. Funzione RAG manuale: recupera documenti, costruisce il prompt, chiama il modello
    def rag_qa(question: str) -> str:
        """
        Esegue RAG manuale:
        - recupera i documenti rilevanti
        - costruisce il contesto
        - chiama il modello con QA_PROMPT
        - restituisce risposta + estratti + fonti
        """
        docs = retriever.get_relevant_documents(question)

        # Se non trova niente di sensato, risponde in modo onesto
        if not docs:
            return "Non lo so in base ai documenti disponibili."

        # Costruiamo il contesto per il prompt
        context_chunks = []
        snippets = []
        refs = []

        for i, doc in enumerate(docs, start=1):
            text = doc.page_content.strip().replace("\n", " ")
            # Estratto breve per mostrare cosa ha letto
            if len(text) > 350:
                text_short = text[:350] + "..."
            else:
                text_short = text

            source_path = doc.metadata.get("source", "")
            filename = os.path.basename(source_path)
            page = doc.metadata.get("page", None)

            context_chunks.append(f"[{i}] {text}")

            if page is not None:
                refs.append(f"- {filename} (pag. {page + 1})")
                snippets.append(f"📄 *{filename}, pag. {page + 1}*: {text_short}")
            else:
                refs.append(f"- {filename}")
                snippets.append(f"📄 *{filename}*: {text_short}")

        # Contesto finale passato al prompt
        context = "\n\n".join(context_chunks)

        # Prompt finale per il modello (LLM)
        full_prompt = QA_PROMPT.format(context=context, question=question)

        # Chiamata al modello (ChatOllama come chat model)
        answer = llm.predict(full_prompt)

        # Aggiungiamo gli estratti utilizzati
        if snippets:
            answer += "\n\n---\nEstratti dai documenti utilizzati:\n"
            answer += "\n\n".join(snippets)

        # Aggiungiamo la lista delle fonti
        if refs:
            answer += "\n\nFonti nei documenti:\n" + "\n".join(refs)

        return answer

    # 4. Tool RAG per l'agente
    rag_tool = Tool(
        name="KnowledgeBase_Universitaria",
        func=rag_qa,
        description=(
            "Utile per rispondere a domande teoriche su regolamenti, scadenze, "
            "procedure, tasse e leggi universitarie. "
            "Deve essere usato SEMPRE prima di rispondere su questi argomenti. "
            "Se dalle fonti non emerge una risposta chiara, l'assistente deve dire "
            "\"Non lo so in base ai documenti disponibili.\""
        ),
    )

    # 5. Cassetta degli attrezzi: RAG + calcolatrice
    tools = [rag_tool, calcolatrice_tasse]

    # 6. Messaggio di sistema che vincola il comportamento
    system_message = """
Sei UniLaw Agent, assistente dell'Università. 

REGOLE:
1. Usa SEMPRE lo strumento KnowledgeBase_Universitaria per rispondere a domande
   su regolamenti, piani di studio, tesi, borse di studio, esami, immatricolazioni, Erasmus, ecc. 
2. Rispondi SOLO in base alle informazioni presenti nei documenti. 
3. Se i documenti non contengono una risposta chiara, scrivi esattamente:
   "Non lo so in base ai documenti disponibili." 
4. Non inventare mai regole, cifre, date o procedure. 
5. Quando il tool ti restituisce la risposta con le fonti (file e pagina),
   non modificarla e non aggiungere riferimenti che non esistono nei PDF. 
"""

    # 7. Costruzione dell'agente ReAct
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,             # Mostra il ragionamento nel terminale
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_message},
    )

    return agent


# --- 4. INTERFACCIA UTENTE STREAMLIT ---

# Carichiamo il database vettoriale all'avvio
vector_db, msg = inizializza_conoscenza()

# Sidebar laterale
with st.sidebar:
    st.title("⚙️ Pannello Agente")

    # Mostriamo lo stato del caricamento
    st.success(msg if vector_db else "Errore DB")

    # Tasto per pulire la chat
    if st.button("🗑️ Svuota Chat"):
        st.session_state.messages = []
        st.rerun()

# Titoli principali
st.markdown('<div class="main-header">UniLaw Agent 🤖</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Non solo ricerca, ma ragionamento autonomo.</div>',
    unsafe_allow_html=True,
)

# Inizializza la cronologia chat se vuota
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Sono UniLaw Agent. Posso consultare i documenti e usare strumenti di calcolo. "
                "Fammi una domanda su regolamenti, piani di studio, tesi, borse di studio..."
            ),
        }
    ]

# Ristampa tutta la cronologia chat (per non farla sparire a ogni refresh)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. GESTIONE DELL'INPUT UTENTE ---
if prompt := st.chat_input(
    "Es: Quali sono i requisiti formali della tesi per il corso L-31?"
):
    # 1. Aggiungiamo la domanda dell'utente alla chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Risposta dell'Agente
    with st.chat_message("assistant"):
        if not vector_db:
            st.error(
                "⚠️ Attenzione: non trovo il database vettoriale. Controlla la cartella 'documenti'."
            )
        else:
            # Box di stato per mostrare che l'agente sta pensando
            with st.status("🧠 L'Agente sta ragionando...", expanded=True) as status:
                st.write("Analisi della richiesta e scelta dello strumento...")
                try:
                    # Creiamo l'agente al volo (con il vector_db già caricato)
                    agent = get_agent_executor(vector_db)

                    # --- QUI AVVIENE LA MAGIA ---
                    # agent.run fa partire il loop ReAct: Pensiero -> Azione -> Osservazione
                    response = agent.run(prompt)

                    # Se tutto va bene, chiudiamo il box di caricamento
                    status.update(
                        label="✅ Risposta elaborata!",
                        state="complete",
                        expanded=False,
                    )

                    # Mostriamo la risposta finale
                    st.markdown(response)

                    # Salviamo la risposta in cronologia
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    status.update(label="❌ Errore", state="error")
                    st.error(f"Errore Agente: {e}")
