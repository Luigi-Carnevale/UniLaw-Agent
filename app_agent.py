import streamlit as st
import os
import glob
import time
import logging

# --- ZITTIAMO GLI AVVISI DEI PDF ---
# Questo nasconde i messaggi "Ignoring wrong pointing object"
logging.getLogger("pypdf").setLevel(logging.ERROR)

# --- IMPORT STANDARD PER IL RAG (Come prima) ---
import langchain
from langchain_community.cache import RedisCache
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama 
from langchain.chains import RetrievalQA
import redis

# --- 🚀 NUOVI IMPORT FONDAMENTALI PER L'AGENTE 🚀 ---
# Questi sono i pezzi che trasformano il sistema da "lettore" a "pensatore"
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool

# --- CONFIGURAZIONE CACHE (Con Fallback) ---
# Questo blocco serve per rendere il codice robusto.
# Se LangChain è vecchio o nuovo, lui si adatta.
try:
    from langchain.globals import set_llm_cache
except ImportError:
    def set_llm_cache(cache):
        langchain.llm_cache = cache

# --- TENTATIVO DI CONNESSIONE A REDIS ---
# L'Agente ha bisogno di memoria veloce. Qui proviamo a connetterci.
try:
    # Cerchiamo il server Redis locale
    r = redis.Redis(host='localhost', port=6379, db=0)
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
st.markdown("""
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
""", unsafe_allow_html=True)

# --- 1. IL CERVELLO DOCUMENTALE (RAG) ---
# Questa funzione prepara la "biblioteca" dell'agente.
@st.cache_resource(show_spinner=False)
def inizializza_conoscenza():
    folder_path = "documenti"
    # Controlli di sicurezza se la cartella non esiste
    if not os.path.exists(folder_path): return None, "⚠️ Cartella 'documenti' non trovata."
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files: return None, "⚠️ Metti dei PDF nella cartella 'documenti'!"

    all_texts = []
    # Ciclo su ogni PDF trovato
    for pdf_path in pdf_files:
        print(f"👉 Sto analizzando: {os.path.basename(pdf_path)}...")
        try:
            # Carichiamo il PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            # Lo spezzettiamo in chunk da 1000 caratteri
            # (I modelli AI preferiscono piccoli bocconi di testo, non libri interi)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_texts.extend(text_splitter.split_documents(documents))
        except Exception as e: print(f"Errore leggendo {pdf_path}: {e}")

    # Creiamo gli "Embeddings" (la traduzione da parole a numeri vettoriali)
    # Usiamo un modello piccolo e veloce di HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Salviamo tutto in ChromaDB (il database vettoriale)
    db = Chroma.from_documents(all_texts, embeddings)
    return db, f"✅ Agente Operativo. Ho letto {len(pdf_files)} documenti."

# --- 2. GLI STRUMENTI DELL'AGENTE (TOOLS) ---

# STRUMENTO A: La Calcolatrice
# L'Agente capirà da solo QUANDO usare questa funzione.
# Il testo tra le virgolette ("Utile per...") è il "Libretto di Istruzioni" per l'AI.
@tool
def calcolatrice_tasse(espressione: str):
    """Utile SOLO quando devi fare calcoli matematici precisi (somme, percentuali, tasse). 
    Input: una espressione matematica scritta come stringa (es: '20000 * 0.05').
    Non usare questo strumento per cercare testo."""
    try:
        # Esegue il calcolo matematico
        return str(eval(espressione))
    except:
        return "Errore nel calcolo."

# --- 3. COSTRUZIONE DELL'AGENTE ---
def get_agent_executor(vector_db):

    # Configurazione del modello Llama 3.1 servito da Ollama
    llm = ChatOllama(
        model="llama3.1:8b",       
        temperature = 0.01,     # modello "serio", non creativo (senza allucinazioni)
        num_ctx = 4096          # contesto massimo (adattare poi ai limiti corretti)
    )

    # CREIAMO LO STRUMENTO B: Il RAG (Ricerca Documentale)
    # solo un "attrezzo" nella cassetta degli attrezzi.
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3})
    )

    # Definiamo il Tool per la ricerca
    rag_tool = Tool(
        name="KnowledgeBase_Universitaria",
        func=rag_chain.run,
        description="Utile per rispondere a domande teoriche su regolamenti, scadenze, procedure, tasse e leggi. Usa sempre questo strumento prima di rispondere a domande generiche."
    )

    # --- IL MOMENTO MAGICO: DIAMO GLI STRUMENTI ALL'AGENTE ---
    tools = [rag_tool, calcolatrice_tasse]

    # Inizializziamo l'Agente ReAct (Reasoning + Acting)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        # ZERO_SHOT_REACT_DESCRIPTION è il "cervello" che permette all'AI di decidere:
        # "Devo cercare nel PDF o devo usare la calcolatrice?"
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, # IMPORTANTE: Fa vedere i "pensieri" nel terminale nero
        handle_parsing_errors=True # Se l'agente si confonde, riprova invece di crashare
    )
    return agent

# --- 4. INTERFACCIA UTENTE STREAMLIT ---
# Carichiamo il database all'avvio
vector_db, msg = inizializza_conoscenza()

# Sidebar laterale per le chiavi
with st.sidebar:
    # LOGO UNISA st.image("https://seeklogo.com/images/U/universita-degli-studi-di-salerno-unisa-logo-145842C77D-seeklogo.com.png", width=200)
    st.title("⚙️ Pannello Agente")
    
    # Mostriamo lo stato del caricamento
    st.success(msg if vector_db else "Errore DB")
    
    # Tasto per pulire la chat
    if st.button("🗑️ Svuota Chat"):
        st.session_state.messages = []
        st.rerun()

# Titoli principali
st.markdown('<div class="main-header">UniLaw Agent 🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Non solo ricerca, ma ragionamento autonomo.</div>', unsafe_allow_html=True)

# Inizializza la cronologia chat se vuota
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Sono un Agente AI. Posso consultare i documenti e usare strumenti di calcolo. Mettimi alla prova."}]

# Ristampa tutta la cronologia chat (per non farla sparire a ogni refresh)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. GESTIONE DELL'INPUT UTENTE ---
if prompt := st.chat_input("Es: Ho un reddito di 20.000€ e la tassa è il 5%, quanto pago?"):
    # 1. Aggiungiamo la domanda dell'utente alla chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Risposta dell'Agente
    with st.chat_message("assistant"):
        if not vector_db:
            st.error("⚠️ Attenzione: non trovo il database vettoriale. Controlla la cartella 'documenti'.")
        else:
            # Creiamo un box espandibile per far vedere all'utente che l'AI sta pensando
            with st.status("🧠 L'Agente sta ragionando...", expanded=True) as status:
                st.write("Analisi della richiesta e scelta dello strumento...")
                try:
                    # Creiamo l'agente al volo 
                    agent = get_agent_executor(vector_db)
                    
                    # --- QUI AVVIENE LA MAGIA ---
                    # agent.run fa partire il loop: Pensiero -> Azione -> Osservazione
                    response = agent.run(prompt)
                    
                    # Se tutto va bene, chiudiamo il box di caricamento
                    status.update(label="✅ Risposta elaborata!", state="complete", expanded=False)
                    
                    # Mostriamo la risposta finale
                    st.markdown(response)
                    
                    # Salviamo la risposta in cronologia
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    status.update(label="❌ Errore", state="error")
                    st.error(f"Errore Agente: {e}")
