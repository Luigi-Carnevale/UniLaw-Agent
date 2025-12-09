# Importiamo la libreria principale per l'interfaccia web: Streamlit.
# Streamlit trasforma script Python in app web interattive senza bisogno di HTML/JS.
import streamlit as st

# Importiamo le funzioni di configurazione (setup_environment) e lo stile CSS (CSS_STYLES)
# dal nostro file 'config.py'. Serve per tenere questo file pulito.
from config import setup_environment, CSS_STYLES

# Importiamo la funzione che crea o carica il database vettoriale dal file 'database.py'.
from database import inizializza_conoscenza

# Importiamo il motore dell'agente (get_cached_agent) e la cache Redis dal file 'agent.py'.
from agent import get_cached_agent, setup_redis_cache

# -----------------------------------------------------------------------------
# 1. CONFIGURAZIONE INIZIALE (Eseguita ad ogni ricaricamento della pagina)
# -----------------------------------------------------------------------------

# Prepariamo l'ambiente (es. disabilitiamo telemetria, settiamo log).
setup_environment()

# Configuriamo le impostazioni base della pagina web (Titolo tab browser, icona, layout largo).
st.set_page_config(page_title="UniLaw Agent", page_icon="🤖", layout="wide")

# Iniettiamo il CSS personalizzato nella pagina per abbellire titoli e nascondere elementi inutili.
# 'unsafe_allow_html=True' permette di inserire codice HTML/CSS grezzo.
st.markdown(CSS_STYLES, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SETUP LOGICA (Backend)
# -----------------------------------------------------------------------------

# Attiviamo la cache Redis per velocizzare le risposte ripetute (se Redis è installato).
setup_redis_cache()

# Inizializziamo la conoscenza (Vector DB).
# Questa funzione è "cached" in database.py, quindi non rifarà tutto il lavoro se il DB esiste già.
# vector_db: è l'oggetto database vero e proprio.
# msg: è il messaggio di stato (es. "Database caricato").
vector_db, msg = inizializza_conoscenza()

# -----------------------------------------------------------------------------
# 3. SIDEBAR (Colonna laterale sinistra)
# -----------------------------------------------------------------------------

with st.sidebar:
    # Titolo della barra laterale.
    st.title("⚙️ Pannello Agente")
    
    # Mostriamo un box verde (successo) o rosso (errore) con lo stato del Database.
    st.success(msg if vector_db else "Errore DB")
    
    # --- PULSANTE 1: SVUOTA CHAT ---
    # Se l'utente clicca questo bottone...
    if st.button("🗑️ Svuota Chat"):
        # ...resettiamo la lista dei messaggi nella memoria di sessione.
        st.session_state.messages = []
        # ...ricarichiamo la pagina per mostrare la chat vuota.
        st.rerun()

    # Linea divisoria orizzontale.
    st.markdown("---")
    
    # --- PULSANTE 2: AGGIORNA DOCUMENTI (Cruciale per il refactoring) ---
    # Questo bottone serve a cancellare il vecchio DB e forzare la creazione dei nuovi "Macro-Chunk".
    if st.button("🔄 Aggiorna Documenti"):
        # Mostriamo uno spinner di caricamento mentre lavora.
        with st.spinner("Cancellazione database in corso..."):
            try:
                # Importiamo shutil qui dentro perché serve solo per questa operazione di pulizia.
                import shutil 
                import os # Serve per controllare se la cartella esiste
                
                # Se la cartella del database esiste fisicamente sul disco...
                if os.path.exists("chroma_db_storage"):
                    # ...la cancelliamo completamente (rimozione ricorsiva).
                    shutil.rmtree("chroma_db_storage")
                
                # Svuotiamo la cache interna di Streamlit (@st.cache_resource).
                # Questo obbliga la funzione 'inizializza_conoscenza' a rieseguirsi da zero al prossimo riavvio.
                st.cache_resource.clear()
                
                # Messaggio di conferma.
                st.success("Database cancellato! Riavvio in corso...")
                
                # Riavvia l'app automaticamente per far partire la re-indicizzazione.
                st.rerun()
            except Exception as e:
                # Se qualcosa va storto (es. file aperti), mostriamo l'errore.
                st.error(f"Errore: {e}")

# -----------------------------------------------------------------------------
# 4. HEADER UI (Intestazione Principale)
# -----------------------------------------------------------------------------

# Usiamo HTML personalizzato (definito in config.py) per il titolo grande rosso.
st.markdown('<div class="main-header">UniLaw Agent 🤖</div>', unsafe_allow_html=True)
# Sottotitolo descrittivo.
st.markdown('<div class="sub-header">Non solo ricerca, ma ragionamento autonomo.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. GESTIONE STATO CHAT (Memoria di Sessione)
# -----------------------------------------------------------------------------

# Streamlit riesegue tutto il codice ad ogni interazione.
# Dobbiamo usare 'st.session_state' per ricordarci la cronologia della chat tra un refresh e l'altro.

# Se la lista 'messages' non esiste ancora (primo avvio)...
if "messages" not in st.session_state:
    # ...la creiamo con un messaggio di benvenuto dell'assistente.
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Sono UniLaw Agent. Posso consultare i documenti e usare strumenti di calcolo."
    }]

# Ridisegniamo TUTTI i messaggi salvati nella cronologia.
# Questo serve perché quando premi invio la pagina si pulisce e si ridisegna.
for m in st.session_state.messages:
    # Crea un contenitore grafico (bubble) per il ruolo specifico (user o assistant).
    with st.chat_message(m["role"]):
        # Scrive il contenuto del messaggio dentro la bubble.
        st.markdown(m["content"])

# -----------------------------------------------------------------------------
# 6. LOOP PRINCIPALE (Input Utente e Risposta)
# -----------------------------------------------------------------------------

# Creiamo la casella di input in basso. Se l'utente scrive e preme invio, 'prompt' conterrà il testo.
if prompt := st.chat_input("Es: Quali sono i requisiti formali della tesi per il corso L-31?"):
    
    # 1. Salviamo subito la domanda dell'utente nella cronologia.
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Mostriamo subito la domanda nell'interfaccia grafica.
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Prepariamo la risposta dell'assistente.
    with st.chat_message("assistant"):
        # Controllo di sicurezza: se il DB non è caricato, fermiamo tutto.
        if not vector_db:
            st.error("⚠️ Database non trovato.")
        else:
            # Creiamo un blocco "status" espandibile che mostra cosa succede "dietro le quinte".
            with st.status("🧠 L'Agente sta ragionando...", expanded=True) as status:
                try:
                    # Recuperiamo l'istanza del motore (Agent).
                    agent = get_cached_agent(vector_db)
                    
                    # ESECUZIONE: Qui chiamiamo la funzione 'run' di agent.py.
                    # È qui che parte il Router, il Retrieval e la generazione GPU.
                    response = agent.run(prompt)
                    
                    # Se tutto va bene, aggiorniamo lo stato a "Completo" e chiudiamo il box.
                    status.update(label="✅ Risposta elaborata!", state="complete", expanded=False)
                    
                    # Scriviamo la risposta finale nella chat.
                    st.markdown(response)
                    
                    # Salviamo la risposta nella cronologia per il futuro.
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    # Gestione errori: se l'agente crasha, lo mostriamo qui invece di rompere l'app.
                    status.update(label="❌ Errore", state="error")
                    st.error(f"Errore: {e}")