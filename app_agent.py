# Punto di ingresso. Questo è il file da lanciare. Tutta la logica è altrove. 
import streamlit as st
from config import setup_environment, CSS_STYLES
from database import inizializza_conoscenza
from agent import get_cached_agent, setup_redis_cache

# 1. Configurazione Iniziale
setup_environment()
st.set_page_config(page_title="UniLaw Agent", page_icon="🤖", layout="wide")
st.markdown(CSS_STYLES, unsafe_allow_html=True)

# 2. Setup Logica
setup_redis_cache()
vector_db, msg = inizializza_conoscenza()

# 3. Sidebar
with st.sidebar:
    st.title("⚙️ Pannello Agente")
    st.success(msg if vector_db else "Errore DB")
    if st.button("🗑️ Svuota Chat"):
        st.session_state.messages = []
        st.rerun()

# 4. Header UI
st.markdown('<div class="main-header">UniLaw Agent 🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Non solo ricerca, ma ragionamento autonomo.</div>', unsafe_allow_html=True)

# 5. Stato Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Sono UniLaw Agent. Posso consultare i documenti e usare strumenti di calcolo."
    }]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 6. Loop Principale
if prompt := st.chat_input("Es: Quali sono i requisiti formali della tesi per il corso L-31?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not vector_db:
            st.error("⚠️ Database non trovato.")
        else:
            with st.status("🧠 L'Agente sta ragionando...", expanded=True) as status:
                try:
                    agent = get_cached_agent(vector_db)
                    response = agent.run(prompt)
                    
                    status.update(label="✅ Risposta elaborata!", state="complete", expanded=False)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    status.update(label="❌ Errore", state="error")
                    st.error(f"Errore: {e}")