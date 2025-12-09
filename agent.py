# Importiamo il modulo 'os' per interagire con il sistema operativo (es. percorsi file).
import os
# Importiamo 're' (Regular Expressions) per analizzare stringhe (usato nella calcolatrice).
import re
# Importiamo Streamlit per l'interfaccia web e la gestione della cache.
import streamlit as st
# Importiamo le librerie fondamentali di LangChain per l'orchestrazione dell'IA.
import langchain
# Importiamo Redis, un database veloce in memoria, per salvare la cache delle risposte.
import redis
# Importiamo il sistema di cache di LangChain per non rifare due volte la stessa domanda all'LLM.
from langchain_community.cache import RedisCache
# Importiamo il connettore per Ollama (il software che fa girare Llama 3.1 in locale).
from langchain_community.chat_models import ChatOllama
# Importiamo gli schemi per costruire i messaggi (Sistema e Utente) da inviare all'IA.
from langchain.schema import HumanMessage, SystemMessage
# Importiamo i prompt (le istruzioni "psicologiche" per l'IA) dal file di configurazione.
from config import QA_PROMPT, SYSTEM_MESSAGE

# --- Configurazione Cache LLM ---
# Proviamo a configurare la cache globale di LangChain.
# Questo serve a velocizzare le risposte: se chiedi una cosa già chiesta, risponde subito senza ricalcolare.
try:
    from langchain.globals import set_llm_cache
except ImportError:
    # Fallback per versioni vecchie di LangChain
    def set_llm_cache(cache):
        langchain.llm_cache = cache

# Funzione per attivare Redis (il database della cache).
def setup_redis_cache():
    try:
        # Tentiamo di connetterci a Redis sulla porta standard (6379) del computer locale (localhost).
        r = redis.Redis(host="localhost", port=6379, db=0)
        # Facciamo un "ping" per vedere se Redis è vivo.
        if r.ping():
            # Se risponde, diciamo a LangChain di usare Redis come memoria cache.
            set_llm_cache(RedisCache(redis_=r))
            print("✅ Redis Cache attivata!") # Feedback nel terminale
        else:
            print("⚠️ Redis non risponde.")
    except Exception:
        # Se Redis non è installato o spento, l'agente funzionerà comunque (solo un po' più lento nel ripetere risposte).
        print("⚠️ Cache disattivata (Redis non trovato).")

# --- CLASSE MOTORE PRINCIPALE ---
# Questa classe è il cuore del sistema. Sostituisce l'Agente generico con una logica su misura.
class UniLawEngine:
    def __init__(self, vector_db):
        # Salviamo il riferimento al database vettoriale (dove sono salvati i PDF indicizzati).
        self.vector_db = vector_db
        
        # CONFIGURAZIONE HARDWARE (Ottimizzato per GTX 1070 8GB)
        # Impostiamo il limite di contesto a 12288 token (molto alto).
        # Questo permette all'IA di leggere capitoli interi di regolamenti e tabelle lunghe in una volta sola.
        self.CTX_LIMIT = 12288 

        # Inizializziamo il modello di linguaggio (LLM).
        self.llm = ChatOllama(
            model="llama3.1:8b", # Usiamo Llama 3.1 versione 8 Miliardi di parametri.
            temperature=0.0,     # Temperature a 0: creatività annullata. Vogliamo risposte precise e ripetibili (Notarili).
            num_ctx=self.CTX_LIMIT, # Passiamo il limite di memoria aumentato.
        )
        
        # Configuriamo il "Retriever" (il pescatore di documenti).
        # Ne pesca 60 (k=60) inizialmente. Sono tanti, ma poi li filtreremo noi con il codice Python.
        self.retriever = vector_db.as_retriever(
            search_type="similarity", # Cerca per somiglianza semantica.
            search_kwargs={"k": 60} 
        )

    # --- FUNZIONE ROUTER (Il Vigile Urbano) ---
    # Questa funzione decide QUALE file guardare in base alla domanda.
    def _select_target_file(self, question: str):
        """
        ROUTER DETERMINISTICO: Decide quale file è l'AUTORITÀ per la domanda.
        Restituisce una stringa parziale del nome file o None se la domanda è generica.
        """
        # Convertiamo la domanda in minuscolo per facilitare i controlli.
        q = question.lower()
        
        # 1. LOGICA ACCESSO / TOLC
        # Se la domanda contiene parole come "tolc", "punteggio", "ofa"...
        if any(x in q for x in ["tolc", "accesso", "test", "punteggio", "ofa", "immatricola", "ammission"]):
            return "regolamento-di-accesso" # ...forza l'uso SOLO del Regolamento di Accesso.
        
        # 2. LOGICA BORSA / SOLDI
        # Se la domanda parla di soldi, isee, tasse...
        if any(x in q for x in ["borsa", "isee", "ispe", "reddito", "tasse", "alloggio", "mensa", "benefici"]):
            return "bando borsa" # ...forza l'uso SOLO del Bando Borsa di Studio.
        
        # 3. LOGICA TESI
        # Se la domanda parla di laurea o tesi...
        if any(x in q for x in ["tesi", "laurea", "seduta", "prova finale", "voto di laurea"]):
            return "prova-finale" # ...forza l'uso del Regolamento Prova Finale.
            
        # 4. LOGICA PIANO STUDI
        # Se la domanda parla di esami o materie...
        if any(x in q for x in ["piano", "materie", "esami", "insegnament"]):
            return "piano-di-studi" # ...forza l'uso del Piano di Studi.

        # Se nessuna parola chiave è trovata, restituisce None (cerca ovunque).
        return None

    # --- LOGICA RAG (Retrieval-Augmented Generation) ---
    # Questa è la pipeline principale che esegue il lavoro.
    def _rag_logic(self, question: str) -> str:
        
        # A. ROUTING: Chiediamo al "Vigile" quale file dobbiamo consultare.
        target_file_key = self._select_target_file(question)
        
        # B. RETRIEVAL MIRATO (Recupero)
        # Chiediamo al database vettoriale i 60 frammenti più simili alla domanda.
        raw_docs = self.retriever.get_relevant_documents(question)
        
        # C. FILTRAGGIO DETERMINISTICO
        if target_file_key:
            # Se il Router ha scelto un file specifico (es. "bando borsa")...
            print(f"🔒 TARGET LOCK: Analisi esclusiva su file contenenti '{target_file_key}'")
            # ...Creiamo una lista tenendo SOLO i documenti che hanno quel nome nel metadata.
            # Buttiamo via tutto il resto (es. se cerco TOLC, butto via i documenti sulle tasse).
            filtered_docs = [d for d in raw_docs if target_file_key in d.metadata.get("source", "").lower()]
            
            # Se abbiamo trovato qualcosa col filtro, usiamo quello.
            if filtered_docs:
                docs_to_process = filtered_docs
            else:
                # Fallback: se il filtro ha svuotato tutto (strano), torniamo ai docs grezzi per sicurezza.
                docs_to_process = raw_docs
        else:
            # Se il router non ha deciso nulla (domanda generica), usiamo tutti i documenti trovati.
            docs_to_process = raw_docs

        # Se alla fine non abbiamo documenti, ci fermiamo subito.
        if not docs_to_process:
            return "Non ho trovato documenti pertinenti."

        # D. RE-RANKING PER CONTENUTO (Ordinamento Intelligente)
        # Anche se abbiamo filtrato il file giusto, dobbiamo capire quale PAGINA è più importante.
        q_lower = question.lower()
        
        # Funzione interna per dare un punteggio ai paragrafi
        def content_booster(doc):
            content = doc.page_content.lower() # Il testo del paragrafo
            score = 0
            
            # CASO SOLDI: Se cerchiamo borse o ISEE...
            if "borsa" in q_lower or "isee" in q_lower:
                if "25.500" in content: score += 1000 # BONUS ENORME se troviamo la cifra esatta del limite ISEE.
                if "art. 4" in content: score += 500  # BONUS per l'articolo 4 (che contiene i requisiti).
                if "tabella" in content: score += 200 # BONUS se c'è una tabella.
            
            # CASO TOLC: Se cerchiamo ammissione...
            if "tolc" in q_lower:
                if "tabella 1" in content or "tabella" in content: score += 500 # Cerchiamo la tabella dei voti.
                if "16" in content and "ofa" in content: score += 500 # Cerchiamo la soglia del 16.
                if "matematica discreta" in content: score += 500 # Cerchiamo la modalità di recupero specifica.
            
            return score

        # Riordiniamo i documenti: quelli con score più alto vanno per primi.
        # Questo assicura che la tabella ISEE o i voti TOLC siano letti per primi dall'LLM.
        docs_to_process = sorted(docs_to_process, key=content_booster, reverse=True)

        # E. CONTEXT STUFFING (Riempimento della Memoria)
        # Riempiamo la memoria della GPU (il Prompt) finché c'è spazio.
        final_docs = []
        current_char_count = 0
        MAX_CHARS = 35000  # Limite di caratteri (~9-10k token). Lasciamo un po' di spazio per la risposta.

        for doc in docs_to_process:
            doc_len = len(doc.page_content)
            # Se c'è ancora spazio nel "secchio", aggiungiamo il documento.
            if current_char_count + doc_len < MAX_CHARS:
                final_docs.append(doc)
                current_char_count += doc_len
            else:
                break # Se siamo pieni, ci fermiamo.
        
        # Debug nel terminale per vedere quanto stiamo leggendo.
        print(f"📚 Context Load: {len(final_docs)} macro-sezioni ({current_char_count} caratteri).")

        # F. GENERAZIONE PROMPT
        # Costruiamo il testo finale da passare all'IA.
        context_chunks = []
        snippets = []
        
        for i, doc in enumerate(final_docs, start=1):
            text = doc.page_content
            filename = os.path.basename(doc.metadata.get("source", ""))
            page = doc.metadata.get("page", 0)
            
            # Aggiungiamo un'etichetta chiara per dire all'IA da dove viene quel testo.
            context_chunks.append(f"--- DOCUMENTO: {filename} (Pagina {page + 1}) ---\n{text}\n")
            
            # Prepariamo la citazione breve da mostrare all'utente alla fine (es. "Pag. 5").
            ref = f"📄 {filename} (Pag. {page + 1})"
            if ref not in snippets: snippets.append(ref)

        # Uniamo tutti i pezzi di testo in un'unica stringa gigante.
        full_context = "\n".join(context_chunks)
        
        # Creiamo i messaggi per l'IA:
        # 1. SYSTEM_MESSAGE: "Sei un notaio..." (la personalità).
        # 2. QA_PROMPT: "Ecco i documenti: ... Ecco la domanda: ..."
        messages = [
            SystemMessage(content=SYSTEM_MESSAGE),
            HumanMessage(content=QA_PROMPT.format(context=full_context, question=question))
        ]
        
        # G. CHIAMATA ALL'IA
        # Qui avviene la magia: la GPU elabora tutto e genera la risposta.
        ai_response = self.llm.invoke(messages)
        answer = ai_response.content

        # Aggiungiamo le fonti usate in fondo alla risposta per trasparenza.
        if snippets:
            answer += "\n\n---\n**Fonti Utilizzate:**\n" + "\n".join(snippets[:5])
            
        return answer

    # --- METODO RUN PUBBLICO ---
    # Questo è il metodo chiamato dall'interfaccia grafica.
    def run(self, user_input: str):
        # 1. Controllo Calcolatrice (Sicurezza)
        # Se l'utente scrive solo un calcolo matematico (es. "25000 / 2"), lo eseguiamo direttamente con Python.
        # La regex controlla che ci siano solo numeri e operatori sicuri.
        if re.match(r'^[0-9+\-*/%.()\s]{3,}$', user_input.strip()):
            try:
                # 'eval' esegue il calcolo.
                return f"🔢 **Risultato:** {eval(user_input, {'__builtins__': None}, {})}"
            except: pass # Se fallisce, prosegue con il RAG normale.
        
        # 2. Altrimenti, esegui la logica RAG completa definita sopra.
        return self._rag_logic(user_input)

# --- WRAPPER STREAMLIT ---
# Questa funzione serve a Streamlit per non creare un nuovo Agente ogni volta che clicchi un bottone.
# Lo crea una volta sola e lo tiene in memoria (Cache Resource).
@st.cache_resource(show_spinner=False)
def get_cached_agent(_vector_db):
    return UniLawEngine(_vector_db)