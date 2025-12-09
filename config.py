# Importa il modulo 'os' del sistema operativo.
# Serve per interagire con le variabili d'ambiente (impostazioni di sistema).
import os

# Importa il modulo 'logging'.
# Serve per gestire i messaggi di log (errori, avvisi, informazioni) che appaiono nel terminale.
import logging

# Importa la classe 'PromptTemplate' dalla libreria LangChain.
# Serve per creare modelli di testo (template) dinamici dove inseriremo i dati (domanda e documenti).
from langchain.prompts import PromptTemplate

# --- CONFIGURAZIONE AMBIENTE ---
# Definisco una funzione per preparare l'ambiente di esecuzione prima di avviare l'app.
def setup_environment():
    # La riga seguente è commentata (#). Se fosse attiva, nasconderebbe la GPU al programma.
    # Lasciandola commentata, permettiamo a Ollama e agli Embeddings di usare la scheda video (GPU) per andare veloce.
    # os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    
    # Imposto una variabile d'ambiente per disabilitare la telemetria (raccolta dati anonima) di ChromaDB/LangChain.
    # Serve per privacy e per evitare rallentamenti di rete inutili.
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    
    # Configuro il livello di log per la libreria 'pypdf' (che legge i PDF).
    # Impostandolo su ERROR, nascondiamo tutti i messaggi di avviso (Warning) inutili nel terminale,
    # mostrando solo gli errori gravi. Mantiene la console pulita.
    logging.getLogger("pypdf").setLevel(logging.ERROR)

# --- CSS E STILE ---
# Definisco una stringa multilinea contenente codice CSS (Cascading Style Sheets).
# Questo serve per personalizzare l'aspetto grafico dell'interfaccia Streamlit.
CSS_STYLES = """
<style>
    /* Nasconde l'icona dello "stato" di Streamlit (l'omino che corre) in alto a destra */
    [data-testid="stStatusWidget"] { visibility: hidden; }
    
    /* Arrotonda i bordi dei messaggi della chat per renderli più gradevoli (raggio 15px) */
    .stChatMessage { border-radius: 15px; }
    
    /* Stile per il titolo principale (Main Header) */
    .main-header { 
        font-size: 3rem;           /* Dimensione testo molto grande */
        color: #b71c1c;            /* Colore rosso scuro (stile universitario) */
        text-align: center;        /* Testo centrato */
        font-weight: 800;          /* Testo molto grassetto (Bold) */
        font-family: 'Helvetica Neue', sans-serif; /* Font moderno */
        margin-top: -20px;         /* Sposta il titolo un po' più in alto */
    }
    
    /* Stile per il sottotitolo (Sub Header) */
    .sub-header { 
        font-size: 1.3rem;         /* Dimensione testo media */
        color: white;              /* Colore bianco (per contrasto su sfondo scuro) */
        text-align: center;        /* Testo centrato */
        margin-bottom: 30px;       /* Spazio vuoto sotto il sottotitolo */
        font-style: italic;        /* Testo in corsivo */
    }
</style>
"""

# --- PROMPT NOTARILE ---
# Definisco il "template" (modello) di istruzioni per l'Intelligenza Artificiale.
# Questo è il testo che l'AI leggerà prima di rispondere. Definisce il suo comportamento "rigido".
qa_template_notary = """
Sei l'Assistente Ufficiale del Dipartimento di Informatica.
Il tuo compito è agire come un NOTAIO: riporta solo fatti, numeri e regole scritte nei documenti.

DOCUMENTI UFFICIALI (Fonte Unica):
{context}  
---
DOMANDA UTENTE: {question}
---

REGOLE DI RISPOSTA (ASSOLUTE):
1. **TERMINOLOGIA**: Usa le parole esatte del testo (es. se dice "sconsigliata", non scrivere "vietata").
   2. **NUMERI**: Se c'è una tabella con punteggi o soldi, copiala. Riporta le cifre esatte (es. 25.500,00 €).
   3. **COMPLETEZZA**:
   - Per domande su OFA: elenca TUTTE le modalità di recupero citate.
   - Per domande su BORSE: elenca SIA i requisiti economici (ISEE/ISPE) CHE quelli di merito (CFU).
   4. **ONESTÀ**: Se l'informazione non è scritta esplicitamente nei frammenti qui sopra, scrivi: "Il documento non specifica questo dettaglio."
   RISPOSTA (Strutturata e fedele al testo):
"""

# Creo l'oggetto PromptTemplate effettivo usando la libreria LangChain.
# Questo oggetto collega la stringa di testo sopra alle variabili che verranno riempite dinamicamente.
QA_PROMPT = PromptTemplate(
    template=qa_template_notary,       # Uso la stringa definita sopra come modello
    input_variables=["context", "question"], # Dico a LangChain che dovrà cercare {context} e {question} per riempirli
)

# Definisco il "Messaggio di Sistema".
# Questa è l'istruzione di base che definisce la personalità dell'AI a livello profondo.
SYSTEM_MESSAGE = """
Sei un notaio digitale.
La tua priorità è l'accuratezza fattuale.
Non riassumere se questo comporta perdere dettagli tecnici.
Cita sempre: [NomeFile, Pagina X].
"""
# L'ultima riga obbliga l'AI a citare sempre la fonte per ogni affermazione.