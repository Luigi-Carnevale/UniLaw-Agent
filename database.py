# Importa il modulo 'os' per interagire con il sistema operativo (es. controllare se esistono cartelle).
import os
# Importa il modulo 'glob' per cercare file nel computer usando pattern (es. "*.pdf").
import glob
# Importa Streamlit per gestire l'interfaccia e, soprattutto, la cache (memoria temporanea).
import streamlit as st

# Importa il caricatore di PDF specifico 'PDFPlumber'.
# È stato scelto al posto di PyPDF perché è molto più bravo a leggere le TABELLE (voti, ISEE) senza scombussolarle.
from langchain_community.document_loaders import PDFPlumberLoader 

# Importa lo strumento per "spezzettare" il testo (Text Splitter).
# Serve a dividere i lunghi PDF in pezzi più piccoli (chunk) che l'IA può gestire.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importa Chroma, che è il nostro Database Vettoriale.
# È il "cervello" dove vengono salvati i testi trasformati in numeri (vettori).
from langchain_community.vectorstores import Chroma

# Importa il modello di Embeddings di HuggingFace.
# Questo è il "traduttore" che converte le parole in liste di numeri (vettori) comprensibili al computer.
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- COSTANTI DI CONFIGURAZIONE ---
# Definisco il percorso della cartella dove verrà salvato fisicamente il database sul disco.
CHROMA_PATH = "chroma_db_storage"
# Definisco il percorso della cartella dove devo leggere i file PDF originali.
DOCS_PATH = "documenti"

# --- FUNZIONE DI INIZIALIZZAZIONE ---
# Uso il decoratore @st.cache_resource.
# Questo dice a Streamlit: "Esegui questa funzione pesante UNA SOLA VOLTA all'avvio e ricorda il risultato".
# Senza questo, il database verrebbe ricaricato ogni volta che clicchi un bottone, rallentando tutto.
@st.cache_resource(show_spinner=False)
def inizializza_conoscenza():
    
    # 1. SETUP DEGLI EMBEDDINGS (Il Traduttore)
    # Inizializzo il modello che trasforma il testo in numeri.
    # Usiamo "paraphrase-multilingual-MiniLM-L12-v2" che funziona bene con l'italiano.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # Impostiamo "device": "cpu". Per creare il database la CPU è più stabile e sufficientemente veloce.
        # La GPU la teniamo libera per il "Cervello" (Ollama) che ne ha più bisogno.
        model_kwargs={"device": "cpu"}, 
    )

    # 2. CONTROLLO PERSISTENZA (Il Database esiste già?)
    # Controllo se la cartella 'chroma_db_storage' esiste E se non è vuota.
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("💾 Database trovato su disco. Caricamento rapido...")
        try:
            # Se esiste, carico il database dal disco invece di ricrearlo da zero.
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            # Restituisco il database pronto e un messaggio di successo.
            return db, "✅ Database caricato dal disco (Ready)."
        except Exception as e:
            # Se il caricamento fallisce (es. file corrotto), stampo l'errore.
            print(f"⚠️ Errore caricamento DB, rigenerazione forzata: {e}")

    # ---------------------------------------------------------
    # SE IL CODICE ARRIVA QUI: Il DB non esiste (o è stato cancellato).
    # Inizia la creazione da zero (Ingestione dei Documenti).
    # ---------------------------------------------------------
    
    # Controllo di sicurezza: se la cartella 'documenti' non esiste, mi fermo.
    if not os.path.exists(DOCS_PATH):
        return None, "⚠️ Cartella 'documenti' non trovata."
    
    # Cerco tutti i file che finiscono con ".pdf" dentro la cartella documenti.
    pdf_files = glob.glob(os.path.join(DOCS_PATH, "*.pdf"))
    
    # Se non trovo nessun PDF, mi fermo.
    if not pdf_files:
        return None, "⚠️ Nessun PDF trovato nella cartella!"

    # Lista vuota che conterrà tutti i pezzi di testo (chunk) di tutti i PDF.
    all_texts = []
    print(f"🔄 Inizio indicizzazione di {len(pdf_files)} documenti...")

    # Ciclo su ogni file PDF trovato.
    for pdf_path in pdf_files:
        # Stampo il nome del file che sto leggendo (utile per capire a che punto siamo).
        print(f"👉 Lettura profonda: {os.path.basename(pdf_path)}...")
        try:
            # Carico il PDF usando PDFPlumberLoader.
            # Questo è fondamentale per leggere correttamente le TABELLE del bando borsa.
            loader = PDFPlumberLoader(pdf_path) 
            documents = loader.load() # Legge tutto il testo e lo mette in memoria.
            
            # --- CONFIGURAZIONE CHUNK (Macro-Chunking) ---
            # Qui decidiamo come spezzettare il testo.
            # chunk_size=2500: Pezzi molto grandi (circa una pagina). Serve per non spezzare articoli di legge o tabelle a metà.
            # chunk_overlap=300: Sovrapposizione. Le ultime 300 lettere di un pezzo vengono ripetute nel successivo per mantenere il filo del discorso.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500,    
                chunk_overlap=500,  # Aumentato l'overlap per sicurezza
                # Separatori: prova a spezzare prima ai doppi a capo (paragrafi), poi a capo, ecc.
                # Aggiungendo "Art." cerchiamo di rispettare la struttura legale.
                separators=["\n\n", "Art.", "ARTICOLO", "\n", " "] 
            )
            
            # Eseguo il taglio effettivo del documento corrente.
            splitted_docs = text_splitter.split_documents(documents)
            
            # --- ARRICCHIMENTO METADATI (Cruciale per il Router) ---
            # Per ogni pezzetto creato, aggiungo un'etichetta con il nome del file.
            # Questo servirà all'agente per dire "Cerca solo nel file che si chiama 'regolamento'".
            for doc in splitted_docs:
                fname = os.path.basename(pdf_path).lower() # Prendo il nome file in minuscolo
                doc.metadata["filename"] = fname # Lo salvo nei metadati del pezzetto
                
            # Aggiungo i pezzetti di questo file alla lista generale.
            all_texts.extend(splitted_docs)
            
        except Exception as e:
            # Se un file dà errore, lo stampo ma continuo con gli altri.
            print(f"❌ Errore su {pdf_path}: {e}")

    # 3. CREAZIONE E SALVATAGGIO SU DISCO
    print("💾 Salvataggio database vettoriale...")
    
    # Creo il database Chroma partendo da tutti i testi raccolti.
    # Questa operazione trasforma le parole in vettori (numeri) e li salva nella cartella 'chroma_db_storage'.
    db = Chroma.from_documents(
        documents=all_texts, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    # Restituisco il database pronto all'uso e un messaggio di conferma.
    return db, f"✅ Nuova Knowledge Base creata ({len(all_texts)} frammenti)."