# Gestisce solo i dati: legge i file, crea i vettori e il database (non sa nulla dell'interfaccia grafica). 
import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource(show_spinner=False)
def inizializza_conoscenza():
    folder_path = "documenti"

    if not os.path.exists(folder_path):
        return None, "⚠️ Cartella 'documenti' non trovata."
    
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        return None, "⚠️ Metti dei PDF nella cartella 'documenti'!"

    all_texts = []

    for pdf_path in pdf_files:
        print(f"👉 Sto analizzando: {os.path.basename(pdf_path)}...")
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=200,
            )
            all_texts.extend(text_splitter.split_documents(documents))
        except Exception as e:
            print(f"Errore leggendo {pdf_path}: {e}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    db = Chroma.from_documents(all_texts, embeddings)
    return db, f"✅ Agente Operativo. Ho letto {len(pdf_files)} documenti."