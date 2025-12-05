# 🎓 UniLaw AI - Assistente Zero-Coda

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![IBM watsonx](https://img.shields.io/badge/AI-IBM%20watsonx-052FAD.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)

> **L'Intelligenza Artificiale che legge i bandi al posto tuo.** > Un sistema RAG (Retrieval-Augmented Generation) basato su IBM watsonx.ai per interrogare documenti universitari in linguaggio naturale.

---

## 📋 Indice
1. [Il Problema](#-il-problema)
2. [La Soluzione](#-la-soluzione)
3. [Architettura Tecnica](#-architettura-tecnica)
4. [Installazione](#-installazione)
5. [Configurazione](#-configurazione)
6. [Utilizzo](#-utilizzo)
7. [Roadmap Futura](#-roadmap-futura)

---

## 🚨 Il Problema
Ogni anno, le segreterie universitarie vengono inondate da migliaia di email e ticket per domande ripetitive:
* *"Qual è la scadenza dell'ISEE?"*
* *"Come calcolo la media ponderata?"*
* *"Quanto pago di tasse se sono fuori corso?"*

Le risposte esistono, ma sono sepolte in **PDF di 100 pagine** (Bandi, Regolamenti, Guide) che nessuno legge perché complessi e lunghi da consultare.

## 💡 La Soluzione
**UniLaw AI** è un assistente intelligente che:
1.  **Ingerisce** i documenti ufficiali (PDF) forniti dall'Ateneo.
2.  **Indicizza** il contenuto creando una mappa semantica locale.
3.  **Risponde** alle domande degli studenti citando l'articolo e la pagina esatta.

**Vantaggi:**
* 🕒 **Risparmio tempo:** Risposte immediate 24/7.
* 🎯 **Precisione:** Zero allucinazioni, risponde solo basandosi sui documenti.
* 🔒 **Privacy:** I dati sensibili non lasciano mai l'infrastruttura dell'ateneo (RAG Locale).

---

## ⚙️ Architettura Tecnica

Il progetto utilizza un approccio **RAG Puro (Retrieval-Augmented Generation)**:

1.  **Document Loading:** `PyPDFLoader` estrae il testo dai documenti ufficiali.
2.  **Chunking:** Il testo viene diviso in blocchi da 1000 caratteri per mantenere il contesto.
3.  **Embeddings & Vector Store:** Utilizziamo **HuggingFace** (locale) per trasformare il testo in numeri e **ChromaDB** per archiviarli e ricercarli velocemente.
4.  **Generation (LLM):** Il contesto recuperato viene inviato a **IBM watsonx.ai** (modello `ibm/granite-13b-chat-v2`), che formula la risposta finale.
5.  **Frontend:** Interfaccia **Streamlit** moderna e reattiva.

---

## 🚀 Installazione

### Prerequisiti
* Python 3.9 o superiore
* Account IBM Cloud (per le API watsonx)

### Passaggi
1.  **Clona il repository:**
    ```bash
    git clone [https://github.com/tuo-username/assistente-zero-coda.git](https://github.com/tuo-username/assistente-zero-coda.git)
    cd assistente-zero-coda
    ```

2.  **Crea l'ambiente virtuale:**
    ```bash
    python -m venv .venv
    # Windows:
    .\.venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Installa le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🔧 Configurazione

1.  Crea una cartella chiamata `documenti` nella root del progetto.
2.  Inserisci al suo interno i PDF ufficiali (es. `Regolamento_Tasse_2024.pdf`).
3.  All'avvio, l'app indicizzerà automaticamente qualsiasi file presente in questa cartella.

---

## ▶️ Utilizzo

Avvia l'applicazione con il comando:
```bash
streamlit run app_precaricata.py

---

## All'apertura:

1. Attendi il caricamento della barra di progresso (Indicizzazione).

2.Apri la barra laterale a sinistra.

3.Inserisci le tue IBM Cloud API Key e Project ID.

4. Fai una domanda in chat (es: "Quali sono le scadenze per l'Erasmus?").

---

## 🔮 Roadmap Futura

[ ] Integrazione con watsonx Orchestrate per automatizzare l'apertura di ticket.

[ ] Supporto multilingua per studenti internazionali.

[ ] Canale vocale (Speech-to-Text) per accessibilità.
