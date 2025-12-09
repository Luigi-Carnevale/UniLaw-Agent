# 🎓 UniLaw AI – Assistente Zero-Coda

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Llama 3.1](https://img.shields.io/badge/AI-Llama%203.1%208B-ff69b4.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)
![GPU](https://img.shields.io/badge/Hardware-GPU%20Accelerated-orange.svg)


> **Un motore RAG di precisione progettato per eliminare le allucinazioni e garantire risposte "notarili" su bandi e regolamenti universitari.** > UniLaw AI abbandona l'approccio probabilistico generico in favore di un'architettura deterministica con **Router Semantico**, **Macro-Chunking** per la lettura di tabelle complesse e un **Engine Custom** ottimizzato per GPU NVIDIA.

---

## 📋 Indice
1. [Il Problema](#-il-problema) 
2. [La Soluzione](#-la-soluzione)
3. [Nuova Architettura Tecnica](#️-nuova-architettura-tecnica)
4. [Componenti del Sistema](#-componenti-del-sistema)
5. [Installazione](#-installazione)
6. [Configurazione](#-configurazione)
7. [Utilizzo](#️-utilizzo)
8. [Roadmap Futura](#-roadmap-futura) 

---

## 🚨 Il Problema

I sistemi RAG tradizionali falliscono su documenti burocratici complessi:
- **Spezzano le tabelle:** I chunk piccoli rendono illeggibili i requisiti ISEE o i voti TOLC.
- **Confondono i contesti:** Cercando "scadenze", l'AI mischia le date della Borsa di Studio con quelle dell'Erasmus.
- **Allucinano:** Inventano regole quando non trovano il paragrafo esatto.

---

## 💡 La Soluzione

**UniLaw AI** evolve il concetto di assistente universitario passando da un "Chatbot Generico" a un **"Notaio Digitale"**.

1. **Router Deterministico:** Capisce l'intento (es. "TOLC") e **blocca fisicamente** l'accesso ai documenti non pertinenti (es. Bando Tasse).
2. **Macro-Chunking:** Legge blocchi di **2500 caratteri** (pagine intere) preservando tabelle e articoli di legge nella loro integrità.
3. **Precisione Assoluta:** Istruito per copiare dati numerici esatti (Euro, CFU, Voti) senza interpretarli.
4. **Deep Reading:** Sfrutta la GPU per analizzare contesti molto ampi (fino a 12k token).

---

## ⚙️ Nuova Architettura Tecnica

### 1️⃣ Data Ingestion (Macro-Chunking)
- **Parser:** `PDFPlumberLoader` (essenziale per l'estrazione accurata di tabelle e layout complessi).
- **Chunking:** Dimensione aumentata a **2500 caratteri** (con overlap 500) per mantenere uniti articoli di regolamento e griglie di dati.
- **Metadati:** Arricchimento automatico per il filtraggio deterministico.

### 2️⃣ Core Engine (UniLaw Custom)
Non usiamo più agenti LangChain generici (lenti e imprecisi), ma una pipeline custom:
- **Semantic Router:** Una logica condizionale che seleziona il "Documento Sacro" in base alla domanda.
- **Ranking "Cecchino":** Algoritmo di re-ranking che premia i paragrafi contenenti parole chiave critiche (es. "Art. 4", "ISEE", "Tabella 1").
- **Context Stuffing:** Riempimento intelligente della memoria della GPU fino al limite fisico.

### 3️⃣ LLM & Hardware
- **Modello:** Llama 3.1 8B (Quantizzato).
- **Hardware:** Ottimizzato per **NVIDIA GTX 1070** (8GB VRAM).
- **Settings:** `temperature=0.0` (Creatività annullata per massima fedeltà) e `num_ctx=12288`.

---

## 🧱 Componenti del Sistema

### 📂 Documenti
Cartella `documenti/` contenente i PDF ufficiali (Regolamenti, RAD, Bandi).  
Il sistema ora gestisce perfettamente:
- Tabelle ISEE
- Griglie voti TOLC
- Elenchi puntati complessi

### 🧠 UniLaw Engine
Il cuore del sistema. Sostituisce l'agente ReAct con una logica:
1. **Analisi Intento:** (TOLC? Soldi? Tesi?)
2. **Target Lock:** Selezione esclusiva del file pertinente.
3. **Extraction:** Prelievo dei dati esatti.

### 🛡️ Prompt "Notaio"
Un set di istruzioni di sistema (`config.py`) che obbliga l'AI a:
- Usare la terminologia esatta ("Sconsigliata" vs "Vietata").
- Riportare cifre esatte.
- Dichiarare se un'informazione è assente invece di inventarla.

### 🛠️ Tools
- **Calcolatrice Sicura:** Esecuzione sandboxata di espressioni matematiche per calcoli rapidi.

---

## 🚀 Installazione

### Prerequisiti
- Python 3.10+  
- **Ollama** installato e funzionante.
- GPU NVIDIA consigliata (ma funziona anche su CPU, più lentamente).

### 1️⃣ Setup
```bash
git clone [https://github.com/Luigi-Carnevale/UniLaw-Agent.git](https://github.com/Luigi-Carnevale/UniLaw-Agent.git)
cd UniLaw-Agent
```

### 2️⃣ Ambiente virtuale

**Creazione**  
```bash
python -m venv venv   # Creazione
```
**Attivazione**  
```bash
source venv/bin/activate   # per Linux/macOS
venv\Scripts\activate    # per Windows
```

### 3️⃣ Installazione delle dipendenze
```bash
pip install -r requirements.txt
```

---

## 🔧 Configurazione

1. Assicurati che esista la cartella:
```
documenti/
```

2. Inserisci dentro i PDF ufficiali dell’Ateneo:
- bandi  
- regolamenti  
- guide studenti  
- RAD  
- piani di studio  
- linee guida tesi  

3. All’avvio l’indicizzazione parte automaticamente.

4. Importante: Al primo avvio (o se cambi i PDF), usa il pulsante "Aggiorna Documenti" nella sidebar per creare i Macro-Chunk ottimizzati.

---

## ▶️ Utilizzo

Avvia l'applicazione: 
```bash
streamlit run app_agent.py
```

## All’apertura:

### 1. Primo avvio:   
Attendi che il terminale completi la "Lettura Profonda (Macro-Chunk) dei PDF. È un'operazione una tantum per indicizzare tabelle e articoli interi. 

### 2. Verifica
Apri la sidebar. Se hai aggiunto nuovi file, premi "🔄 Aggiorna Documenti". 

### 3. Interazione
Fai domande specifiche per testare la precisione "Notarile":

#### 3.1 Esempi: 
- "Ho fatto  12 punti al TOLC, sono ammesso?" (Test Router Accesso)
- "Qual'è il limite ISEE estratto per la borsa?" (Test Router Borsa + Estrazione Numeri)
- "Calcola il 20% di 24.000" (Test Tool Calcolatrice)

## Cosa succede dietro le quinte? 
L’agente segue una pipeline rigorosa:
### Analisi intento: 
- Se è un calcolo (es. "20000 * 5%"), esegue la Calcolatrice Sandbox.
- Se è una domanda (es. "Scadenza TOLC"), attiva il Semantic Router.
### Target Lock:
- Il sistema identifica l'argomento (es. Ammissione) e blocca l'accesso ai documenti irrilevanti (es. Bando Tasse), prevenendo contaminazioni.
### Deep Retrieval
- La GPU recupera intere pagine o articoli di regolamento (2500 caratteri) per preservare il contesto di tabelle e liste.
### Risposta "Notarile"
- L'LLM estrae i dati esatti (cifre, date, voti) senza riassunti approssimativi.

---

## 🔮 Roadmap Futura

- Upload PDF via UI  
- Citazione puntuale con link alla pagina del PDF.
- Esportazione risposte in PDF 
- Dashboard richieste  
- Modalità "Confronto" (es. differenze tra Bando 2024 e 2025).
- Login SSO   

---
