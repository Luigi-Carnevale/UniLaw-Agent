# 🎓 UniLaw AI – Assistente Zero-Coda

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Llama 3.1](https://img.shields.io/badge/AI-Llama%203.1%208B-ff69b4.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)
![GPU](https://img.shields.io/badge/Hardware-GPU%20Accelerated-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Un motore RAG di precisione progettato per trasformare il labirinto burocratico universitario in risposte immediate e "notarili".** > UniLaw AI elimina le code in segreteria e le allucinazioni dell'IA generica, utilizzando un'architettura deterministica con **Router Semantico** e **Macro-Chunking**.

---

## 📋 Indice
1. [Il Problema](#-il-problema-il-labirinto-burocratico) 
2. [La Soluzione](#-la-soluzione-un-notaio-digitale)
3. [Nuova Architettura Tecnica](#️-nuova-architettura-tecnica)
4. [Componenti del Sistema](#-componenti-del-sistema)
5. [Installazione](#-installazione)
6. [Configurazione](#-configurazione)
7. [Utilizzo](#️-utilizzo)
8. [Roadmap Futura](#-roadmap-futura) 

---

## 🚨 Il Problema: Il "Labirinto Burocratico"

Ogni anno, studenti e segreterie si scontrano con tre ostacoli critici:

1.  **Frammentazione dell'Informazione:** Le regole sono disperse in decine di PDF (Bandi, RAD, Regolamenti Didattici, Guide). Capire se si ha un *OFA* o se si rientra nella *No Tax Area* richiede di incrociare dati da 3 documenti diversi.
2.  **Il Limite di ChatGPT:** Se chiedi a un'IA generica *"Qual è la scadenza per la borsa di studio?"*, questa inventa una data plausibile ma falsa (allucinazione), perché non conosce il bando specifico del tuo Ateneo per l'anno corrente.
3.  **Il "Muro" del Linguaggio:** I documenti usano un linguaggio tecnico ("coorte", "ISEE parificato", "CFU caratterizzanti") che confonde le matricole, generando migliaia di ticket ripetitivi per le segreterie.

---

## 💡 La Soluzione: Un "Notaio Digitale"

**UniLaw AI** non è un semplice chatbot. È un motore di consultazione che agisce con il rigore di un funzionario esperto.

Invece di "indovinare", il sistema:
1.  **Isola il contesto:** Se chiedi di *Tasse*, chiude a chiave i documenti sulla *Didattica* (niente interferenze).
2.  **Legge tutto:** Usa la GPU per leggere intere pagine di regolamento (Macro-Chunking), preservando tabelle complesse e liste puntate che i RAG normali spezzano.
3.  **Cita le fonti:** Ogni risposta è ancorata a un documento ufficiale preciso (es. *Art. 4 del Bando Borsa*).

### Perché UniLaw è diverso?

| Feature | Chatbot Generico (ChatGPT/Gemini) | RAG Standard | 🎓 UniLaw AI |
| :--- | :--- | :--- | :--- |
| **Fonte Dati** | Internet (generico, spesso obsoleto) | Frammenti di PDF a caso | **Router Deterministico** (Solo il PDF giusto) |
| **Precisione** | Bassa (Allucina date e regole) | Media (Perde il contesto delle tabelle) | **Massima** (Copia cifre e norme esatte) |
| **Privacy** | Dati inviati al cloud | Dati inviati al cloud | **100% Locale & Offline** |

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
