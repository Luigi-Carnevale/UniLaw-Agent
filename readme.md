# 🎓 UniLaw AI – Assistente Zero-Coda

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Llama 3.1](https://img.shields.io/badge/AI-Llama%203.1%208B-ff69b4.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)
![Redis](https://img.shields.io/badge/Cache-Redis-darkred.svg)

> **Un assistente AI progettato per ridurre a zero le attese degli studenti, automatizzando la consultazione di bandi, regolamenti e documentazione accademica tramite un sistema RAG avanzato.**  
> UniLaw AI combina RAG locale, agente ReAct, LLM open‑source, database vettoriale e un'interfaccia moderna per fornire risposte affidabili basate esclusivamente sui documenti ufficiali dell’Ateneo.

---

## 📋 Indice
1. [Il Problema](#-il-problema) 
2. [La Soluzione](#-la-soluzione)
3. [Architettura Tecnica](#️-architettura-tecnica)
4. [Componenti del Sistema](#-componenti-del-sistema)
5. [Installazione](#-installazione)
6. [Configurazione](#-configurazione)
7. [Utilizzo](#️-utilizzo)
8. [Roadmap Futura](#-roadmap-futura) 

---

## 🚨 Il Problema

Ogni anno le segreterie universitarie gestiscono migliaia di richieste ripetitive:

- “Come si calcola il voto di laurea?”  
- “Quando scade il bando Erasmus?”  
- “Dove trovo il regolamento tesi?”  
- “Quante tasse pago con questo ISEE?”

Le informazioni esistono, ma sono disperse in PDF lunghi, regolamenti scritti in linguaggio burocratico e documenti difficili da navigare.

---

## 💡 La Soluzione

**UniLaw AI** è un assistente intelligente che:

1. Legge automaticamente i documenti ufficiali dell’Ateneo.  
2. Indicizza e comprende il contenuto tramite RAG.  
3. Risponde in linguaggio naturale **citando le fonti**.  
4. Ragiona in modo autonomo grazie a un **agente ReAct**.

### Vantaggi
- 🕒 Risposte immediate 24/7  
- 📘 Basate solo su documenti ufficiali  
- 🔒 Funziona completamente offline  
- 🎯 Allucinazioni drasticamente ridotte grazie a RAG sui soli documenti ufficiali 
- 🧠 Capacità di ragionamento multi-step  

---

## ⚙️ Architettura Tecnica

### 1️⃣ RAG (Retrieval-Augmented Generation)

- Parsing PDF con PyPDFLoader  
- Chunking (700 caratteri + overlap 200)  
- Embeddings MiniLM multilingua  
- ChromaDB vettoriale  
- Recupero semantico via MMR

### 2️⃣ Modello LLM Locale — Llama 3.1 8B (Ollama)

Usato per: 
- Interpretare domande
- Analizzare contesto
- Generare risposte accurate

### 3️⃣ Agente ReAct

Decide autonomamente come rispondere combinando: 
- Reasoning multi-step  
- Uso strumenti intelligenti:
  - KnowledgeBase_Universitaria  
  - Calcolatrice_tasse  
- Recupero informazioni
  
### 4️⃣ Redis Cache (opzionale)

Accelera risposte e caching LLM.

### 5️⃣ UI Streamlit

- Design moderno
- Chat persistente
- Indicatori di stato
- Sidebar funzionale

---

## 🧱 Componenti del Sistema

### Documenti

Cartella:

```
documenti/
```

- Contiene regolamenti, bandi, guide studenti, piani di studio.
- Parsing PDF
- Indicizzazione automatica

### Vector Store

ChromaDB + Embeddings HuggingFace.

### 🧠 LLM
- Llama 3.1 8B via Ollama  
- inferenza offline 

### Agente

- ReAct con strumenti dedicati.
- Riduzione drastica delle allucinazioni

### 🛠️ Tools
#### 🔍 KnowledgeBase_Universitaria

- Ricerca semantica nei PDF.

#### 🧮 Calcolatrice_tasse
- Esegue calcoli matematici tramite espressioni Python
- Utile per tasse, percentuali, contributi


---

## 🚀 Installazione

### Prerequisiti

- Python 3.10+  
- Ollama installato → https://ollama.com  
- Modello:  
```bash
ollama pull llama3.1:8b
```
- Redis (facoltativo)

---

### 1️⃣ Clona repo

```bash
git clone https://github.com/Luigi-Carnevale/UniLaw-Agent.git
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

---

## ▶️ Utilizzo

Avvia l'applicazione: 
```bash
streamlit run app_agent.py
```

## All’apertura:
1. Attendi la creazione della knowledge base  
2. Apri la sidebar per verificare lo stato  
3. Fai la tua domanda, ad esempio:
   - “Requisiti prova finale L‑31?”
   - “Cosa prevede il regolamento Erasmus?”
   - “Calcola il 5% di 20.000€”  

L’agente:
- analizza la domanda  
- decide se usare il VectorDB  
- utilizza la calcolatrice se necessario  
- produce una risposta chiara e basata sui documenti 

---

## 🔮 Roadmap Futura

- Upload PDF via UI  
- Citazioni pagina PDF  
- Dashboard richieste  
- Login SSO  
- PWA mobile  
- Esportazione risposte in PDF  

---
