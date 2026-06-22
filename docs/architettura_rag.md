# Architettura RAG — UniLaw Agent

Documento di riferimento sull'architettura logica del sistema. Descrive i
componenti, le responsabilità dei moduli e il flusso di elaborazione di una
domanda. Riflette lo stato del codice a **roadmap completata (FASI 0–9,
2026-06-17)**: descrive ciò che esiste, non ciò che è pianificato (gli elementi
futuri sono indicati come tali).

## 1. Componenti e tecnologie

| Componente | Tecnologia | Ruolo |
|---|---|---|
| Interfaccia | Streamlit | Chat, sidebar, trace, KPI, rebuild |
| Modello linguistico | Ollama (`llama3.1:8b`, `temperature=0`) | Generazione grounded |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace, CPU) | Vettorializzazione chunk e query |
| Vector store | ChromaDB persistente (in `~/Library/Application Support/UniLawAgent/`, fuori da iCloud; configurabile con `UNILAW_CHROMA_DIR`) | Indice e ricerca per similarità |
| Indice lessicale | BM25 Okapi (`rank_bm25`) sui chunk di ChromaDB | Recall sui termini esatti/codici (FASE 3) |
| Reranker neurale (opz.) | cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` (via `sentence-transformers`) | Riordino opzionale dei candidati (FASE 4, default OFF) |
| Cache LLM (opzionale) | Redis | Cache delle risposte del modello, se disponibile |
| Calcolo sicuro | `ast` + whitelist operatori | Aritmetica e percentuali senza `eval` |

## 2. Responsabilità dei moduli

- **`app_agent.py`** — interfaccia e sessione: inizializza la knowledge base,
  gestisce la chat, mostra il `RagTrace`, i KPI e i comandi (rebuild, reset).
  Non contiene logica RAG.
- **`agent.py`** — orchestratore RAG: classe `UniLawResponder`. Coordina il flusso
  (`answer`), ospita retrieval multi-query, reranking euristico, regole
  deterministiche (template), costruzione del contesto, generazione e
  post-processing. Dalla FASE 2 **delega** ai moduli sotto e riesporta il modello
  dati per retrocompatibilità.
- **`rag_types.py`** *(FASE 2)* — modello dati: `QueryIntent`, `RetrievedSource`,
  `RagTrace`, `COURSE_LABELS`, `TOPIC_LABELS`.
- **`intent.py`** *(FASE 2)* — riconoscimento dell'intento (`infer_query_intent`)
  e predicati di disambiguazione (`asks_*`). Dal Ciclo 2 — FASE 11 accetta un
  classificatore semantico opzionale che **affianca** le keyword.
- **`semantic_intent.py`** *(Ciclo 2 — FASE 11)* — classificatore d'intento per
  similarità di embedding (`SemanticIntentClassifier`), **opzionale (default OFF)**,
  con embedder iniettabile e fallback sicuro; riusa il modello di embedding del
  vector store. Riempie solo le caselle (corso/argomento) che le keyword lasciano
  vuote, senza sovrascriverle.
- **`rules_tolc.py`** *(FASE 2)* — regole deterministiche sul punteggio TOLC
  (`extract_tolc_score`, `classify_tolc_score`).
- **`confidence.py`** *(FASE 2)* — stima euristica dell'affidabilità.
- **`citations.py`** *(FASE 2)* — estrazione e formattazione delle citazioni;
  verifica del grounding delle frasi citanti (FASE 5, lessicale) con una **rete di
  recupero semantica opzionale** (Ciclo 2 — FASE 12, `grounding_report(..., embedder)`,
  default OFF) che recupera le parafrasi per similarità di embedding senza sostituire
  il lessicale.
- **`retrieval.py`** *(FASE 3)* — retrieval ibrido: query di espansione, ricerca
  vettoriale (MMR), indice lessicale BM25 sui chunk di ChromaDB, fusione RRF
  (`hybrid_retrieve`).
- **`reranking.py`** *(FASE 3)* — reranker euristico e filtro metadata per corso.
- **`neural_reranker.py`** *(FASE 4)* — reranker neurale opzionale (cross-encoder
  multilingua) con caricamento pigro e fallback automatico all'euristica.
- **`evidence.py`** *(FASE 5)* — evidence selection: estrae i passaggi più
  pertinenti alla domanda (contesto più breve e mirato per il modello).
- **`abstention.py`** *(FASE 6)* — layer di astensione: classifica la causa del
  "non lo so" (fuori dominio corso/topic, ambigua, retrieval debole, fonte
  insufficiente) tramite la *retrieval strength*. Dal Ciclo 2 — FASE 13 la
  distinzione `fuori_dominio` vs `evidenza_insufficiente` ha una variante
  **semantica opzionale** (`semantic_retrieval_strength`, default OFF): misura la
  forza per similarità di embedding query↔fonte invece che per overlap di token,
  con una soglia ricalibrata e validata sull'held-out della FASE 6.
- **`knowledge.py`** *(FASE 7)* — conoscenza normativa strutturata e tracciabile
  (soglie TOLC, prova L-19) con provenienza dalle fonti; unica fonte di verità letta
  da classificatore e template.
- **`trace_export.py`** *(FASE 8)* — esportazione del `RagTrace` in JSON e Markdown.
- **`database.py`** — ingest e indice: caricamento PDF (`PyPDFLoader`),
  chunking (`RecursiveCharacterTextSplitter`, 900/150 caratteri), metadata per
  file/pagina/corso/tipo, embeddings, ChromaDB, manifest e firma SHA-256 del
  corpus per il rebuild incrementale.
- **`config.py`** — costanti, prompt (`QA_PROMPT`, `ANSWER_STYLE_GUIDE`),
  configurazione ambiente e stile (CSS).
- **`tools.py`** — calcolo numerico sicuro.

> Nota: retrieval e reranking sono stati estratti in FASE 3 (`retrieval.py`,
> `reranking.py`). Restano in `agent.py` i template deterministici e la
> formattazione, in attesa di essere riscritti nelle FASI 5/7.

## 3. Flusso di elaborazione (query → risposta)

```
Domanda utente
   │
   ├─▶ Calcolo sicuro?  ──sì──▶ risposta aritmetica (tools.py)         [no LLM]
   │        │ no
   ├─▶ Intent detection (corso, argomento, ambiguità, memoria)
   │        │
   │        ├─ corso non riconosciuto ─▶ astensione "corso ignoto"     [no LLM]
   │        ├─ domanda ambigua ────────▶ richiesta di chiarimento      [no LLM]
   │        │
   ├─▶ Retrieval IBRIDO: vettoriale (MMR multi-query) + BM25, fusi con RRF, dedup
   ├─▶ Reranking euristico (corso/tipo/keyword/filename)
   ├─▶ Filtro metadata per corso
   ├─▶ [opz.] Reranker neurale cross-encoder sui top-N (default OFF, fallback euristico)
   ├─▶ top-k (MAX_CONTEXT_DOCUMENTS = 5)
   │        │
   │        ├─ nessuna evidenza ───────▶ astensione "non lo so"        [no LLM]
   │        │
   ├─▶ Preparazione fonti (dedup pagina/contenuto) + stima confidenza
   │        │
   ├─▶ Guard numerico TOLC applicabile?  ──sì──▶ risposta con soglie verificate [no LLM]
   │        │ no   (i 5 template "di prosa" sono OFF di default — cfr. valutazione)
   ├─▶ Evidence selection: passaggi brevi e mirati per fonte (FASE 5)
   ├─▶ Costruzione contesto + profilo risposta
   │        └─ [opz., default ON] consultabilità tesi + regolamento generale fra le fonti
   │           ─▶ hint "regola generale di Ateneo" che riduce la falsa astensione (FASE 14)
   ├─▶ Generazione (Ollama)                                            [LLM]
   ├─▶ Post-processing (rilevamento incertezza)
   ├─▶ Verifica citazioni: rimozione [F#] inventati + grounding (FASE 5)
   │        ├─ supporto lessicale; opz. rete semantica per le parafrasi (FASE 12, default OFF)
   │        └─ supporto debole ─▶ "reduce" (confidenza ↓ + nota)
   ├─▶ Se astensione: classificazione della causa (FASE 6)
   │        └─ fuori dominio vs fonte presente ma insufficiente (retrieval strength
   │           lessicale; opz. semantica per le parafrasi — FASE 13, default OFF)
   └─▶ Blocco fonti (citate o, in fallback, prime 3 "utilizzate")

Ogni ramo di astensione (corso ignoto, ambigua, retrieval debole, astensione del
modello) registra una **causa** in `trace.abstention_reason` (FASE 6). Il
post-processing si limita a *rilevare* l'incertezza (e ad abbassare la confidenza):
dal Ciclo 2 — FASE 10 **non riscrive** più l'astensione con un testo generico (un
vecchio caso speciale per «accesso» la rendeva invisibile a `is_abstention`,
scavalcando classificazione FASE 6 e blocco fonti onesto FASE 3); così il ramo «Se
astensione» è sempre raggiunto e l'esito resta coerente con la causa.
            │
            ▼
      Risposta + interpretazione + confidenza + fonti + RagTrace
```

I rami marcati `[no LLM]` non richiedono Ollama: sono coperti dai test offline e
costituiscono la parte più rapida e deterministica della pipeline.

## 4. Osservabilità: `RagTrace`

Ogni risposta popola un `RagTrace` con: domanda, corso e argomento rilevati, uso
della memoria, confidenza e motivo, profilo di risposta, modalità di retrieval e
scoring di fusione (RRF), reranker, evidence, grounding, causa di astensione, query
generate, fonti selezionate, documenti scartati dopo il reranking, regola
deterministica usata. È mostrato in sidebar e sotto la risposta (debug opzionale) ed
è **esportabile in JSON/Markdown** (`trace_export.py`, FASE 8).

## 5. Limiti architetturali noti

- I 5 template "di prosa" sono **disattivati di default**: il comportamento primario
  è RAG generativo; resta attivo solo il guard numerico TOLC (riattivabili con
  `UNILAW_PROSE_TEMPLATES=1`).
- Reranking e regole deterministiche residue **fortemente adattati ai 22 file attuali**
  e a frasi italiane specifiche; sensibili a nuovi documenti o nuovi corsi.
- Il riconoscimento dell'intento (`intent.py`) è a **liste di parole chiave**: preciso
  sulle formulazioni note, fragile sulle parafrasi. Dal Ciclo 2 — FASE 11 esiste un
  **affiancamento semantico opzionale** (`semantic_intent.py`, default OFF) che recupera
  corso/argomento sulle parafrasi (misurato 7/9 su sonde reali) senza sovrascrivere le
  keyword; le soglie del coseno sono provvisorie e restano da validare prima di un
  eventuale default ON.
- Conoscenza normativa centralizzata in `knowledge.py` con provenienza (FASE 7). Il
  prompt `ANSWER_STYLE_GUIDE` è stato in parte snellito: le fasce numeriche TOLC sono
  state rimosse (Ciclo 2 — FASE 8) perché già coperte dal guard deterministico
  (`knowledge.py`). Le regole di routing per corso, invece, sono state **valutate per la
  rimozione** (sono ridondanti col reranker + filtro metadata di `reranking.py` per la
  *selezione dei documenti*) ma l'A/B le ha **mantenute** (Ciclo 2 — FASE 9, risultato
  negativo): fanno anche da framing che riduce le false astensioni del modello 8B su
  domande borderline. Resta da snellire la prosa dei template deterministici.
- La verifica delle citazioni (FASE 5) è lessicale/euristica: rimuove i riferimenti
  inventati e segnala il supporto debole. Dal Ciclo 2 — FASE 12 è disponibile una **rete
  di recupero semantica opzionale** (default OFF) che recupera le frasi parafrasate per
  similarità di embedding (sonda: 7/7 a soglia calibrata 0,16) senza sostituire il
  lessicale; la separazione misurata è sottile, quindi la soglia resta da validare su un
  insieme più ampio prima di un eventuale default ON. Non è ancora inferenza semantica
  piena (NLI).
- La *retrieval strength* dell'astensione (FASE 6) è lessicale (overlap di token):
  penalizza le query parafrasate, che pur essendo in dominio non condividono token con
  la fonte. Dal Ciclo 2 — FASE 13 esiste una variante **semantica opzionale**
  (`semantic_retrieval_strength`, default OFF) con soglia **ricalibrata dai dati** (0,53,
  ≈ massimo margine 0,5286) e **validata 3/3 sull'held-out** q35/q36/q37; resta OFF
  perché in valore assoluto la forza semantica fuori dominio è alta (banda compressa), da
  validare su un insieme più ampio.
- **Falsa astensione su regolamento generale (q14): mitigata, non eliminata.** Quando la
  regola richiesta vive in un regolamento **generale di Ateneo** (non specifico del corso),
  il modello 8B tende ad astenersi cercando un dettaglio dedicato al corso pur avendo la
  regola nel contesto. Dal Ciclo 2 — FASE 14 il profilo di risposta, sulle domande di
  **consultabilità della tesi** con un regolamento generale fra le fonti
  (`has_general_tesi_regulation`), riceve un hint che **autorizza l'uso della regola
  generale** (default ON, gated): q14 passa da astensione a risposta corretta (behavior
  0,90 → 0,925), con effetto **isolato a q14**. È un intervento di *prompting*: riduce ma
  non garantisce l'esito (dipende dal modello) e non copre le altre false astensioni della
  famiglia (q16/q21/q29), di natura diversa.

## 6. Evoluzione architetturale

Le componenti della roadmap (FASI 3–8) sono **implementate**; di seguito il
riepilogo con l'esito misurato:

- ~~**Hybrid retrieval** (vettoriale + BM25, fusione RRF)~~ — implementato in FASE 3.
- ~~**Reranker cross-encoder** opzionale con fallback euristico~~ — implementato in
  FASE 4 (default OFF: su questo corpus tarato non migliora; vedi valutazione).
- ~~**Evidence selection** a passaggi brevi e **citation verification**~~ —
  implementati in FASE 5 (evidence selection: behavior 0,90→0,95, abstention →1,0).
- ~~**Layer di astensione** classificato e testato~~ — implementato in FASE 6
  (abstention_reason_accuracy 1,0 sul set etichettato).
- ~~**Riduzione dell'hard-coding**: conoscenza normativa in dati tracciabili~~ —
  implementata in FASE 7 (`knowledge.py`); template di prosa disattivati di default
  dopo ESP-07 (resta il guard numerico TOLC).
- ~~**UI e usabilità**: trace esportabile, stato pipeline, upload PDF~~ —
  implementati in FASE 8.

L'**intent detection semantica** è stata introdotta come affiancamento **opzionale**
alle keyword nel Ciclo 2 — FASE 11 (`semantic_intent.py`, default OFF): recupera
corso/argomento sulle parafrasi non coperte dalle keyword, in attesa di validazione
delle soglie su un set più ampio. Sullo stesso principio, il **grounding semantico delle
citazioni** è stato aggiunto come rete di recupero **opzionale** al grounding lessicale
nel Ciclo 2 — FASE 12 (`grounding_report(..., embedder)`, default OFF, soglia calibrata
0,16): recupera le frasi parafrasate riducendo i falsi "supporto debole", anch'esso in
attesa di validazione su un insieme più ampio. Lo stesso disegno è stato applicato alla
**retrieval strength dell'astensione** nel Ciclo 2 — FASE 13
(`semantic_retrieval_strength`, default OFF): la causa `fuori_dominio` vs
`evidenza_insufficiente` può essere decisa per similarità di embedding query↔fonte, con
soglia ricalibrata (0,53) e validata fuori campione (3/3 sull'held-out della FASE 6).

Nel Ciclo 2 — FASE 14 il **profilo di risposta** è stato esteso con un hint mirato (default
**ON**, gated): sulle domande di consultabilità della tesi con un regolamento generale di
Ateneo fra le fonti, autorizza l'uso della regola generale e riduce la falsa astensione del
caso q14 (behavior 0,90 → 0,925, effetto isolato a q14). A differenza degli affiancamenti
semantici opt-in delle FASI 11–13, è una correzione di prompting a basso rischio, perciò
attiva di default.

Direzioni di lavoro ancora aperte (cfr. `docs/relazione_unilaw_agent.md`, sezione
"Sviluppi futuri"): validazione/abilitazione di default dell'intent semantico;
verifica delle citazioni con inferenza semantica (NLI); reranker calibrato sul
dominio o fusione euristico/neurale; parsing tabellare dedicato; ampliamento del
corpus e del dataset di valutazione.
