<!--
  Documento universitario principale di UniLaw Agent.
  Convertibile in PDF/DOCX (es. con pandoc; vedi Appendice, sezione 15).
  Logo di copertina: assets/logo_unisa.png (presente). Copertina completa di dati
  istituzionali (anno accademico, docente, data).
  Stato: versione finale, aggiornata a chiusura del Ciclo 2 (FASE 1-16 completate),
  2026-06-22. 230 test; comportamento predefinito behavior 0,925.
-->

<div align="center">

![Logo Università degli Studi di Salerno](../assets/logo_unisa.png){ width=180px }

# Università degli Studi di Salerno
## Dipartimento di Informatica

**Corso: Fondamenti di Intelligenza Artificiale**

---

# UniLaw Agent
### Assistente RAG locale per la consultazione di documenti universitari

---

**Autore:** Luigi Carnevale

**Matricola:** 0512119029

**Anno accademico:** 2025/2026

**Docente:** Fabio Palomba

**Data:** 24/06/2026

**Repository:** <https://github.com/Luigi-Carnevale/UniLaw-Agent>

</div>

<div style="page-break-after: always;"></div>

## Indice

1. Abstract
2. Introduzione
3. Requisiti del sistema
4. Architettura generale
5. Pipeline RAG
6. Intent detection e memoria controllata
7. Regole deterministiche
8. Miglioramenti implementati
9. Valutazione del sistema
10. Test automatici
11. Interfaccia utente
12. Limiti del sistema
13. Possibili sviluppi futuri
14. Conclusioni
15. Appendice
16. Glossario dei termini tecnici

<div style="page-break-after: always;"></div>

## 1. Abstract

La consultazione della documentazione universitaria ufficiale — regolamenti di
accesso, piani di studio, bandi, regolamenti della prova finale — è resa
difficile dalla lunghezza dei documenti e dalla loro distribuzione su numerosi
file PDF. UniLaw Agent affronta questo problema con un sistema di
*Retrieval-Augmented Generation* (RAG) **interamente locale**, che risponde a
domande in linguaggio naturale citando le fonti utilizzate.

L'obiettivo del lavoro è costruire un assistente che riduca il rischio di
risposte non fondate tramite il radicamento sulle fonti, l'astensione quando le
informazioni non sono disponibili e la richiesta di chiarimento quando la domanda
è ambigua. Il sistema integra retrieval vettoriale (ChromaDB), reranking,
generazione con un modello locale (Ollama) e citazione di file e pagina.

A partire da una versione iniziale funzionante ma fortemente basata su regole
codificate, il progetto è stato condotto come un percorso di ingegnerizzazione
**misurabile**, in fasi successive: una rete di **test automatici** (230 test) e una
**valutazione sperimentale** ripetibile hanno permesso di confrontare ogni
intervento con la baseline. Sono stati introdotti un **retrieval ibrido**
(vettoriale + BM25 con fusione RRF), un **reranker neurale opzionale**, un livello di
**evidence selection** e **verifica delle citazioni**, un **layer di astensione
classificata** e la trasformazione della conoscenza normativa codificata in **dati
tracciabili**. Rispetto alla baseline, l'evidence selection ha portato la
*behavior accuracy* da 0,90 a 0,95 e l'astensione corretta sui casi negativi al
100% **sul dataset di 20 domande**; l'astensione viene classificata per causa con
accuratezza piena sullo stesso set. Nel Ciclo 2 il dataset è stato **ampliato a 40
domande** (FASE 4): su questa baseline, più rappresentativa, la configurazione di allora
otteneva *behavior* 0,90 e astensione 0,923 (corso/argomento e retrieval/citazioni
restano 1,00), con il calo dovuto a una famiglia di false astensioni del modello
locale e non a una regressione del sistema (cfr. sezione 9). La mitigazione mirata della
FASE 14 (q14) porta poi il comportamento predefinito a **0,925 (37/40)**, a parità delle
altre metriche. Il documento riporta anche i limiti residui
(in particolare la variabilità del modello locale e la dimensione del corpus di
test) e cosa rende il sistema utilizzabile davvero: citazioni verificate, astensione
affidabile e tracciabilità esportabile.

## 2. Introduzione

Lo studente universitario deve spesso reperire un'informazione puntuale (una
soglia di punteggio, una scadenza, un requisito) all'interno di documenti
normativi estesi e scritti in linguaggio burocratico. La ricerca manuale è lenta
e soggetta a errori; un assistente conversazionale generico, d'altra parte,
rischia di inventare dettagli non presenti nei documenti.

Un sistema RAG è adatto a questo dominio perché vincola la generazione del modello
ai contenuti effettivamente recuperati da una base documentale controllata,
riducendo le risposte non fondate e rendendo possibile la citazione delle fonti.
La scelta **local-first** (modello, embeddings e indice in locale) è motivata da
ragioni di privacy e di riproducibilità: nessun documento lascia la macchina
dell'utente.

Il dominio applicativo è la documentazione di alcuni corsi di laurea
dell'Università degli Studi di Salerno: Informatica L-31, Scienze dell'Educazione
L-19, Scienze dell'Amministrazione e dell'Organizzazione L-16, oltre a documenti
trasversali (bando borsa di studio, bando Erasmus, guide alla tesi online).

## 3. Requisiti del sistema

**Requisiti funzionali.**
- Rispondere a domande in linguaggio naturale sui documenti indicizzati.
- Citare le fonti (file e, quando disponibile, pagina).
- Riconoscere corso e argomento della domanda.
- Astenersi quando l'informazione non è nei documenti.
- Chiedere chiarimento quando la domanda è ambigua (manca il corso).
- Gestire un corpus aggiornabile, con ricostruzione dell'indice al variare dei PDF.

**Requisiti non funzionali.**
- Esecuzione locale (privacy, nessuna dipendenza da servizi cloud).
- Riproducibilità (`temperature=0`, indice persistente, firma del corpus).
- Tracciabilità del comportamento (RagTrace, debug opzionale).
- Robustezza a PDF problematici (errori gestiti, non bloccanti).

**Vincoli.**
- Local-first: Ollama + ChromaDB + embeddings locali.
- Affidabilità e citazione delle fonti come requisiti di qualità prioritari.

## 4. Architettura generale

Il sistema è organizzato in moduli con responsabilità distinte (dettaglio in
`docs/architettura_rag.md`):

- `app_agent.py` — interfaccia Streamlit e gestione della sessione;
- `agent.py` — orchestratore RAG (`UniLawResponder`);
- `rag_types.py` — modello dati condiviso (FASE 2);
- `intent.py` — riconoscimento dell'intento e predicati (FASE 2);
- `rules_tolc.py` — regole deterministiche sul punteggio TOLC (FASE 2);
- `confidence.py` — stima dell'affidabilità (FASE 2);
- `citations.py` — gestione delle citazioni (FASE 2);
- `retrieval.py` — retrieval ibrido vettoriale + BM25 con fusione RRF (FASE 3);
- `reranking.py` — reranking euristico e filtro metadata (FASE 3);
- `neural_reranker.py` — reranker neurale opzionale (cross-encoder, FASE 4);
- `evidence.py` — evidence selection: passaggi brevi e mirati (FASE 5);
- `abstention.py` — layer di astensione: classifica la causa del "non lo so" (FASE 6);
- `knowledge.py` — conoscenza normativa strutturata e tracciabile (FASE 7);
- `trace_export.py` — esportazione del trace in JSON/Markdown (FASE 8);
- `database.py` — ingest PDF, chunking, embeddings, ChromaDB, manifest;
- `config.py` — costanti, prompt, configurazione, stile;
- `tools.py` — calcolo numerico sicuro.

Tecnologie: Streamlit (UI), Ollama con `llama3.1:8b` (generazione), embeddings
multilingua `paraphrase-multilingual-MiniLM-L12-v2`, ChromaDB persistente (indice)
e, opzionalmente, Redis come cache delle risposte del modello. Lo schema del
flusso applicativo è riportato nella sezione 5 e in `docs/architettura_rag.md`.

## 5. Pipeline RAG

1. **Caricamento documenti** dalla cartella `documenti/` (`PyPDFLoader`).
2. **Parsing e metadata**: per ogni documento vengono dedotti corso (`course_tag`)
   e tipo (`doc_type`) dal nome file, oltre a sorgente e pagina.
3. **Chunking** con `RecursiveCharacterTextSplitter` (900 caratteri, overlap 150).
4. **Embeddings** multilingua per ogni chunk.
5. **Indicizzazione** in ChromaDB persistente; una **firma SHA-256** del corpus
   (nome, dimensione, data, hash) determina quando ricostruire l'indice.
6. **Retrieval ibrido** (FASE 3): arm vettoriale multi-query con *Maximal Marginal
   Relevance* e arm lessicale **BM25** sui chunk indicizzati, fusi con
   *Reciprocal Rank Fusion* (RRF); deduplicazione e scoring tracciato.
7. **Reranking** euristico per corso, tipo documento, parole chiave e nome file.
8. **Filtro metadata per corso**; **(opzionale, FASE 4)** reranker neurale
   cross-encoder sui primi candidati, disattivato di default e con fallback
   euristico; selezione dei primi `MAX_CONTEXT_DOCUMENTS` (5).
9. **Evidence selection** (FASE 5): per ogni fonte si estraggono i passaggi più
   pertinenti alla domanda (contesto più breve e mirato).
10. **Costruzione del contesto** con etichette di citazione `[F1] … [F5]`.
11. **Generazione della risposta** con il modello locale, vincolata al contesto.
12. **Verifica delle citazioni** (FASE 5): si rimuovono i riferimenti `[F#]`
    inventati e si controlla il supporto lessicale delle frasi che citano; in caso
    di supporto debole si applica la politica "reduce" (confidenza più bassa + nota).
13. **Astensione classificata** (FASE 6): se mancano evidenze o il modello segnala
    incertezza, la risposta è di astensione e ne viene dichiarata la **causa**
    (corso fuori dominio, ambigua, retrieval debole, fuori dominio, fonte presente
    ma insufficiente).

Tutti gli stadi di questa pipeline sono implementati; il reranker neurale (stadio 8)
è opzionale e disattivato di default (cfr. sezioni 8 e 12). Le evoluzioni ancora
possibili sono descritte nella sezione 13.

## 6. Intent detection e memoria controllata

Il sistema riconosce **corso** e **argomento** della domanda tramite analisi
lessicale (`_infer_query_intent`). Distingue i corsi noti (Informatica L-31,
Scienze dell'Educazione L-19, Scienze dell'Amministrazione L-16, area
economico-statistica), rileva i **corsi non presenti nel corpus** (es. Medicina,
Giurisprudenza) per astenersi, e segnala le **domande ambigue** quando un
argomento è citato senza il corso.

La **memoria è a slot**, non a cronologia completa: vengono conservati solo
`ultimo corso` e `ultimo argomento`. Questo consente di gestire domande ellittiche
("E per la tesi?") riusando il corso precedente, **senza** contaminare il retrieval
con l'intera conversazione. Vantaggio rispetto alla cronologia completa: minore
rumore nel retrieval e comportamento più prevedibile. Limite attuale: il
riconoscimento è basato su liste di parole chiave, fragili verso formulazioni
nuove (cfr. sezione 12 e FASE 7 della roadmap).

## 7. Regole deterministiche

In FASE 7 tutte le regole deterministiche sono state **classificate** (non
eliminate ciecamente) e la conoscenza normativa codificata è stata **trasformata in
dati strutturati e tracciabili** (`knowledge.py`), con provenienza verificata sulle
fonti. Per ogni regola la strategia è una tra: *mantenere* (accettabile),
*trasformare in dato*, *mantenere con test* (euristica utile), *ridurre* (futuro).

| Regola | Modulo | Tipo | Strategia / stato |
|---|---|---|---|
| Calcolo aritmetico sicuro (AST) | `tools.py` | calcolo/parsing sicuro | **Mantenere**, testata; difetto del separatore di migliaia corretto nel Ciclo 2 (FASE 1) |
| Estrazione punteggio TOLC (regex) | `rules_tolc.py` | parsing sicuro | **Mantenere**, testata |
| Soglie TOLC 9/16 | `knowledge.py` | conoscenza normativa | **Trasformata in dato** (FASE 7): provenienza dal regolamento di accesso; `classify_tolc` legge da qui |
| Prova L-19 (80, 2h30, 30/10/20/20) | `knowledge.py` | conoscenza normativa | **Trasformata in dato** (FASE 7): provenienza dal bando di immatricolazione; tabella generata dai dati |
| Verifica citazioni / grounding | `citations.py` | controllo citazioni | **Mantenere**, testata (FASE 5) |
| Classificazione astensione (soglia strength) | `abstention.py` | controllo | **Mantenere**, testata (FASE 6); soglia da ri-validare |
| Reranking euristico (filename/keyword) | `reranking.py` | euristica di dominio | **Mantenere con test**: fallback del reranker neurale opzionale; tarato sul corpus |
| Filtro metadata per corso | `reranking.py` | euristica | **Mantenere con test** |
| Intent detection (liste di keyword) | `intent.py` | classificazione fragile | **Ridurre (futuro)**: coperta da test; sostituibile con approccio semantico |
| Guard numerico TOLC-I | `agent.py` | esattezza soglie | **Mantenere**: unico template attivo di default (verdetto + fonte canonica) |
| 5 template di prosa (OFA, tesi, Erasmus, borsa, L-19) | `agent.py` | risposte scritte a mano | **Ridotti**: disattivati di default (ESP-07: ridondanti col RAG); riattivabili con `UNILAW_PROSE_TEMPLATES=1` |
| Regole normative in `ANSWER_STYLE_GUIDE` | `config.py` | conoscenza nel prompt | **Ridurre (futuro)**: non modificata (alto rischio sul comportamento), duplica `knowledge`, da snellire |
| Metadata da filename (`course_tag`/`doc_type`) | `database.py` | euristica | **Mantenere con test**; futuro: anche da contenuto |

**Risultato chiave di FASE 7.** I valori normativi (9, 16, 80, 2h30, 30/10/20/20)
ora vivono in **un solo punto** (`knowledge.py`), ciascuno con la citazione testuale
della fonte da cui è stato verificato. La trasformazione è a **comportamento
invariato**: le tabelle generate dai dati coincidono byte-per-byte con quelle prima
codificate (verificato da test dedicati). Le regole che restano nel codice hanno una
motivazione esplicita e test dedicati, come richiesto.

## 8. Miglioramenti implementati

*Questa sezione viene aggiornata a ogni fase. Per ciascun intervento: problema
iniziale, soluzione, file modificati, impatto atteso, limiti residui.*

### FASE 1 — Test automatici e baseline (2026-06-16)
- **Problema iniziale.** Assenza di test e di una misura oggettiva della qualità:
  i miglioramenti non sarebbero stati verificabili e i refactoring rischiavano
  regressioni silenziose.
- **Soluzione.** Suite di 62 test offline (`tests/`); dataset di 20 domande
  etichettate (`eval/questions_baseline.jsonl`); harness di valutazione
  (`eval/run_eval.py`) con report JSON/Markdown; documentazione tecnica.
- **File creati.** `tests/*`, `pytest.ini`, `eval/questions_baseline.jsonl`,
  `eval/run_eval.py`, `docs/*`, `assets/README.md`, `requirements-dev.txt`.
  **File modificati.** `README.md`.
- **Impatto.** Fornisce la baseline quantitativa di riferimento e una rete di
  sicurezza contro le regressioni. Nessun cambiamento al comportamento dell'app.
- **Limiti residui.** I test sono di *characterization* (descrivono lo stato
  attuale, inclusi i difetti); l'harness classifica l'esito da marcatori testuali.

### FASE 2 — Refactoring leggero (2026-06-16)
- **Problema iniziale.** `agent.py` era un monolite di 2369 righe che concentrava
  tutte le responsabilità (intent, retrieval, reranking, regole, generazione,
  citazioni, confidenza), difficile da estendere e testare in modo isolato.
- **Soluzione.** Estrazione delle responsabilità **pure e già testate** in moduli
  dedicati (`rag_types`, `intent`, `rules_tolc`, `confidence`, `citations`); i
  metodi della classe sono diventati deleganti, preservando la superficie pubblica.
- **File creati.** `rag_types.py`, `intent.py`, `rules_tolc.py`, `confidence.py`,
  `citations.py`, `tests/test_confidence.py`. **File modificati.** `agent.py`
  (da 2369 a 1976 righe), `README.md`.
- **Impatto.** Nessun cambiamento di comportamento (verificato: 66 test verdi ed
  eval `--limit 9` identico alla baseline). Migliora manutenibilità e prepara le
  fasi successive su moduli isolati.
- **Limiti residui.** Retrieval, reranking, template deterministici e
  formattazione restano in `agent.py`: saranno estratti quando verranno riscritti
  (FASI 3/4/5/7), per evitare churn rischioso.

### FASE 3 — Retrieval ibrido (2026-06-16)
- **Problema iniziale.** Il recupero era solo semantico: i termini esatti, i codici
  e le sigle potevano non essere recuperati; mancava una misura della robustezza.
- **Soluzione.** Arm lessicale BM25 (`rank_bm25`) sui chunk già in ChromaDB,
  fuso con l'arm vettoriale tramite *Reciprocal Rank Fusion*; il reranker euristico
  resta l'ordinamento finale. Estratti `retrieval.py` e `reranking.py` da `agent.py`
  (sceso a 1438 righe). Toggle `use_bm25` per l'A/B; scoring di fusione nel trace.
- **File creati.** `retrieval.py`, `reranking.py`, `tests/test_retrieval.py`,
  `eval/retrieval_ablation.py`. **File modificati.** `agent.py`, `rag_types.py`,
  `app_agent.py`, `requirements.txt`, `requirements-mac.txt`.
- **Impatto (misurato).** Nessuna regressione di retrieval: nell'ablation il
  documento corretto resta al rango 1 in entrambe le modalità (0 peggiorati);
  BM25 aggiunge recall (chunk di supporto diversi in 4/13 casi, e un documento che
  l'arm vettoriale non restituisce nel test). Nell'eval completa cambia una sola
  domanda (q17), dove il maggior recall porta il modello a rispondere invece di
  astenersi (abstention 0,857 → 0,714): effetto di generazione, non di retrieval.
- **Limiti residui.** L'aumento di recall richiede evidence selection (FASE 5) e
  astensione robusta (FASE 6); la robustezza va ri-misurata su corpus ampliato.

### FASE 4 — Reranker neurale opzionale (2026-06-17)
- **Problema iniziale.** Verificare se un reranker neurale migliora la pertinenza
  rispetto all'euristica, senza imporre un modello pesante di default.
- **Soluzione.** `neural_reranker.py`: cross-encoder multilingua
  (`mmarco-mMiniLMv2-L12-H384-v1`, via `sentence-transformers` già presente),
  **opt-in** (`UNILAW_RERANKER=1` o toggle in sidebar), caricamento pigro,
  **fallback automatico** all'euristica, applicato ai primi 15 candidati.
- **File creati.** `neural_reranker.py`, `tests/test_neural_reranker.py`.
  **File modificati.** `config.py`, `agent.py`, `rag_types.py`, `app_agent.py`,
  `eval/run_eval.py`, `eval/retrieval_ablation.py`.
- **Costi misurati (CPU).** Caricamento ~77 s a freddo (una tantum), inferenza
  ~4 ms/coppia, RAM picco ~800 MB, disco ~458 MB.
- **Impatto (misurato).** Trade-off, non guadagno netto: nell'ablation il rango del
  documento corretto **peggiora in 6/13** casi (il modello generico ignora i priori
  di dominio dell'euristica); nell'eval completa risolve un caso borderline (q14,
  falsa astensione) ma il behavior aggregato resta 0,90 come la baseline euristica.
- **Decisione.** Reranker **disattivato di default**, euristica come ordinamento
  primario; mantenuto perché può aiutare su corpus più ampi o meno tarati.

### FASE 5 — Evidence selection e verifica delle citazioni (2026-06-17)
- **Problema iniziale.** Il modello riceveva chunk interi (rumorosi), il che
  favoriva l'over-answering (q17, q18) e non impediva citazioni inventate.
- **Soluzione.** `evidence.py`: selezione dei passaggi più pertinenti alla domanda
  (contesto più breve e mirato). `citations.py`: rimozione dei `[F#]` inventati e
  controllo di supporto lessicale, con politica "reduce" (confidenza + nota), mai
  blocco cieco. Applicate solo al ramo LLM; configurabili.
- **File creati.** `evidence.py`, `tests/test_evidence.py`. **File modificati.**
  `citations.py`, `config.py`, `agent.py`, `rag_types.py`, `app_agent.py`,
  `eval/run_eval.py`.
- **Impatto (misurato).** behavior **0,90 → 0,95**, abstention **0,857 → 1,00**;
  **q17 e q18 (over-answering) risolti**. L'A/B con `--no-evidence` attribuisce il
  guadagno **interamente all'evidence selection** (il grounding non cambia la
  classificazione). retrieval/citation restano 1,0.
- **Limiti residui.** q14 resta una falsa astensione: la fonte corretta è recuperata
  e contiene l'informazione (verificato), ma il modello locale 8B si astiene; è un
  limite di generazione, caso di studio per la FASE 6. Il grounding è lessicale
  (non semantico).

### FASE 6 — Astensione affidabile (2026-06-17)
- **Problema iniziale.** L'astensione esisteva ma non era classificata: il sistema
  diceva "non lo so" senza distinguere il motivo, e non era misurabile.
- **Soluzione.** `abstention.py`: tassonomia delle cause e classificazione
  automatica. La distinzione "fuori dominio" vs "fonte presente ma insufficiente"
  usa la *retrieval strength* (sovrapposizione lessicale domanda↔fonti), con soglia
  0,37 calibrata sui casi reali. La causa è registrata nel trace e spiegata
  all'utente; politica conservativa (nessun blocco).
- **File creati.** `abstention.py`, `tests/test_abstention_reasons.py`.
  **File modificati.** `config.py`, `agent.py`, `rag_types.py`, `app_agent.py`,
  `eval/questions_baseline.jsonl`, `eval/run_eval.py`.
- **Impatto (misurato).** Nuova metrica `abstention_reason_accuracy = 1,0` (7/7
  cause corrette); behavior 0,95 e abstention 1,0 invariati (nessuna regressione).
  La calibrazione ha corretto un'assunzione (q18 "mensa" è insufficienza di
  evidenza, non fuori dominio, perché il corpus cita il servizio mensa).
- **Limiti residui.** Soglia e accuratezza erano calibrate/misurate sullo stesso
  piccolo set; la ri-validazione su negativi *held-out* è stata poi eseguita nel
  Ciclo 2 — FASE 6 (cfr. sezione 9 e `docs/valutazione_rag.md` §15). La
  `retrieval_strength` resta lessicale (una versione semantica è un possibile
  sviluppo futuro).

### FASE 7 — Riduzione dell'hard-coding (2026-06-17)
- **Problema iniziale.** Conoscenza normativa (soglie TOLC 9/16, tabella prova L-19)
  **codificata e duplicata** in classificatore, template e prompt: difficile da
  verificare e mantenere, rischio di disallineamento se la normativa cambia.
- **Soluzione.** Classificazione di tutte le regole deterministiche (sezione 7) e
  creazione di `knowledge.py`: unica fonte di verità per i valori normativi, con
  **provenienza** (citazione testuale verificata sulle fonti). `classify_tolc` e le
  tabelle dei template leggono da qui.
- **Verifica sulle fonti.** I valori sono stati confermati sui PDF: TOLC "non
  inferiore a 16 / inferiore a 16 e non inferiore a 9" (regolamento di accesso) e
  "80 quesiti, 2 ore e 30 minuti, 30/10/20/20" (bando immatricolazione L-19).
- **File creati.** `knowledge.py`, `tests/test_knowledge.py`. **File modificati.**
  `rules_tolc.py` (delega), `agent.py` (tabelle dai dati).
- **Impatto (misurato).** Comportamento **invariato per costruzione**: le tabelle
  generate coincidono byte-per-byte con quelle codificate (test dedicati); 110 test
  verdi. L'eval ha riprodotto i risultati a meno della variabilità del modello locale
  (sezione 9).
- **Limiti residui.** La prosa dei template e le regole normative nel prompt
  (`ANSWER_STYLE_GUIDE`) non sono state toccate (rischio sul comportamento): restano
  da snellire. Intent detection a keyword ancora presente (coperta da test).

### FASE 8 — UI e usabilità reale (2026-06-17)
- **Problema iniziale.** L'osservabilità raccolta nel trace non era esportabile e lo
  stato della pipeline non era visibile; mancavano upload documenti e messaggi
  d'errore chiari.
- **Soluzione.** `trace_export.py` (esportazione del trace in JSON/Markdown);
  in `app_agent.py`: pulsanti di download del trace, blocco "PIPELINE" in sidebar,
  uploader di PDF con ricostruzione dell'indice, messaggi d'errore con cause e rimedi.
- **File creati.** `trace_export.py`, `tests/test_trace_export.py`. **File
  modificati.** `app_agent.py`.
- **Impatto.** 115 test verdi (+5); avvio headless verificato con `AppTest` senza
  eccezioni. Migliora trasparenza e usabilità senza appesantire l'interfaccia.
- **Limiti residui.** I pulsanti di export non hanno test d'integrazione UI (testata
  la sola generazione del contenuto).

### Decisione finale — RAG puro come comportamento primario (ESP-07)
- **Esperimento.** Misurato l'impatto della disattivazione dei template
  deterministici: senza i 5 template "di prosa" il RAG generativo mantiene la stessa
  accuratezza aggregata (sezione 9 e `docs/esperimenti_rag.md`, ESP-07), dimostrando
  che il sistema **non dipende** da risposte codificate per leggere e comprendere i
  documenti.
- **Decisione (applicata).** Template di prosa **disattivati di default**
  (riattivabili con `UNILAW_PROSE_TEMPLATES=1`); resta attivo solo il **guard
  numerico TOLC-I** per l'esattezza delle soglie e la citazione della fonte canonica.
- **File modificati.** `config.py` (`PROSE_TEMPLATES_ENABLED`), `agent.py`,
  `eval/run_eval.py` (flag `--prose-templates`).
- **Impatto (misurato).** Configurazione predefinita: behavior 0,95, retrieval /
  citation / abstention 1,0; su 13 domande con risposta solo 4 (TOLC) usano il guard.

## 9. Valutazione del sistema

Risultati **effettivamente eseguiti** con `llama3.1:8b` (dettaglio, attribuzione e
report in `docs/valutazione_rag.md` ed `docs/esperimenti_rag.md`). La tabella mostra
l'**evoluzione attraverso le fasi del Ciclo 1**, misurata sul dataset storico di **20
domande**; la baseline corrente sul dataset ampliato a 40 domande è riportata più sotto.

| Configurazione | behavior | abstention | retrieval / citation |
|---|---|---|---|
| FASE 1 — baseline euristica | 0,90 | 0,857 | 1,0 / 1,0 |
| FASE 3 — retrieval ibrido | 0,85 | 0,714 | 1,0 / 1,0 |
| FASE 4 — + reranker neurale | 0,90 | 0,714 | 1,0 / 1,0 |
| FASE 5 — + evidence selection | 0,95 | 1,00 | 1,0 / 1,0 |
| **FASE 6 — + astensione classificata** | **0,95** | **1,00** | 1,0 / 1,0 |

course_accuracy e topic_accuracy restano 1,00 in tutte le configurazioni. In FASE 6
si aggiunge la metrica `abstention_reason_accuracy = 1,0` (7/7 cause di astensione
classificate correttamente sul set etichettato).

**Baseline a 40 domande (Ciclo 2 — FASE 4) e configurazione finale.** Per rendere
le metriche più informative, il dataset è stato portato da 20 a **40 domande** (nuovi
corsi tra cui Economia, parafrasi, distrattori e sei negativi *held-out*). La baseline
della FASE 4 era: *behavior* **0,90** (36/40),
course/topic **1,00**, retrieval/citation **1,00** (27/27), astensione **0,923** (12/13),
causa di astensione **0,923** (12/13). Il calo da 0,95 a 0,90 **non è una regressione**
(il sistema è invariato): l'ampliamento ha esposto una *famiglia* di false astensioni
del modello locale (q14 storico più i nuovi q21 e q29, con la fonte corretta recuperata
e citata) oltre a un caso di over-answering (q18); tutti e sei i negativi *held-out* si
astengono con la causa corretta. Dettaglio in `docs/valutazione_rag.md` (§14). La
**configurazione predefinita finale** del Ciclo 2, dopo la mitigazione di q14 (FASE 14),
porta il *behavior* a **0,925 (37/40)** lasciando invariate le altre metriche.

**Consolidamento della misura (Ciclo 2 — FASE 5, 6 e 7).** Su questa baseline lo
scoring dell'eval è stato reso più robusto deducendo l'esito dai segnali
strutturati del trace anziché dai soli marcatori testuali: i verdetti coincidono
riga per riga con il report precedente (metriche identiche, maggiore stabilità al
variare del *wording*). La soglia di astensione (`ABSTENTION_OOD_MAX_STRENGTH = 0,37`)
è stata poi validata *fuori campione*: calibrata sui soli casi storici essa è
riprodotta dai dati (≈0,367) e, sui negativi *held-out* mai usati per la
calibrazione, classifica correttamente 3 casi su 3. È la ri-validazione che mancava
al layer di affidabilità; dettaglio in `docs/valutazione_rag.md` (§15). Infine la
variabilità del modello locale, finora dichiarata solo come «±1 domanda», è stata
**quantificata** (FASE 7) con l'opzione `--repeat N`: su 5 esecuzioni delle 40 domande
le metriche risultano identiche (**σ = 0** ovunque, nessuna domanda che oscilla), il che
mostra un determinismo *greedy* effettivo a `temperature=0` entro la sessione; la banda
storica «≤±1 domanda» è un effetto fra sessioni diverse (cfr. `docs/valutazione_rag.md`
§16). Con questa banda di rumore misurata è stato avviato il Blocco C dei refactor di
qualità: nel **Ciclo 2 — FASE 8** le fasce numeriche TOLC (soglie 9/16), già presenti
come unica fonte di verità in `knowledge.py` e applicate dal guard deterministico, sono
state **rimosse dalla loro duplicazione nel prompt** `ANSWER_STYLE_GUIDE`. L'A/B mostra
risposte byte-identiche sulle sei domande gestite dal guard (che corto-circuita prima
del prompt) e uno scostamento ≤1 domanda sui casi LLM, **entro la banda di rumore della
FASE 7**: la duplicazione è eliminata a comportamento invariato. Il totale dei test
automatici sale così a **175**.

Non tutti i refactor di pulizia, tuttavia, sono migliorativi. Nel **Ciclo 2 — FASE 9** si
è valutato di rimuovere dal prompt anche le regole di *routing* per corso, ridondanti per
il recupero dei documenti (già gestito dal reranker e dal filtro per metadata). L'A/B
nella stessa sessione ha però **misurato una regressione** (behavior 0,90 → 0,875, fino a
0,85 con la rimozione pura): pur essendo ridondante per il retrieval, quella prosa funge
anche da *framing* su cui il modello 8B si appoggia, e senza di essa aumentano le false
astensioni su domande answerable borderline. La decisione, riportata per onestà come il
risultato negativo del reranker neurale (FASE 4), è stata quindi di **mantenere il routing
nel prompt**; un test di guardia ne impedisce la rimozione accidentale (175 test
invariati).

Nel **Ciclo 2 — FASE 10** è stato ripulito un caso speciale di `_postprocess_answer`: un
ramo, attivo per argomento "accesso" e `doc_type ∈ {accesso, regolamento, altro}`,
**riscriveva** l'astensione con un testo generico, rendendola però invisibile a
`is_abstention` e scavalcando così la classificazione della causa (FASE 6) e il blocco
fonti onesto (FASE 3). Il ramo è stato rimosso: il post-processing ora rileva solo
l'incertezza (abbassa la confidenza) e restituisce il testo del modello senza riscriverlo.
L'A/B conferma un comportamento invariato (il ramo era di fatto dormiente sul dataset) e
copre il refactor con 6 test unitari, portando il totale a **181**.

Sempre nel Blocco C, il **Ciclo 2 — FASE 11** affronta un limite noto dell'intent
detection a keyword (fragile verso le parafrasi). Il nuovo modulo `semantic_intent.py`
classifica corso e argomento per **similarità di embedding** (riusando l'embedder già
caricato, senza nuove dipendenze) e **affianca** le keyword: riempie solo le caselle che
restano vuote, senza mai sovrascrivere un riconoscimento esistente. È **opt-in e
disattivato di default**, perché le soglie del coseno sono ancora provvisorie. Sul dataset
attuale le keyword coprono già corso e argomento su tutte le 40 domande, quindi
l'attivazione è **neutra per costruzione** sull'eval; su 9 sonde parafrasate prive dei
token-keyword il classificatore semantico ne recupera **7 su 9**. Resta disponibile e
configurabile in attesa di validare le soglie su un set più ampio; con i 17 test del modulo
il totale sale a **198**.

Con lo stesso spirito, il **Ciclo 2 — FASE 12** estende la verifica delle citazioni
(FASE 5), che misurava il supporto delle frasi citanti per **sola sovrapposizione
lessicale** e bocciava le frasi corrette ma parafrasate. Il grounding affianca ora un
**controllo per similarità di embedding**: le frasi respinte dal lessicale vengono
recuperate se abbastanza vicine, semanticamente, a una frase della fonte. La rete semantica
si aggiunge al lessicale (non lo sostituisce); con embedder assente il risultato è
byte-identico. La soglia del coseno **0,16** è calibrata dai dati (massimo margine) e su una
sonda separa **7/7** parafrasi/estranee, ma la separazione è sottile sulle parafrasi
normative: come per il reranker neurale (FASE 4) e l'intent semantico (FASE 11), la funzione
è **opt-in e disattivata di default**. Agendo solo sulla confidenza (politica «reduce»), è
neutra sulle metriche-vetrina; con i 10 test del modulo il totale sale a **208**.

A chiusura del Blocco C, il **Ciclo 2 — FASE 13** porta la stessa misura semantica nella
distinzione delle cause di astensione (FASE 6), che separava «fuori dominio» da «fonte
insufficiente» sulla *retrieval strength* **lessicale** (soglia 0,37) penalizzando le query
parafrasate. Una nuova forza per **similarità di embedding** query↔fonte affianca quella
lessicale; la soglia è stata **ri-calibrata sui dati** a **0,53** (riproducibile, massimo
margine) e **validata 3/3 sull'held-out** della FASE 6, con i due segnali concordi su tutti
i negativi. Come le altre estensioni semantiche è **opt-in e disattivata di default**
(separazione regge ma compressa, insiemi piccoli) ed è neutra sull'eval per costruzione; con
gli 11 test del modulo il totale sale a **219**.

Il Blocco D si apre con una **mitigazione mirata** (Ciclo 2 — FASE 14) del caso di studio
aperto fin dal Ciclo 1: la falsa astensione su **q14** («la tesi di Informatica L-31 è
consultabile dopo la laurea?»), dove la regola vive in un *regolamento generale* di Ateneo,
recuperato e completo nel contesto, ma il modello 8B si astiene cercando un dettaglio
specifico del corso. Un predicato puro riconosce il regolamento generale sulla tesi fra le
fonti e, nel ramo consultabilità, aggiunge al profilo di risposta un *hint* che **autorizza
l'uso della regola generale** senza dettare il contenuto. È una correzione a basso rischio,
**attiva di default** e gated sulla situazione di retrieval: con blast radius verificato
(su 40 domande **solo q14** attiva l'hint), q14 passa da astensione a risposta corretta e il
comportamento predefinito sale da 0,90 a **0,925 (37/40)**, a parità di tutte le altre
metriche. Con gli 11 test del modulo il totale dei test automatici raggiunge **230**. Restano
fuori scopo, come limite di generazione residuo, le altre false astensioni della stessa
famiglia (q16/q21/q29), non di consultabilità.

**Configurazione predefinita finale (ESP-07).** Una verifica sperimentale ha
mostrato che, disattivando i template "di prosa" che riscrivevano intere risposte,
il RAG generativo mantiene la stessa accuratezza leggendo direttamente i documenti.
La configurazione predefinita conserva quindi **solo il guard numerico TOLC-I** (per
l'esattezza delle soglie e la citazione della fonte canonica): su 13 domande con
risposta, solo 4 (punteggio TOLC) passano dal guard, le altre 9 sono RAG puro.

| Configurazione | behavior | citation | retrieval | abstention |
|---|---|---|---|---|
| RAG puro (nessuna regola) | 0,90 | 0,923 | 1,00 | 0,857 |
| **Predefinita (solo guard numerico TOLC)** | **0,95** | **1,00** | **1,00** | **1,00** |

Togliendo del tutto le regole, l'accuratezza aggregata e il retrieval **non calano**:
le domande prima gestite dai template (graduatorie e requisiti della borsa, Erasmus,
accesso L-19) vengono risposte correttamente leggendo le fonti. Report:
`eval/reports/baseline_20260617_155116.*` (predefinita), `_153822.*` (RAG puro).
Dettaglio in `docs/esperimenti_rag.md` (ESP-07) e `docs/valutazione_rag.md` (sez. 13).

**Nota di riproducibilità (rilevata in FASE 7).** Le metriche dipendenti dal modello
hanno una **variabilità run-to-run di ±1 domanda** su casi borderline: la stessa
pipeline (FASE 7 è un refactor a comportamento invariato, verificato byte-per-byte)
ha prodotto in una riesecuzione behavior 0,90 e abstention 0,857, per il solo caso
q17 ("orario delle lezioni") che oscilla tra astensione e risposta. Il modello locale
`llama3.1:8b` non è perfettamente deterministico nemmeno a `temperature=0`. I
componenti deterministici (template, classificatore, q01–q13) restano invece stabili.

**Esempi riusciti.** Classificazione del punteggio TOLC con fonte corretta e citata
(q01); robustezza a domanda mal formulata (q09); astensione su corsi non presenti
(q10/q11); con l'evidence selection (FASE 5) si astiene correttamente anche su
domande fuori corpus prima soggette a over-answering (q17, q18).

**Caso problematico residuo.** *Falsa astensione su evidenza buona* (q14): il
documento corretto è recuperato e **contiene** l'informazione (verificato:
`regolamento-tesi-2023.pdf` p.2), ma il modello locale 8B si astiene. È un limite
di generazione del modello, non di retrieval.

**Interpretazione.** L'intent e l'astensione sono solidi; la retrieval-hit massima
riflette l'adattamento al corpus dei 22 file e non va confusa con robustezza
generale. I miglioramenti più efficaci hanno agito sulla **generazione** (evidence
selection), non sul recupero, già saturo su questo corpus.

## 10. Test automatici

Comando: `python -m pytest` (230 test, offline). Copertura:

- **`tools.py`**: aritmetica, percentuali, rifiuto di input non ammessi; include
  la verifica del calcolo del separatore di migliaia (difetto corretto nel
  Ciclo 2 — FASE 1).
- **Classificazione TOLC**: confini delle fasce (8/9/15.9/16) ed estrazione del
  punteggio (incl. il caso "L-31" da non confondere con 31).
- **Intent detection**: corso, argomento, corso ignoto, ambiguità, memoria
  ellittica.
- **Metadata (`database.py`)**: `course_tag`, `doc_type`, firma del corpus.
- **Astensione**: corso ignoto, domanda ambigua, nessuna evidenza, domanda vuota.
- **Citazioni**: estrazione dei soli `[F#]` validi, blocco fonti, citazione
  ancorata a una fonte reale nel template TOLC.
- **Confidenza** (FASE 2): caso senza fonti, forte coerenza, fonte di corso
  diverso, e allineamento del metodo delegante.
- **Retrieval ibrido** (FASE 3): tokenizzazione, BM25, RRF (preservazione
  dell'ordine su un solo arm, ricompensa dei documenti in più arm), fusione
  vettoriale+lessicale.
- **Reranker neurale** (FASE 4): riordino per punteggio, conservazione della coda
  oltre i top-N, e fallback quando il modello non è disponibile (scorer iniettato,
  nessun download nei test).
- **Evidence selection e citazioni** (FASE 5): split in frasi, selezione dei
  passaggi pertinenti con minimo garantito e tetto di caratteri; rimozione delle
  citazioni inventate; grounding lessicale (frase supportata / non supportata).
- **Astensione classificata** (FASE 6): `retrieval_strength`, classificazione
  fuori dominio / fonte insufficiente / retrieval debole, e verifica che `answer()`
  registri la causa nel trace sui rami deterministici.
- **Esportazione trace** (FASE 8): JSON valido, sezioni Markdown, gestione del
  trace vuoto/None e dei placeholder per liste vuote.
- **Conoscenza normativa** (FASE 7): soglie e tabelle generate dai dati coincidono
  byte-per-byte con le versioni codificate; coerenza dei totali e presenza della
  provenienza.

Test su retrieval/reranking end-to-end sono affidati all'harness di valutazione
(sezione 9), poiché richiedono l'indice e il modello.

## 11. Interfaccia utente

La dashboard Streamlit presenta: un pannello di stato (knowledge base, cache,
documenti, modello), un blocco **PIPELINE** (retrieval ibrido, stato di reranker
neurale, evidence selection, verifica citazioni, astensione classificata), opzioni
di visualizzazione, comandi operativi e una chat. Il **debug RAG** mostra il
`RagTrace` completo: domanda, corso/argomento, memoria, confidenza e motivo,
modalità di retrieval e scoring di fusione (RRF), reranker, evidence, grounding,
causa di astensione, query generate, fonti selezionate e documenti scartati.

Funzionalità di usabilità introdotte in FASE 8:
- **Trace esportabile** in **JSON e Markdown** (pulsanti di download nel debug), per
  allegare l'osservabilità a una relazione o per l'analisi offline;
- **Upload di documenti PDF** dalla sidebar, con salvataggio in `documenti/` e
  ricostruzione dell'indice;
- **Messaggi di errore più comprensibili**, con cause comuni e rimedi (es. Ollama
  non attivo, knowledge base da ricostruire);
- **Indicazione dei limiti** della risposta resa esplicita nel testo (confidenza,
  causa di astensione, nota di grounding) e nel trace.

## 12. Limiti del sistema

- **Parsing PDF**: l'estrazione di tabelle può essere imperfetta.
- **Retrieval ibrido su corpus saturo**: su 22 file ben separati la metrica di
  hit è già massima; il valore del BM25 (recall sui termini esatti) andrà
  ri-misurato su un corpus più ampio.
- **Reranking euristico** e **regole deterministiche** adattati ai 22 file
  attuali e a frasi specifiche: sensibili a nuovi documenti o corsi.
- **Dipendenza dalla qualità del corpus**: la risposta è buona solo quanto i PDF
  indicizzati.
- **Modello locale**: capacità limitate rispetto ai modelli di grande scala; può
  astenersi pur avendo buone fonti o rispondere oltre la domanda.
- **Valutazione**: dataset ampliato a 40 domande nel Ciclo 2 (FASE 4), ancora
  contenuto e da estendere con documenti-trappola.
- **Verifica citazioni lessicale**: la verifica (FASE 5) rimuove i riferimenti
  inventati e segnala il supporto debole, ma è basata su sovrapposizione di termini,
  non su inferenza semantica (NLI).
- **Caso residuo q14**: la fonte corretta è recuperata e contiene l'informazione
  (verificato), ma il modello locale 8B si astiene comunque (fonte presente ma non
  sfruttata): è un limite di generazione del modello, non di retrieval.

## 13. Possibili sviluppi futuri

- **Hybrid retrieval** (vettoriale + BM25, fusione RRF): implementato in FASE 3;
  da validare su corpus ampliato con documenti "trappola".
- **Reranker cross-encoder** multilingua: implementato in FASE 4 (opzionale,
  default OFF). Evoluzioni: reranker calibrato sul dominio o fusione
  euristico⊕neurale, per non perdere i priori di dominio.
- **Evidence selection**: implementata in FASE 5 (default). Evoluzione: verifica
  delle citazioni con inferenza semantica (NLI) invece del solo grounding lessicale.
- **Layer di astensione** classificato: implementato in FASE 6. Evoluzione:
  *retrieval strength* semantica (non solo lessicale) e validazione su negativi
  held-out.
- **Riduzione dell'hard-coding**: avviata in FASE 7 (soglie TOLC e tabella L-19 →
  `knowledge.py`). Evoluzione: snellire la prosa dei template e le regole normative
  nel prompt; intent detection semantica al posto delle liste di keyword;
  estrazione automatica dei valori normativi dal testo delle fonti.
- **UI e usabilità** (FASE 8): trace esportabile in JSON/Markdown, stato della
  pipeline, upload PDF, errori più chiari. Evoluzione: visualizzazione delle pagine
  citate, gestione versioni dei documenti, esportazione delle conversazioni,
  gestione multiutente.
- **Parsing tabellare** dedicato; conversione automatica della relazione in
  PDF/DOCX.

## 14. Conclusioni

Il lavoro ha trasformato UniLaw Agent da prototipo funzionante ma poco verificabile
in un sistema RAG **misurabile, document-grounded e tracciabile**, attraverso un
percorso a fasi con verifica a ogni passo.

**Sintesi del lavoro.** Dopo un audit iniziale (FASE 0), sono stati introdotti: una
rete di test e una baseline misurata (FASE 1); una modularizzazione a comportamento
invariato (FASE 2); un retrieval ibrido vettoriale + BM25 con fusione RRF (FASE 3);
un reranker neurale opzionale, con fallback euristico, misurato e mantenuto OFF di
default perché non vantaggioso su questo corpus (FASE 4); l'evidence selection e la
verifica delle citazioni (FASE 5); un layer di astensione classificata (FASE 6); la
trasformazione della conoscenza normativa codificata in dati tracciabili (FASE 7);
e una UI più usabile con trace esportabile (FASE 8). Una verifica conclusiva (ESP-07)
ha infine mostrato che il sistema risponde altrettanto bene leggendo i documenti
senza i template di prosa, ora disattivati di default: resta attivo solo il guard
numerico delle soglie TOLC, dove l'esattezza è essenziale.

**Risultati raggiunti.** Sul dataset di 20 domande del Ciclo 1, l'evidence selection
ha migliorato la *behavior accuracy* da 0,90 a 0,95 e portato l'astensione corretta
sui casi negativi al 100%; il riconoscimento di corso e argomento è stabile al 100%;
le citazioni recuperate e citate corrispondono alle fonti attese. Con l'ampliamento a
40 domande (Ciclo 2 — FASE 4) la baseline corrente, più rappresentativa, è *behavior*
0,90 e astensione 0,923, con corso/argomento e retrieval/citazioni ancora a 1,00 (il
calo riflette un test più severo, non una regressione: cfr. sezione 9). Ogni
affermazione di miglioramento è supportata da una misura riproducibile e da un
confronto con la baseline.

**Cosa rende il sistema utilizzabile davvero.** Non la sola fluidità delle risposte,
ma: citazioni verificate (niente riferimenti inventati), astensione affidabile e
motivata quando le fonti non bastano, tracciabilità completa ed esportabile, e una
conoscenza normativa auditabile con provenienza dalle fonti.

**Limiti residui.** La qualità dipende dai PDF indicizzati; il modello locale 8B
introduce variabilità di ±1 domanda sulle metriche di generazione e talvolta si
astiene pur avendo fonti adeguate; la valutazione si basa su un dataset ancora
contenuto (40 domande dopo l'ampliamento del Ciclo 2) e su un corpus di 22 file su cui
le metriche di retrieval sono sature;
restano regole deterministiche (prosa dei template, prompt, intent a keyword) da
snellire ulteriormente.

**Valore didattico.** Il contributo principale è metodologico: aver reso esplicito e
documentato il ciclo *misura → intervento → ri-misura*, riportando con onestà anche
gli interventi che non hanno pagato (il reranker neurale) e le fonti di variabilità,
invece di dichiarare miglioramenti non provati.

## 15. Appendice

**Installazione e avvio.** Vedi `README.md`. In sintesi: installare le dipendenze,
scaricare il modello (`ollama pull llama3.1:8b`), avviare `ollama serve` e
`streamlit run app_agent.py`.

**Comandi di test e valutazione.**
```bash
python -m pytest                       # 230 test automatici (offline, no Ollama)
python eval/run_eval.py                # valutazione completa (Ollama attivo)
python eval/run_eval.py --no-evidence  # A/B: disattiva l'evidence selection
python eval/run_eval.py --reranker     # con reranker neurale
python eval/retrieval_ablation.py      # confronto retrieval vettoriale vs hybrid (no LLM)
```

**Struttura del progetto (moduli principali).**
```
app_agent.py     Interfaccia Streamlit
agent.py         Orchestrazione RAG (UniLawResponder)
intent.py        Riconoscimento intento + predicati
retrieval.py     Retrieval ibrido (vettoriale + BM25, RRF)
reranking.py     Reranking euristico e filtro metadata
neural_reranker.py  Reranker neurale opzionale (cross-encoder)
evidence.py      Evidence selection
citations.py     Gestione e verifica delle citazioni
abstention.py    Layer di astensione classificata
confidence.py    Stima di affidabilità
knowledge.py     Conoscenza normativa strutturata e tracciabile
trace_export.py  Esportazione del trace (JSON/Markdown)
rag_types.py     Modello dati (QueryIntent, RetrievedSource, RagTrace)
database.py      Ingest PDF, chunking, embeddings, ChromaDB, manifest
config.py        Costanti, prompt, configurazione, stile
tools.py         Calcolo numerico sicuro
documenti/  tests/  eval/  docs/  assets/
```

**Esempi di domande.** Cfr. sezione 9 e `eval/questions_baseline.jsonl`.

**Changelog sintetico.** Cfr. `docs/changelog_tecnico.md`; esperimenti in
`docs/esperimenti_rag.md`; metodologia e risultati in `docs/valutazione_rag.md`;
architettura in `docs/architettura_rag.md`.

**Conversione in PDF/DOCX.** Il documento è in Markdown e si converte con
[pandoc](https://pandoc.org). Lanciare dalla radice del progetto; `--resource-path`
permette di risolvere il logo di copertina (`../assets/logo_unisa.png`):
```bash
# DOCX (Word)
pandoc docs/relazione_unilaw_agent.md -o relazione_unilaw_agent.docx --toc --resource-path=docs
# PDF (richiede un motore LaTeX, es. tinytex/xelatex)
pandoc docs/relazione_unilaw_agent.md -o relazione_unilaw_agent.pdf --pdf-engine=xelatex --toc --resource-path=docs
```
Copertina completa (logo `assets/logo_unisa.png`, anno accademico, docente, data):
il documento è pronto per la conversione.

## 16. Glossario dei termini tecnici

Questa sezione raccoglie, in ordine alfabetico, i principali termini tecnici usati
nella relazione, con una definizione sintetica pensata anche per lettori non
specialisti. Dove utile, è indicato il significato che il termine assume
specificamente in UniLaw Agent.

**Allucinazione (*hallucination*).** In un sistema generativo, la produzione di
un'affermazione plausibile ma non fondata sui dati: un dettaglio "inventato" e non
presente nelle fonti. Ridurne il rischio è uno degli obiettivi centrali del progetto.

**Astensione.** Comportamento per cui il sistema rinuncia a rispondere quando non ha
fondamento sufficiente, dichiarando la causa (corso fuori dominio, domanda ambigua,
retrieval debole, fuori dominio, fonte presente ma insufficiente).

**Baseline.** Misura di riferimento iniziale delle prestazioni, congelata per
confrontare in modo oggettivo l'effetto di ogni intervento successivo.

**BM25.** Algoritmo classico di ricerca *lessicale* (per parole esatte) che assegna a
ciascun frammento un punteggio in base alla frequenza dei termini della domanda;
complementare alla ricerca semantica, utile per sigle, codici e denominazioni.

**Calcolo sicuro (AST).** Tecnica con cui il modulo di calcolo interpreta
un'espressione aritmetica come struttura sintattica (*Abstract Syntax Tree*),
ammettendo solo operatori in una lista bianca, senza eseguire codice arbitrario (a
differenza di `eval`).

**Chunk (frammento) / Chunking.** Porzione di testo (qui circa 900 caratteri) in cui
ogni PDF viene suddiviso per essere indicizzato e recuperato; il *chunking* è il
processo di suddivisione, con una sovrapposizione fra frammenti contigui per non
spezzare a metà un'informazione.

**ChromaDB.** Database vettoriale (*vector store*) che memorizza gli embedding dei
frammenti e ne consente la ricerca per similarità; nel progetto è persistente e
locale.

**Citazione [F#].** Riferimento, nella forma `[F1]`, `[F2]`…, con cui la risposta
indica la fonte (file e pagina) da cui proviene un'affermazione.

**Confidenza.** Stima euristica dell'affidabilità della risposta (alta/media/bassa),
con motivazione; viene abbassata quando le citazioni risultano poco supportate.

**Corpus.** L'insieme dei documenti (qui ventidue PDF ufficiali) su cui il sistema
può rispondere.

**Cross-encoder.** Modello neurale che valuta *congiuntamente* domanda e passaggio per
stimarne la pertinenza; è il cuore del reranker neurale opzionale. Più accurato ma
più costoso della sola ricerca vettoriale.

**Embedding.** Rappresentazione di un testo come vettore numerico in uno spazio in
cui testi semanticamente simili risultano vicini; abilita la ricerca per significato.

**Evidence selection (selezione delle evidenze).** Stadio che riduce ciascuna fonte
ai soli passaggi più pertinenti alla domanda, fornendo al modello un contesto più
breve e mirato, con meno rumore.

**Grounding (ancoraggio).** Verifica che le affermazioni della risposta siano
effettivamente supportate dalle fonti citate; nel progetto è misurato per
sovrapposizione lessicale.

**Hit-rate (*retrieval hit-rate*).** Percentuale di domande per cui il documento
atteso compare tra le fonti recuperate.

**Intent detection (riconoscimento dell'intento).** Analisi della domanda per dedurne
corso e argomento (e rilevare ambiguità o corsi non presenti nel corpus) prima del
recupero.

**LLM (*Large Language Model*, modello linguistico).** Modello di intelligenza
artificiale addestrato a produrre testo; qui `llama3.1:8b`, eseguito localmente
tramite Ollama.

**Local-first.** Scelta architetturale per cui dati, indice e modelli risiedono ed
eseguono sulla macchina dell'utente, senza servizi esterni (privacy e
riproducibilità).

**Manifest / Firma SHA-256.** Registro con un'impronta (*hash*) di ciascun PDF (nome,
dimensione, data, contenuto): confrontando la firma corrente con quella salvata,
l'indice viene ricostruito solo quando i documenti cambiano.

**Memoria a slot.** Memoria conversazionale minimale che conserva solo l'ultimo corso
e l'ultimo argomento, per gestire le domande ellittiche senza contaminare il recupero
con l'intera cronologia.

**Metadati (`course_tag`, `doc_type`).** Etichette inferite dal nome del file (corso e
tipo di documento) che guidano il filtro per corso e il reranking.

**MMR (*Maximal Marginal Relevance*).** Criterio di selezione dei risultati che
bilancia pertinenza e diversità, per evitare candidati troppo simili tra loro.

**NLI (*Natural Language Inference*).** Inferenza semantica che stabilisce se un testo
è implicato da un altro; indicata fra gli sviluppi futuri per una verifica delle
citazioni più profonda di quella lessicale.

**OFA (Obblighi Formativi Aggiuntivi).** Debiti formativi assegnati alla matricola il
cui punteggio di accesso ricade in una fascia intermedia; rilevanti per le soglie
TOLC di Informatica L-31.

**Ollama.** Ambiente di esecuzione che permette di far girare localmente modelli
linguistici; nel progetto ospita `llama3.1:8b`.

**Pipeline.** La sequenza ordinata di stadi che trasforma la domanda in risposta
(intento, recupero, reranking, evidenze, generazione, verifica, astensione).

**Provenienza.** Per un valore normativo, la citazione testuale del documento da cui è
stato verificato; rende la conoscenza *auditabile* (controllabile alla fonte).

**Query / Espansione della query.** La domanda usata per il recupero; l'*espansione*
ne genera più riformulazioni mirate per migliorare la copertura dei risultati.

**RAG (*Retrieval-Augmented Generation*).** Paradigma in cui il modello genera la
risposta a partire da documenti *recuperati* da una base di conoscenza, anziché dalla
sola memoria del modello: riduce le risposte non fondate e abilita la citazione delle
fonti.

**RagTrace (traccia).** Registro strutturato di tutto ciò che accade per una risposta
(intento, recupero, punteggi, reranker, evidenze, citazioni, causa di astensione,
fonti, documenti scartati); consultabile ed esportabile in JSON/Markdown.

**Recall.** Capacità del recupero di riportare i documenti rilevanti disponibili; il
retrieval ibrido mira ad aumentarla.

**Reranking / Reranker.** Riordino dei candidati recuperati per portare in cima i più
pertinenti; nel progetto un reranker *euristico* (priori di dominio) e, in opzione,
uno *neurale*.

**Retrieval (recupero).** La fase di ricerca, nell'indice, dei frammenti
potenzialmente utili a rispondere.

**RRF (*Reciprocal Rank Fusion*).** Tecnica che fonde due o più liste di risultati
sommando, per ciascun documento, 1/(k + rango) nelle diverse liste, senza dover
normalizzare punteggi eterogenei; qui unisce ricerca vettoriale e BM25.

**Streamlit.** Libreria Python per costruire interfacce web; realizza la chat e i
pannelli del sistema.

**Temperatura (*temperature*).** Parametro che regola la casualità del modello
generativo; a 0 le risposte sono il più possibile deterministiche e riproducibili.

**Template deterministico / Guard numerico.** Risposta prodotta da regole codificate
anziché dal modello. Nel sistema resta attivo per impostazione predefinita solo il
*guard numerico TOLC-I*, che applica le soglie di accesso con esattezza e cita la
fonte canonica.

**TOLC-I (*Test OnLine CISIA, Ingegneria/Informatica*).** Test di valutazione per
l'accesso ai corsi di area informatico-ingegneristica; il suo punteggio determina
l'ammissione a Informatica L-31, con o senza OFA.

**Token / Tokenizzazione.** Unità minime (parole o sottoparole) in cui un testo è
scomposto per l'elaborazione; la *tokenizzazione* è tale scomposizione (usata, ad
esempio, da BM25).

**Vector store / Indice vettoriale.** Struttura dati che conserva gli embedding e
supporta la ricerca per similarità; nel progetto è ChromaDB.
