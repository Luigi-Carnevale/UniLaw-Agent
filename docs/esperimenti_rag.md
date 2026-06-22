# Esperimenti RAG — UniLaw Agent

Registro degli esperimenti. Ogni voce riporta: obiettivo, configurazione,
risultato, confronto con la baseline e decisione tecnica. Gli esperimenti delle
fasi non ancora svolte sono marcati come **pianificati** e non riportano numeri
inventati.

---

## ESP-01 — Baseline del sistema (FASE 1)

- **Stato:** eseguito (2026-06-16).
- **Obiettivo:** misurare il comportamento attuale prima di ogni miglioramento.
- **Configurazione:** modello `llama3.1:8b` (Ollama, `temperature=0`), embeddings
  `paraphrase-multilingual-MiniLM-L12-v2`, indice ChromaDB con 1390 chunk da 22
  PDF; chunking 900/150 caratteri; `DEFAULT_K_RETRIEVAL=12`,
  `MAX_CONTEXT_DOCUMENTS=5`. Dataset: `eval/questions_baseline.jsonl` (20 domande).
- **Risultato:** behavior 0,90; course 1,0; topic 1,0; retrieval-hit 1,0;
  citation-hit 1,0; abstention 0,857. 62 test automatici verdi. Report:
  `eval/reports/baseline_20260616_224551.{json,md}`.
- **Confronto con baseline:** è la baseline stessa.
- **Decisione tecnica:** congelare questi valori come riferimento. I due
  fallimenti (q14 falsa astensione su evidenza pertinente; q18 over-answering)
  diventano obiettivi misurabili per le FASI 5–6. La retrieval-hit massima è
  attribuita all'adattamento del reranking ai file attuali e andrà ri-misurata su
  corpus ampliato.

---

## ESP-02 — Hybrid retrieval (vettoriale + BM25, RRF) — FASE 3

- **Stato:** eseguito (2026-06-16).
- **Obiettivo:** verificare se l'aggiunta di una componente lessicale (BM25) e la
  fusione RRF migliorano il recall **senza** peggiorare la qualità del retrieval.
- **Configurazione:** arm vettoriale (MMR multi-query, invariato) + arm BM25
  (`rank_bm25`, Okapi) sui 1390 chunk; fusione RRF (k=60); ordinamento finale con
  reranker euristico (invariato). Toggle `use_bm25` per l'A/B. Dataset baseline.
- **Risultato — ablation di retrieval (senza LLM, 13 domande con doc attesi):**

  | Modalità | retrieval-hit | rango del documento corretto |
  |---|---|---|
  | vettoriale (baseline) | 13/13 (1,0) | 1 per tutte |
  | hybrid (vettoriale+BM25) | 13/13 (1,0) | 1 per tutte |

  Rango del documento corretto: **0 peggiorati, 0 migliorati, 13 uguali**. Top-k
  identico in 9/13; nei restanti 4 (q02, q03, q09, q20) BM25 cambia i chunk di
  *supporto* alle posizioni 2–5 senza spostare il documento principale (rango 1).
- **Risultato — eval completa con LLM (20 domande):** rispetto alla baseline
  (ESP-01) cambia **una sola** domanda, q17 (no_answer): in hybrid BM25 recupera il
  piano di studi (informazioni sui semestri) e il modello risponde invece di
  astenersi → abstention 0,857 → 0,714, behavior 0,90 → 0,85; retrieval/citation
  e course/topic invariati a 1,0.
- **Interpretazione.** Su questo corpus, già "saturo" (hit 1,0 e documento corretto
  al rango 1), il top-line non può migliorare: il valore di BM25 è di **recall**
  (dimostrato dal test in cui BM25 fa emergere un documento che l'arm vettoriale non
  restituisce, e dal cambio dei chunk di supporto in 4 casi). L'unico effetto
  collaterale è su una domanda borderline negativa (q17), dove il maggior recall
  espone contenuto tangenziale: è un problema di astensione/generazione, non di
  retrieval.
- **Decisione tecnica.** Mantenere il retrieval ibrido come default (direzione
  architetturale corretta, recall superiore, nessuna regressione di retrieval) con
  il toggle `use_bm25=False` che riproduce esattamente la baseline. Il caso q17
  diventa obiettivo delle FASI 5–6. Robustezza da ri-misurare su corpus ampliato
  con documenti "trappola".

---

## ESP-03 — Reranker cross-encoder opzionale — FASE 4

- **Stato:** eseguito (2026-06-17).
- **Obiettivo:** confrontare il reranking euristico con un cross-encoder
  multilingua locale, misurando impatto su pertinenza, tempi e RAM/CPU.
- **Configurazione:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` applicato ai
  primi `RERANKER_TOP_N=15` candidati dopo euristica e filtro corso; via
  `sentence-transformers` (nessuna nuova dipendenza). Opt-in, default OFF.
- **Costi misurati (CPU):** caricamento ~77 s a freddo (~1–2 s a caldo, una tantum
  per processo); inferenza ~4 ms/coppia (~56 ms per 15 candidati); RAM picco
  ~800 MB; disco ~458 MB.
- **Risultato — ablation di retrieval (no LLM, 13 domande):**

  | Configurazione | retrieval-hit | rango del documento corretto |
  |---|---|---|
  | hybrid (euristico) | 13/13 (1,0) | 1 per tutte |
  | hybrid + cross-encoder | 13/13 (1,0) | peggiora in 6/13 (es. q01: 1→5) |

  Rango gold: **0 migliorati, 7 uguali, 6 peggiorati**. Il cross-encoder generico
  riordina per pertinenza semantica e **demolisce i priori di dominio** (boost
  dell'euristica sulle fonti autorevoli), abbassando il documento corretto.
- **Risultato — eval completa con LLM (20 domande):** behavior 0,90, abstention
  0,714, retrieval/citation 1,0. Rispetto all'hybrid (ESP-02): **q14 passa da
  astensione errata a risposta corretta** (il reranker riordina il contesto sulla
  consultabilità della tesi); q17/q18 invariati. Behavior aggregato uguale alla
  baseline euristica.
- **Interpretazione.** Trade-off, non guadagno netto: il cross-encoder generico
  aiuta un caso borderline di generazione (q14) ma peggiora il rango delle fonti
  autorevoli e non migliora le metriche aggregate su questo corpus già tarato.
- **Decisione tecnica.** Mantenerlo **opzionale e disattivato di default**, con
  l'euristica come ordinamento primario; documentarne i costi. Possibili evoluzioni:
  reranker calibrato sul dominio o fusione euristico⊕neurale. Report:
  `eval/reports/baseline_20260617_092819.*` e `retrieval_ablation_reranker.json`.

---

## ESP-04 — Evidence selection + verifica delle citazioni — FASE 5

- **Stato:** eseguito (2026-06-17).
- **Obiettivo:** fornire al modello evidenze più brevi e mirate e verificare le
  citazioni; ridurre falsa astensione e over-answering.
- **Configurazione:** evidence selection per frasi pertinenti (min 3 / max 6 frasi,
  ≤700 caratteri per fonte); verifica citazioni (rimozione [F#] inventati +
  grounding lessicale con politica "reduce"). Solo ramo LLM. A/B con `--no-evidence`.
- **Risultato (eval completa, 20 domande):**

  | Configurazione | behavior | abstention | retrieval/citation |
  |---|---|---|---|
  | hybrid senza evidence (`--no-evidence`) | 0,85 | 0,714 | 1,0 / 1,0 |
  | evidence + grounding (default) | **0,95** | **1,00** | 1,0 / 1,0 |
  | evidence + reranker neurale | 0,95 | 0,857 | 1,0 / 1,0 |

- **Attribuzione.** Il miglioramento è **interamente dell'evidence selection**: con
  `--no-evidence` i valori coincidono con l'hybrid (ESP-02). Concretamente, le
  evidenze focalizzate eliminano il contenuto tangenziale che induceva
  over-answering: **q17 e q18 ora si astengono correttamente** (abstention 7/7).
  Il grounding non cambia la classificazione (politica "reduce": nota + confidenza),
  ma toglie citazioni inventate e segnala supporto debole.
- **Caso q14 (verificato sulle fonti).** `regolamento-tesi-2023.pdf` p.2 contiene le
  regole di consultabilità (consultabile / embargo 24 mesi / non consultabile) ed è
  recuperato, ma il modello locale 8B si astiene comunque: falsa astensione residua
  dovuta al modello, non a retrieval o etichettatura.
- **Decisione tecnica.** Adottare evidence selection + grounding come **default**;
  mantenere il reranker OFF (non migliora, peggiora l'astensione). Report:
  `eval/reports/baseline_20260617_095838.*` (default), `_100348` (--no-evidence),
  `_100804` (evidence+reranker).

---

## ESP-05 — Astensione affidabile — FASE 6

- **Stato:** eseguito (2026-06-17).
- **Obiettivo:** classificare la causa dell'astensione e renderla affidabile e
  misurabile, distinguendo le 5 categorie richieste.
- **Configurazione:** layer `abstention.py` con `retrieval_strength` (overlap
  lessicale query↔fonti) e soglia `ABSTENTION_OOD_MAX_STRENGTH=0,37`. Nuova metrica
  `abstention_reason_accuracy` su 7 domande negative etichettate.
- **Calibrazione (evidence-based).** Strength misurate: q19=0,33 (fuori dominio),
  q17=0,40, q14=0,50, q15=0,57, q18=0,67. La sonda ha corretto l'assunzione iniziale
  su q18: il corpus cita il servizio mensa (bando borsa), quindi è "fonte presente
  ma insufficiente", non "fuori dominio".
- **Risultato (eval completa):** behavior 0,95, abstention 1,0,
  **abstention_reason_accuracy 1,0 (7/7)**: q10/q11 → fuori_dominio_corso, q12/q13 →
  ambigua, q17/q18 → evidenza_insufficiente, q19 → fuori_dominio. Nessuna regressione.
- **Interpretazione.** Il sistema ora dichiara *perché* si astiene, distinguendo in
  particolare "fuori dominio" da "fonte presente ma insufficiente" (caso q14/q17/q18).
  L'accuratezza 1,0 è sullo stesso set usato per calibrare la soglia: va ri-validata
  su negativi held-out.
- **Decisione tecnica.** Adottare il layer di astensione di default; politica
  conservativa (classificazione + nota, nessun blocco). Report:
  `eval/reports/baseline_20260617_104430.*`.

---

## ESP-06 — Riduzione hard-coding (regole → dati) — FASE 7

- **Stato:** eseguito (2026-06-17).
- **Obiettivo:** trasformare la conoscenza normativa codificata in dati strutturati
  e tracciabili, a comportamento invariato, e classificare tutte le regole.
- **Metodo:** verifica dei valori sulle fonti (TOLC 9/16; L-19 80/2h30/30-10-20-20);
  creazione di `knowledge.py` con provenienza; test che pinnano la **parità
  byte-per-byte** tra tabelle generate e stringhe legacy; refactor di `rules_tolc` e
  dei template per leggere dai dati.
- **Risultato:** 110 test verdi (8 nuovi su `knowledge`); tabelle e classificazione
  TOLC identiche; nessuna modifica al ramo LLM.
- **Osservazione su riproducibilità.** La riesecuzione dell'eval ha dato behavior
  0,90 e abstention 0,857 (vs 0,95/1,0 in FASE 6) per il solo caso q17 (ramo LLM)
  che oscilla a parità di pipeline: il refactor è invariante (verificato), quindi è
  **variabilità del modello locale** (`temperature=0` non perfettamente
  deterministico), non una regressione. Le domande deterministiche sono stabili.
- **Decisione tecnica.** Adottare `knowledge.py` come unica fonte di verità dei
  valori normativi; lasciare con motivazione e test le regole non trasformabili a
  basso rischio (prosa template, prompt, intent a keyword). Report:
  `eval/reports/baseline_20260617_120429.*`.

---

## ESP-07 — RAG puro vs template deterministici

- **Stato:** eseguito (2026-06-17).
- **Obiettivo:** misurare quanto il RAG "puro" (retrieval -> evidenze -> lettura del
  modello) regge **senza** i template deterministici, per capire se il sistema
  dipende dalle risposte codificate o sa davvero leggere e capire i documenti.
- **Metodo:** flag `use_deterministic` (env `UNILAW_DETERMINISTIC=0` / `--no-deterministic`)
  che disattiva i 6 template; confronto eval con e senza, stessa sessione.
- **Risultato:**

  | Metrica | con template | RAG puro (no template) |
  |---|---|---|
  | behavior_accuracy | 0,90 | 0,90 |
  | course / topic | 1,0 / 1,0 | 1,0 / 1,0 |
  | retrieval_hit | 1,0 | 1,0 |
  | citation_hit | 1,0 | 0,923 |
  | abstention / reason | 0,857 / 0,857 | 0,857 / 0,857 |

  Nel run "puro" nessun template è stato usato (verificato: `deterministic_rule`
  vuota su tutte le domande). I 4 casi prima gestiti da template
  (accesso Informatica, borsa graduatorie, borsa requisiti, Erasmus) sono stati
  risposti dal modello leggendo i documenti, **senza perdita di accuratezza
  aggregata**.
- **Unica differenza: q01** (accesso con punteggio 11). Con template: "Sì, ma con
  OFA", fonte `regolamento-di-accesso-informatical-31-.pdf`. In RAG puro il modello
  legge correttamente la regola ("Ris_Test non inferiore a 16 -> senza OFA") ma la
  *Risposta breve* resta "Sì, puoi immatricolarti" (meno netta sul "con OFA" per il
  valore 11) e cita `regolamento-l31-informatica.pdf` invece del regolamento di
  accesso (da cui citation_hit 0,923).
- **Interpretazione.** Il RAG puro **legge e comprende i documenti** in modo
  affidabile: su 20 domande l'accuratezza di comportamento e il retrieval non
  calano togliendo i template. Il valore residuo dei template è circoscritto ai casi
  di **esattezza numerica** (applicare una soglia a un valore specifico) e di
  **citazione della fonte canonica**, dove garantiscono precisione. Altrove sono
  ridondanti.
- **Decisione tecnica (APPLICATA).** Il sistema NON dipende dai template per
  funzionare: si è proceduto verso il RAG puro come comportamento primario. I 5
  template "di prosa" sono stati **disattivati di default** (`PROSE_TEMPLATES_ENABLED`),
  mantenendo solo il **guard numerico TOLC-I** per l'esattezza delle soglie e la fonte
  canonica. Verifica nella nuova configurazione predefinita: behavior 0,95,
  retrieval/citation 1,0, abstention/reason 1,0 — su 13 domande con risposta, solo 4
  (TOLC) usano il guard, le altre 9 sono RAG puro. Report:
  `eval/reports/baseline_20260617_153222.*` (con prosa), `_153822.*` (RAG puro),
  `_155116.*` (default attuale: guard TOLC, prosa off).

---

## ESP-08 — Togliere dal prompt le fasce numeriche TOLC — Ciclo 2 — FASE 8

- **Stato:** eseguito (2026-06-21).
- **Obiettivo:** verificare che rimuovere dal prompt `ANSWER_STYLE_GUIDE` le fasce
  numeriche TOLC-I (soglie 9/16), già presenti come unica fonte di verità in
  `knowledge.py` e applicate dal guard deterministico, **non cambi** il comportamento
  oltre la banda di rumore quantificata in FASE 7.
- **Metodo:** A/B nella stessa sessione (config predefinita, `llama3.1:8b`,
  `temperature=0`, 40 domande). A = prompt con le 4 righe delle fasce; B = prompt senza.
  Per separare il segnale dal rumore **cross-process** (FASE 7: within-session σ=0, ma
  ≤±1 domanda tra processi separati) sono stati raccolti **più campioni indipendenti**
  (processi separati, modello ricaricato): 2 con fasce, 3 senza.
- **Risultato:**

  | Campione | behavior | retrieval | citation | abstention / reason | q28 |
  |---|---|---|---|---|---|
  | A — con fasce | 0,90 | 0,963 | 0,963 | 0,923 / 0,923 | answer |
  | con fasce (FASE 7) | 0,90 | 1,00 | 1,00 | 0,923 / 0,923 | answer |
  | B — senza fasce | 0,875 | 1,00 | 1,00 | 0,923 / 0,923 | abstain |
  | B2 — senza fasce | 0,90 | 0,963 | 0,963 | 1,00 / 1,00 | answer |
  | B3 — senza fasce | 0,90 | 0,963 | 0,963 | 1,00 / 1,00 | answer |

  Report: `baseline_20260620_152206` (A), `_131229` (con fasce, FASE 7),
  `baseline_20260620_153742` (B), `baseline_20260621_112434` (B2),
  `baseline_20260621_113108` (B3).
- **Guard TOLC invariante (verificato).** Le 6 domande gestite dal guard (q01, q02, q03,
  q09, q26, q40) hanno risposta **byte-identica** fra A e B: il guard short-circuita prima
  del prompt, quindi la rimozione non può toccarle. È il cuore della fase: la correttezza
  numerica TOLC non dipende dal prompt.
- **L'unico delta è q28** (non-TOLC: posti per Economia Aziendale L-18, ramo LLM): in B
  passa da `answer` a `abstain`, ma **oscilla** sotto lo stesso prompt no-bands (è `answer`
  in B2/B3) → **rumore cross-process**, non effetto della modifica. La domanda è
  borderline: il documento riporta i posti della Magistrale LM-56, non della triennale
  L-18 chiesta, quindi l'astensione è semanticamente difendibile. Anche
  `retrieval`/`citation` (q24) e `abstention` (q18) oscillano allo stesso modo in
  **entrambe** le configurazioni.
- **Interpretazione.** behavior no-bands = {0,875; 0,90; 0,90} contro con-bands = {0,90;
  0,90}: il delta è ≤1 domanda, **entro la banda di rumore di FASE 7**. Le false astensioni
  **stabili** in tutte le run sono {q14, q21, q29} (famiglia nota, target FASE 14), non
  influenzate dalla modifica.
- **Decisione tecnica (APPLICATA).** Rimosse le 4 righe delle fasce numeriche da
  `ANSWER_STYLE_GUIDE`; le soglie 9/16 restano un'unica fonte di verità in `knowledge.py`.
  Mantenute le regole di *routing* per corso della stessa sezione (target FASE 9). 2 guard
  di non-regressione in `tests/test_knowledge.py` impediscono la reintroduzione della
  duplicazione (175 test verdi).

## ESP-09 — Togliere dal prompt il routing per corso — Ciclo 2 — FASE 9 (risultato negativo)

- **Stato:** eseguito (2026-06-21). **Esito: NEGATIVO → routing mantenuto nel prompt.**
- **Obiettivo:** verificare se rimuovere da `ANSWER_STYLE_GUIDE` le cinque sezioni «Regole
  specifiche per <corso/topic>» (instradamento al documento giusto) **non** cambi il
  comportamento, dato che l'instradamento *dei documenti* è già fatto sui metadata dal
  reranker + filtro per corso (`reranking.py`) e le regole di contenuto per topic sono già
  emesse dal profilo dinamico `_build_answer_profile`.
- **Metodo:** A/B nella **stessa sessione** (config predefinita, `llama3.1:8b`,
  `temperature=0`, 40 domande). Within-session σ=0 (FASE 7, riconfermato: 3 run B
  byte-identiche) ⇒ ogni flip è segnale. Per non confondere l'effetto con la deriva
  cross-sessione, la baseline «con routing» è stata **rieseguita in questa sessione**
  (A-fresh). Tre condizioni: A = con routing; B = senza routing + clausola di chiarimento
  generalizzata; B' = senza routing, rimozione pura.
- **Risultato:**

  | Config | behavior | course/topic | retrieval | citation | fallimenti |
  |---|---|---|---|---|---|
  | A-fresh — con routing | **0,90** | 1,0 / 1,0 | 1,0 | 1,0 | q14, q16, q21, q29 |
  | B — senza routing (+ chiarimento) | 0,875 | 1,0 / 1,0 | 0,963 | 0,963 | q18, q28, q29, q30, q31 |
  | B' — senza routing (rimoz. pura) | 0,85 | 1,0 / 1,0 | 1,0 | 0,963 | q18, q21, q28, q29, q30, q31 |

  Report: `baseline_20260621_222535` (A-fresh), `_185818`/`_190346`/`_195327` (B, identici),
  `baseline_20260621_214237` (B').
- **Per domanda (A → B, σ=0).** Rimuovere il routing **risolve 3** (q14, q16, q21) e
  **rompe 4** (q18, q28, q30, q31): q28/q30/q31 passano da `answer` ad `abstain` (false
  astensioni su domande answerable borderline, area borsa/economia), q18 passa da `abstain`
  ad `answer` (over-answering su una no-answer). q29 resta astensione in tutte le config
  (famiglia nota, target FASE 14). Le 6 domande del guard TOLC (q01/q02/q03/q09/q26/q40)
  restano `answer` ovunque (il guard short-circuita prima del prompt).
- **Interpretazione.** `course/topic/retrieval` invariati a 1,0 ⇒ il routing *dei documenti*
  è effettivamente ridondante col reranker (la selezione delle fonti non cambia). Ma il
  **behavior** regredisce: la prosa di routing fa anche da *framing di commit* di cui il
  modello 8B si avvale; senza, aumentano le false astensioni borderline. Net: 0,90 → 0,875
  (−1) → 0,85 (−2). La clausola di chiarimento (B vs B') recupera q21 ma non evita le
  regressioni borsa.
- **Decisione tecnica (APPLICATA).** **Non rimuovere il routing.** `ANSWER_STYLE_GUIDE`
  resta byte-identico; il guard `test_style_guide_keeps_course_routing_rules` impedisce la
  rimozione accidentale (175 test verdi). Come per il reranker neurale (FASE 4), il
  risultato negativo è riportato per onestà metodologica. Possibile estensione: rimozione
  *parziale* (solo le righe di puro instradamento documento, tenendo il framing
  anti-astensione), da validare con ulteriori A/B.

## ESP-10 — Rimozione del ramo di riscrittura dell'astensione (`_postprocess_answer`) — Ciclo 2 — FASE 10

- **Stato:** eseguito (2026-06-22). **Esito: nessuna regressione; modifica neutra sull'eval
  (caso speciale dormiente), valore strutturale (astensioni non più mascherate).**
- **Obiettivo:** verificare che rimuovere da `_postprocess_answer` il ramo che *riscriveva*
  l'astensione (per argomento "accesso" + fonti `doc_type ∈ {accesso, regolamento, altro}`)
  non peggiori il comportamento. Il ramo era ridondante con il layer di astensione (FASE 6)
  e, riscrivendo la risposta con un testo privo di marcatori, la rendeva invisibile a
  `is_abstention` → causa non classificata (FASE 6) e blocco fonti non onesto (FASE 3).
- **Metodo:** A/B (`llama3.1:8b`, `temperature=0`, 40 domande, **stessa sessione**, due
  processi). A = codice con il ramo (importato prima della modifica, confermato via timestamp
  del processo); B = codice senza il ramo.
- **Risultato:**

  | Config | behavior | course/topic | retrieval | citation | abstention / reason | report |
  |---|---|---|---|---|---|---|
  | A — **con** ramo (vecchio) | **0,875** | 1,0 / 1,0 | 1,0 | 1,0 | 0,923 / 0,923 | `baseline_20260622_094539` |
  | B — **senza** ramo (nuovo) | **0,875** | 1,0 / 1,0 | 1,0 | 1,0 | 1,0 / 1,0 | `baseline_20260622_095518` |

- **Attribuzione.** Il testo di riscrittura («potenzialmente rilevanti») compare **0 volte**
  nel report A → il ramo **non si è attivato in nessuna delle due run** (in A tutte le domande
  "accesso" del ramo LLM hanno risposto; in B il ramo è rimosso). Le sole differenze sono q18
  e q28, entrambe **oscillazione cross-process** del modello (testi generati divergenti):
  - **q18** (no-answer): A over-answering → `answer` (✗); B «Non lo so» → `abstain` (✓).
    topic=None ⇒ il ramo non poteva attivarsi comunque.
  - **q28** (accesso): A risposta borderline («non è specificato… consultare p.19») →
    `answer` (✓); B «Non lo so» → `abstain` (✗). Le fonti di q28 sono `doc_type=guida` ⇒ il
    ramo non si sarebbe attivato neppure sotto il codice vecchio.

  Fallimenti **stabili in entrambe**: q14, q16, q21, q29 (famiglia di false astensioni nota,
  target FASE 14). Il salto abstention/reason 0,923 → 1,0 è dovuto a q18, non alla modifica.
- **Interpretazione (onesta).** L'eval conferma la **non regressione** (0,875 = 0,875); la
  modifica è **neutra sul comportamento** perché il dataset corrente non esercita il caso
  speciale (servirebbe una domanda "accesso" del ramo LLM che si astenga con fonti
  `accesso/regolamento/altro`, es. L-16). A differenza della FASE 9, qui non si conclude da
  σ=0 within-session (A e B sono processi distinti, q18/q28 oscillano cross-process) ma
  dall'osservazione **diretta** che il ramo era dormiente. Il guadagno è **strutturale**: il
  «non lo so» non viene più riscritto, resta classificabile (FASE 6) e mostra il blocco fonti
  onesto (FASE 3).
- **Decisione tecnica (APPLICATA).** Ramo **rimosso**; rilevamento dell'incertezza unificato
  su `is_abstention` (niente lista locale duplicata); firma di `_postprocess_answer`
  semplificata. La trasparenza è coperta da 6 test unitari (`tests/test_postprocess.py`,
  181 test verdi). Q28 sotto il codice nuovo illustra la pipeline corretta: «Non lo so» →
  causa `evidenza_insufficiente` + blocco «Documenti consultati (nessuno utilizzato per la
  risposta)».

---

## ESP-11 — Intent detection semantica (opt-in) — Ciclo 2 — FASE 11

- **Stato:** eseguito (2026-06-22). **Esito: neutro sull'eval per costruzione (default OFF);
  guadagno di robustezza misurato sulle parafrasi (7/9 recuperate dal semantico).**
- **Obiettivo:** verificare che un classificatore d'intento per **similarità di embedding**,
  che **affianca** le keyword (riempie solo le caselle vuote, senza sovrascrivere), (a) non
  cambi il comportamento predefinito e (b) recuperi corso/argomento su formulazioni che le
  keyword non coprono. Il modulo `semantic_intent.py` riusa l'embedder già caricato per il
  retrieval (MiniLM multilingue); soglie del coseno **provvisorie** 0,5 (corso) / 0,45
  (argomento). Default **OFF** finché non validato.
- **Metodo.** Due misure, entrambe **senza Ollama** (solo il modello di embedding):
  1. *Neutralità sul dataset.* Per le 40 domande di `eval/questions_baseline.jsonl` si
     confronta l'intent a sole keyword con l'intent keyword + semantico (embedder reale).
  2. *Parafrasi.* 9 sonde costruite **senza** i token-keyword del loro intent (4 corsi + 5
     argomenti meno tesi-doppia), confrontando keyword-only vs keyword + semantico contro
     l'etichetta attesa.

- **Risultato 1 — dataset (neutralità per costruzione).**

  | Misura | Valore |
  |---|---|
  | Domande con corso/argomento **già coperti** dalle keyword | 40 / 40 |
  | Caselle vuote riempite dal semantico | 0 |
  | Sovrascritture di keyword da parte del semantico (override) | 0 |

  L'intent prodotto con il semantico attivo è **byte-identico** a quello a sole keyword su
  tutte e 40 le domande. Poiché domanda e intent restano identici e il modello è deterministico
  a `temperature=0` (σ=0 within-session, ESP/FASE 7), la generazione end-to-end è identica
  **per costruzione**: l'A/B generativo con Ollama non cambierebbe alcun verdetto (lo stesso
  argomento "caso non esercitato dal dataset" della FASE 10). Il dataset attuale, saturo di
  formulazioni canoniche, **non stressa** il layer semantico — come i distrattori non spiazzano
  il `retrieval_hit` (FASE 4).

- **Risultato 2 — parafrasi (modello reale, soglie 0,5 / 0,45).**

  | # | Campo | Sonda (senza keyword) | kw-only | + semantico | atteso | esito |
  |---|---|---|---|---|---|---|
  | 1 | argomento | «…aiuto economico…redditi bassi…» | None | borsa | borsa | ✓ recuperata |
  | 2 | argomento | «…un semestre a studiare all'estero…» | None | erasmus | erasmus | ✓ recuperata |
  | 3 | argomento | «…scritto conclusivo…letto da altri…» | None | None | tesi | sotto soglia |
  | 4 | argomento | «…prove…per entrare al primo anno…» | None | piano_studi | accesso | ✗ falso positivo |
  | 5 | argomento | «…quali materie…quanti crediti…» | None | piano_studi | piano_studi | ✓ recuperata |
  | 6 | corso | «…programmazione e sviluppo software…» | None | informatica | informatica | ✓ recuperata |
  | 7 | corso | «…mercati, aziende e finanza…» | None | economia | economia | ✓ recuperata |
  | 8 | corso | «…diventare un educatore…coi bambini…» | None | scienze_educazione | scienze_educazione | ✓ recuperata |
  | 9 | corso | «…gestire enti pubblici e uffici…» | None | amministrazione | amministrazione | ✓ recuperata |

  **7/9 recuperate** (tutte mancate dalle keyword), **1 sotto soglia** (tesi), **1 falso
  positivo** (accesso↔piano_studi su una formulazione ambigua «prove per entrare»).

- **Interpretazione (onesta).** Il semantico aggiunge robustezza reale sulle parafrasi, ma con
  due limiti documentati (una mancata e una mislabel): le soglie 0,5/0,45 sono **sensate ma non
  perfette**. Sul dataset corrente la funzione è neutra (non esercitata). Come per il reranker
  neurale (FASE 4), il guadagno non è ancora abbastanza solido da cambiare il default.
- **Decisione tecnica (APPLICATA).** Classificatore **implementato e disponibile** (`semantic_intent.py`,
  flag `UNILAW_SEMANTIC_INTENT` / `--semantic-intent` / `use_semantic_intent`), **default OFF**.
  Validato dai 17 test unitari (`tests/test_intent_detection.py`, embedder finto iniettato) +
  dalle 9 sonde con il modello reale. Validazione su un set di parafrasi più ampio e ricalibrazione
  delle soglie = lavoro successivo (in sinergia con l'ampliamento del dataset, FASE 4).

---

## ESP-12 — Grounding semantico delle citazioni (opt-in) — Ciclo 2 — FASE 12

- **Stato:** eseguito (2026-06-22). **Esito: il solo lessicale boccia tutte e 4 le parafrasi
  della sonda; il semantico le recupera (7/7 sulla sonda) con soglia calibrata 0,16; default OFF
  per separazione sottile.**
- **Obiettivo.** Il grounding delle citazioni (FASE 5) misura il supporto delle frasi citanti per
  **sovrapposizione lessicale** (overlap di token, soglia 0,18): una frase corretta ma
  **parafrasata** non condivide token con la fonte e viene segnata come "supporto debole",
  abbassando ingiustamente la confidenza. Si verifica se una **rete di recupero semantica**
  (similarità di embedding frase↔frase) che **affianca** il lessicale riduce questi falsi
  negativi senza recuperare frasi davvero estranee, e a quale soglia.
- **Metodo.** Tutto **senza Ollama** (solo il modello di embedding `paraphrase-multilingual-
  MiniLM-L12-v2`, lo stesso del retrieval). Per ogni coppia (frase citante, contenuto della
  fonte) si calcola la **massima** similarità del coseno fra la frase e le frasi della fonte
  (`split_sentences`). Due insiemi: **4 coppie parafrasi** (stesso significato, parole diverse,
  che il lessicale boccia) e **3 coppie estranee** (significato diverso). Le frasi sono costruite
  in modo che l'overlap lessicale sia nullo, così da isolare il contributo semantico.

- **Risultato — similarità misurate e calibrazione della soglia.**

  | Classe | Coseno massimo (frase↔frasi-fonte) |
  |---|---|
  | Parafrasi 1 — «elaborato finale consultabile dopo il titolo» ↔ «tesi depositata in biblioteca, richiedibile in lettura» | 0,263 |
  | Parafrasi 2 — «sostenere una prova di ammissione» ↔ «immatricolazione subordinata al test TOLC» | 0,605 |
  | Parafrasi 3 — «periodo di studio all'estero» ↔ «bando Erasmus, mobilità internazionale» | 0,271 |
  | Parafrasi 4 — «sostegno a chi ha reddito basso» ↔ «borsa di studio con ISEE entro soglia» | 0,194 |
  | Estranea 1 — «tassa annuale tremila euro» ↔ «bando Erasmus» | 0,017 |
  | Estranea 2 — «mensa a prezzo agevolato» ↔ «test TOLC» | 0,094 |
  | Estranea 3 — «capitale della Francia» ↔ «borsa di studio ISEE» | 0,129 |

  Le parafrasi stanno in **[0,194; 0,605]**, le estranee in **[0,017; 0,129]**. Il punto di
  **massimo margine** è la mediana fra le due classi: (0,129 + 0,194)/2 ≈ **0,16** (soglia
  adottata in `config.py`).

- **Risultato — recupero end-to-end a soglia 0,16** (`grounding_report`, ratio delle frasi
  citanti supportate).

  | Insieme | solo lessicale | + semantico (0,16) | esito |
  |---|---|---|---|
  | 4 parafrasi | 0,0 (4/4 bocciate) | 1,0 (4/4 recuperate) | ✓ recuperate |
  | 3 estranee | 0,0 | 0,0 (3/3 respinte) | ✓ nessun falso recupero |

  **7/7** sulla sonda: il semantico recupera esattamente le parafrasi e respinge le estranee,
  mentre il solo lessicale avrebbe declassato la confidenza su tutte e 4 le risposte parafrasate.

- **Interpretazione (onesta).** Il guadagno è reale ma la **separazione è sottile** (0,13 vs
  0,19) e la similarità di frase del MiniLM sulle parafrasi normative è **debole** (3 parafrasi su
  4 sotto 0,30). A soglia 0,16, su un insieme più ampio, sono attesi alcuni falsi recuperi: la
  sonda è troppo piccola (4+3) per garantire la generalizzazione. La rete semantica **si aggiunge**
  al lessicale (non può togliere un supporto già riconosciuto), quindi nel peggiore dei casi non
  declassa di meno del lessicale; il rischio è il *falso* recupero (confidenza troppo generosa).
- **Neutralità sull'eval (per costruzione).** Il grounding agisce solo sulla **confidenza**
  (politica "reduce": abbassa il livello e aggiunge una nota), non sul testo della risposta né
  sull'astensione. Le metriche-vetrina (behavior, citation_hit, abstention) **non dipendono dalla
  confidenza**, quindi l'A/B generativo non cambierebbe alcun verdetto: come nelle FASE 10/11, il
  valore è qualitativo (meno cautele non meritate), non un cambio di numeri.
- **Decisione tecnica (APPLICATA).** Rete semantica **implementata e disponibile**
  (`citations.py:grounding_report` con `embedder`; flag `UNILAW_SEMANTIC_GROUNDING` /
  `--semantic-grounding` / `use_semantic_grounding`), soglia **calibrata 0,16**, **default OFF**.
  Validata dai 10 test unitari (`tests/test_citations.py`, embedder finto iniettato) + dalla sonda
  con il modello reale. Validazione su un insieme più ampio e ricalibrazione = lavoro successivo
  (sinergia con l'ampliamento del dataset, FASE 4, e con la `retrieval_strength` semantica della
  FASE 13).

## ESP-13 — Retrieval strength semantica per l'astensione (opt-in) — Ciclo 2 — FASE 13

- **Stato:** eseguito (2026-06-22). **Esito: la forza semantica calibra a 0,5286 (≈ la 0,53 di
  config) e valida 3/3 sull'held-out, come la lessicale; i due segnali concordano sui 6 negativi
  threshold-relevant. Default OFF: banda semantica più compressa.**
- **Obiettivo.** La causa di astensione `fuori_dominio` vs `evidenza_insufficiente` (FASE 6) si
  decide sulla *retrieval strength* **lessicale** (overlap di token domanda↔fonte, soglia 0,37):
  una query **parafrasata** ma in dominio condivide pochi token con la fonte e rischia di essere
  marcata erroneamente `fuori_dominio`. Si verifica se una forza **semantica** (similarità di
  embedding query↔fonte) decide la stessa distinzione almeno altrettanto bene, ricalibrando la
  soglia sullo **stesso split** calibrazione/validazione della FASE 6 (§15 valutazione).
- **Metodo.** **Senza Ollama** (solo il modello di embedding `paraphrase-multilingual-
  MiniLM-L12-v2`, lo stesso del retrieval): l'harness `eval/abstention_threshold_validation.py`,
  esteso con una sezione semantica, misura per ogni negativo *threshold-relevant* la massima
  similarità del coseno fra domanda e contenuto delle fonti recuperate (retrieval deterministico,
  Chroma + BM25 + RRF), **calibra** la soglia a massimo margine sui soli storici (q17/q18/q19) e la
  **valida** sugli held-out q34–q39 (mai usati per calibrare). Report:
  `eval/reports/abstention_threshold_validation.*` (sezioni lessicale **e** semantica).

- **Risultato — strength semantiche e soglia calibrata.**

  | Insieme | id | causa attesa | strength semantica |
  |---|---|---|---|
  | calibrazione | q19 | `fuori_dominio` | 0,460 |
  | calibrazione | q17 | `evidenza_insufficiente` | 0,597 |
  | calibrazione | q18 | `evidenza_insufficiente` | 0,674 |
  | held-out | q35 | `fuori_dominio` | 0,324 |
  | held-out | q36 | `fuori_dominio` | 0,404 |
  | held-out | q37 | `evidenza_insufficiente` | 0,663 |

  Calibrazione a massimo margine = (0,460 + 0,597)/2 ≈ **0,5286** → soglia di config **0,53**
  (riproducibile dai dati, come la 0,37 lessicale rispetto alla sua 0,3667).

- **Risultato — accuratezza fuori campione.**

  | Forza | Insieme | Soglia | Accuratezza |
  |---|---|---|---|
  | lessicale | held-out | 0,37 (config) | **3/3** |
  | **semantica** | **held-out** | **0,53 (config)** | **3/3** |
  | semantica | held-out | 0,5286 (calibrata) | 3/3 |

  Le due forze **concordano** su tutti e 6 i negativi: la semantica non degrada la decisione.

- **Verifica unitaria del guadagno atteso.** Con un embedder finto (test offline), una frase
  parafrasata **senza token in comune** con la fonte («la tesi può essere visionata…» ↔
  «l'elaborato finale resta accessibile in biblioteca…») è classificata `evidenza_insufficiente`
  dalla forza **semantica** (in dominio) ma `fuori_dominio` dal solo **lessicale**: è la situazione
  che la fase intende correggere. Il dataset eval attuale non contiene un negativo simile (sui 6
  reali i token bastano), quindi il guadagno è dimostrato per costruzione/unitariamente, non
  ancora misurato end-to-end (estensione: domande parafrasate, sinergia con FASE 4).
- **Smoke test end-to-end (live, 1 domanda).** Con l'opzione ON e l'embedder reale, q19 («capitale
  della Francia») passa per `answer()` senza errori e assegna `fuori_dominio` (strength semantica
  0,46 < 0,53), **come** la lessicale: l'integrazione nel responder è confermata.
- **Neutralità sull'eval (per costruzione).** Con l'opzione OFF (default) l'embedder non viene
  costruito → classificazione byte-identica alla lessicale. Con l'opzione ON cambierebbe solo
  l'etichetta di causa sulle astensioni *threshold-relevant*, su cui i due segnali concordano →
  `abstention_reason_accuracy` invariata; behavior/citation/retrieval non dipendono dalla causa.
- **Interpretazione (onesta).** La separazione semantica **regge** ma è più **compressa**: in
  valore assoluto la forza dei *fuori dominio* è alta (0,32–0,46, il MiniLM dà similarità moderata
  anche off-topic) e il margine fra le classi è più stretto della lessicale. Insiemi piccoli
  (3 + 3): evidenza coerente, non garanzia. Come per reranker (FASE 4), intent semantico (FASE 11)
  e grounding semantico (FASE 12): **implementata e validata fuori campione, default OFF** in
  attesa di un set più ampio.
- **Decisione tecnica (APPLICATA).** Forza semantica **disponibile**
  (`abstention.semantic_retrieval_strength`; `classify_llm_abstention(..., embedder,
  semantic_ood_max_strength)`; flag `UNILAW_SEMANTIC_ABSTENTION` / `--semantic-abstention` /
  `use_semantic_abstention`), soglia **ricalibrata 0,53**, **default OFF**. Validata dai test
  unitari (`tests/test_abstention_reasons.py`, embedder finto) + dalla consistenza del report
  (`tests/test_abstention_threshold.py`) + dall'harness con il modello reale.

---

## ESP-14 — Mitigazione q14: hint "regola generale" sulla consultabilità della tesi — Ciclo 2 — FASE 14

- **Stato:** eseguito (2026-06-22). **Esito: positivo. q14 passa da falsa astensione a risposta
  corretta; effetto isolato a q14; behavior 0,90 → 0,925. Hint default ON.**
- **Obiettivo.** q14 («La tesi di Informatica L-31 è consultabile dopo la laurea?») è una falsa
  astensione **stabile** del Ciclo 1: la regola è nel corpus — un **regolamento generale di
  Ateneo** (`regolamento-tesi-2023.pdf`, `course_tag="generale"`) — ma il modello 8B si astiene
  cercando un dettaglio "specifico per Informatica L-31". Si verifica se un hint nel **profilo di
  risposta**, che autorizza l'uso della regola generale, riduce l'astensione **senza** dettare la
  risposta, e con quale blast radius sul dataset.
- **Diagnosi preliminare (no Ollama).** Il contesto effettivamente costruito per q14 (dopo
  retrieval + evidence selection) **contiene già** la regola completa: «a) Consultabile … b)
  Consultabile dopo un periodo (embargo) di 24 mesi. c) Non consultabile …» (pag. 3 del
  regolamento). Quindi è un limite di **generazione**, non di retrieval: la leva corretta è il
  prompt (l'"evidence boost", l'altra opzione di roadmap, è superfluo).

- **Risultato 1 — tuning del wording dell'hint (live, q14, contesto reale fisso).**

  | Variante hint | q14 | nota |
  |---|---|---|
  | V0 — nessun hint | astensione | baseline del ramo |
  | V1 — soft («le regole valgono per tutti i corsi, applica la generale») | astensione | non basta |
  | V2 — imperativo («il regolamento è UNICO per l'Ateneo; l'assenza del corso non è mancanza d'info; non dire "Non lo so"») | **risposta** | adottato |
  | V3 — prescrittivo («allora la risposta è SÌ…») | risposta | scartato: pre-dichiara la conclusione, rischio per il grounding |

  È stato adottato **V2** (generalizzato a "il corso indicato nella domanda"): sblocca la
  risposta senza mettere parole in bocca al modello.

- **Risultato 2 — A/B isolato su q14 (live, `llama3.1:8b`, `temperature=0`, via `answer()`).**

  | Config | q14 | `abstention_reason` | snippet |
  |---|---|---|---|
  | A — hint **OFF** (`--no-general-tesi-hint`) | astensione (✗) | `evidenza_insufficiente` | «Non lo so in base ai documenti disponibili.» |
  | B — hint **ON** (default) | **risposta** (✓) | — | «Risposta breve: Sì. … embargo di 24 mesi [F1] …» |

  **Riprodotto su due processi freschi** (stesso esito): l'effetto è del prompt, non rumore
  cross-process.

- **Risultato 3 — blast radius (offline, retrieval deterministico su 40 domande).** L'hint si
  attiva solo se la domanda è di **consultabilità** (`asks_tesi_consultazione`) **e** fra le
  fonti c'è un **regolamento generale sulla tesi** (`has_general_tesi_regulation`). Sul dataset
  **solo q14** soddisfa entrambe (q16/q19/q21/q22/q23/q32 recuperano un regolamento generale ma
  **non** sono di consultabilità). Quindi:

  | Metrica | prima (hint OFF) | dopo (hint ON) |
  |---|---|---|
  | behavior | 0,90 (36/40) | **0,925 (37/40)** |
  | course/topic · retrieval · citation | 1,0 | 1,0 (invariati) |

  Tutte le risposte non-q14 sono **identiche per costruzione** (il loro prompt non cambia): il
  +1 è esattamente q14.
- **Interpretazione (onesta).** Il guadagno è reale e **isolato**, ma dipende dal modello 8B:
  l'hint **riduce**, non garantisce, la falsa astensione (sul caso misurato è robusto su processi
  distinti). Lo scope è stretto per scelta (consultabilità + tesi generale); le altre false
  astensioni della famiglia (q16/q21/q29) hanno natura diversa (dettaglio specifico nel
  regolamento di corso, cross-course Erasmus) e restano aperte. Coerenza con i vincoli: l'hint
  spinge a rispondere **solo quando il contesto riporta le condizioni** — non introduce
  informazioni esterne.
- **Decisione tecnica (APPLICATA).** Hint **attivo di default** (`GENERAL_TESI_HINT_ENABLED`,
  env `UNILAW_GENERAL_TESI_HINT`, param `use_general_tesi_hint`, flag eval
  `--no-general-tesi-hint`): a differenza delle reti semantiche opt-in (FASE 11–13) è una
  correzione mirata, gated sulla situazione di retrieval e senza nuove dipendenze. Coperta da 11
  test unitari (`tests/test_answer_profile.py`, **230 test verdi**) + dall'A/B live.
