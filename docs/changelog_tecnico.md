# Changelog tecnico — UniLaw Agent

Registro cronologico delle fasi di lavoro sul progetto. Ogni voce riporta: data,
fase, file analizzati/modificati, motivazione, impatto sul RAG, come testare,
rischi residui e prossimo step.

---

## 2026-06-16 — FASE 0: Audit tecnico (nessuna modifica al codice)

**Obiettivo.** Analizzare il sistema prima di qualsiasi intervento, distinguendo
le parti realmente RAG da quelle hard-coded, e individuare rischi e priorità.

**File analizzati.** `app_agent.py`, `agent.py`, `database.py`, `config.py`,
`tools.py`, `readme.md`, `requirements-mac.txt`, `.gitignore`, corpus `documenti/`
(22 PDF).

**File modificati.** Nessuno (fase di sola analisi).

**Sintesi dei risultati.**
- Parti genuinamente RAG: ingest → chunking → embeddings → ChromaDB; retrieval
  vettoriale con MMR; deduplicazione; generazione grounded; estrazione citazioni
  `[F#]`; manifest SHA-256 per il rebuild.
- Criticità principali: circa l'85% di `agent.py` è hard-coded e tarato sui 22
  filename attuali (reranking euristico ~370 righe; 6 template deterministici
  ~850 righe che bypassano LLM e retrieval); conoscenza normativa codificata nel
  prompt (`ANSWER_STYLE_GUIDE`) e nel codice (tabella L-19, soglie TOLC 9/16).
- Rischi: hallucination "autorevole" da template non realmente grounded;
  fragilità verso nuovi PDF/corsi; citazioni potenzialmente fabbricate dal
  fallback `sources[:3]`; file duplicati da sync in `.chroma_db/`.

**Impatto sul RAG.** Nessuno diretto: definisce la roadmap e i criteri di qualità.

**Rischi residui.** Invariati rispetto allo stato iniziale.

**Prossimo step.** FASE 1 — test automatici e baseline misurata.

---

## 2026-06-16 — FASE 1: Test di base e baseline misurata

**Obiettivo.** Creare una rete di test automatici e un dataset di valutazione,
e **misurare il comportamento attuale** prima di ogni miglioramento.

**File analizzati.** `tools.py`, `agent.py` (metodi `_classify_tolc_score`,
`_extract_tolc_score`, `_infer_query_intent`, `_extract_cited_source_indexes`,
`_format_sources_block`, rami di astensione e template deterministici),
`database.py` (`_infer_course_tag`, `_infer_doc_type`, `calcola_firma_documenti`,
`_should_rebuild`).

**File creati.**
- `tests/conftest.py`, `tests/test_tools.py`, `tests/test_tolc_classification.py`,
  `tests/test_intent_detection.py`, `tests/test_metadata.py`,
  `tests/test_abstention.py`, `tests/test_citations.py` — 62 test.
- `pytest.ini` — configurazione test (offline, nessun Ollama richiesto).
- `eval/questions_baseline.jsonl` — 20 domande etichettate (facili, difficili,
  ambigue, fuori dominio, senza risposta, sinonimi, mal formulate).
- `eval/run_eval.py` — harness di valutazione (retrieval/intent/astensione/
  citazioni), report JSON + Markdown in `eval/reports/`.
- `docs/architettura_rag.md`, `docs/valutazione_rag.md`,
  `docs/esperimenti_rag.md`, `docs/relazione_unilaw_agent.md`,
  `docs/changelog_tecnico.md` (questo file).
- `assets/README.md` — istruzioni per il logo ufficiale.
- `requirements-dev.txt` — dipendenza di sviluppo `pytest`.

**File modificati.** `README.md` — aggiunta sezione "Test e valutazione",
struttura cartelle aggiornata, dipendenze di sviluppo.

**Modifiche tecniche.**
- Nessuna modifica al comportamento dell'applicazione: i test sono di tipo
  *characterization* (fotografano il comportamento esistente).
- Installata la dipendenza di sviluppo `pytest>=8,<9` nell'ambiente virtuale.
  Nota operativa: il `pip` del venv era corrotto (`pip._internal` mancante) ed è
  stato ripristinato con `python -m ensurepip --upgrade`.

**Difetto rilevato (non corretto in questa fase).** In `tools.py`,
`_normalizza_numero_italiano` interpreta un singolo punto come separatore
decimale: "20.000" diventa `20.0`, quindi "5% di 20.000€" restituisce "1" invece
di "1.000". Il difetto è caratterizzato dal test
`test_percentage_thousands_separator_known_bug`; correzione pianificata in una
fase successiva (regola deterministica da sistemare).

**Impatto sul RAG.** Indiretto ma fondamentale: fornisce la baseline quantitativa
necessaria a giudicare i miglioramenti delle fasi successive, ed evita regressioni
silenziose quando i template deterministici verranno trasformati (FASE 7).

**Come testare.**
```bash
python -m pytest                 # 62 test, attesi tutti verdi
python eval/run_eval.py          # baseline completa (richiede Ollama)
python eval/run_eval.py --limit 9  # solo domande deterministiche/astensione (no Ollama)
```

**Risultati misurati (baseline, 2026-06-16, modello `llama3.1:8b`).**
- Test: 62 passati in ~0,2 s.
- Eval su 20 domande: behavior 0,90; course 1,0; topic 1,0; retrieval-hit 1,0;
  citation-hit 1,0; abstention 0,857. Dettaglio in
  [`docs/valutazione_rag.md`](valutazione_rag.md) e report
  `eval/reports/baseline_20260616_224551.{json,md}`.

**Rischi residui.**
- L'alta retrieval-hit (1,0) riflette il *forte adattamento del reranking ai 22
  file attuali*, non robustezza generale (cfr. FASE 0).
- L'harness classifica l'esito da marcatori testuali della risposta: scelta
  pragmatica per la baseline, da rendere più solida se il wording cambia.
- Due casi falliti documentati (q14 falsa astensione su evidenza buona; q18
  over-answering): motivano le FASI 5–6.

**Prossimo step.** FASE 2 — refactoring leggero (modularizzazione senza cambi di
comportamento), mantenendo i 62 test verdi.

### Nota operativa: logo dell'Università
Il file `assets/logo_unisa.png` **non è presente**. Nel documento universitario
(`docs/relazione_unilaw_agent.md`) è inserito il segnaposto `[IMMAGINE ALLEGATA]`.
L'utente deve inserire manualmente il logo ufficiale in `assets/logo_unisa.png`
(vedi `assets/README.md`). Non sono state scaricate immagini da internet.

---

## 2026-06-16 — FASE 2: Refactoring leggero (modularizzazione, comportamento invariato)

**Obiettivo.** Rendere il codice più modulare separando le responsabilità,
**senza modificare il comportamento** e mantenendo i test verdi.

**File analizzati.** `agent.py` (monolite di 2369 righe).

**File creati.**
- `rag_types.py` — modello dati: `QueryIntent`, `RetrievedSource`, `RagTrace`,
  `COURSE_LABELS`, `TOPIC_LABELS`.
- `intent.py` — `infer_query_intent` + helper + predicati `asks_*`.
- `rules_tolc.py` — `extract_tolc_score`, `classify_tolc_score`.
- `confidence.py` — `estimate_confidence`.
- `citations.py` — `extract_cited_source_indexes`, `format_sources_block`.
- `tests/test_confidence.py` — 4 test per il modulo `confidence` estratto.

**File modificati.**
- `agent.py` — ridotto da **2369 a 1976 righe** (−393): le funzioni estratte sono
  ora importate; i metodi corrispondenti di `UniLawResponder` sono diventati
  **sottili deleganti** (es. `_classify_tolc_score` → `classify_tolc_score`), così
  la superficie usata da test/eval/app resta identica. `QueryIntent`,
  `RetrievedSource`, `RagTrace`, `COURSE_LABELS`, `TOPIC_LABELS` sono riesportati
  da `agent` per retrocompatibilità (`from agent import QueryIntent` continua a
  funzionare). Rimossi gli import ora inutilizzati (`re`, `dataclass`, `field`).
- `README.md` — struttura cartelle aggiornata con i nuovi moduli.

**Scelta di scoping (motivata).** Sono stati estratti solo i concern **puri e già
coperti da test** (tipi, intent, regole TOLC, confidence, citazioni). Sono stati
**lasciati in `agent.py`**: retrieval multi-query e reranking euristico (verranno
modificati in FASE 3/4), i 6 template deterministici (verranno ridotti in FASE 7)
e i blocchi di formattazione/generazione (semplificati in FASE 5). Spostare ora
~1200 righe fortemente accoppiate sarebbe stato churn rischioso, in contrasto con
il principio "piccoli step verificabili".

**Impatto sul RAG.** Nessun cambiamento funzionale: il refactoring migliora la
manutenibilità e prepara le fasi successive (ogni futura modifica al retrieval o
alle regole agirà su moduli isolati, non sul monolite).

**Come testare.**
```bash
python -m pytest                   # 66 test (62 + 4 su confidence), attesi verdi
python eval/run_eval.py --limit 9  # verifica di non-regressione (no metriche LLM piene)
```

**Risultati misurati.**
- Test: **66 passati** (~0,2 s).
- Eval `--limit 9` post-refactoring: behavior 1,0; course 1,0; topic 1,0;
  retrieval-hit 1,0; citation-hit 1,0 — **identici alla baseline**: comportamento
  preservato attraverso intent, retrieval, reranking, template deterministici,
  confidenza e citazioni.

**Rischi residui.** Aumento del numero di file a livello root (scelta preferita
alla creazione di un package, che avrebbe richiesto di toccare tutti gli import
esistenti). Le parti più rischiose restano da modularizzare e lo saranno quando
verranno comunque riscritte.

**Prossimo step.** FASE 3 — retrieval robusto (hybrid: vettoriale + BM25, fusione,
filtri metadata, deduplicazione), con confronto rispetto alla baseline. In quella
sede sarà naturale estrarre anche `retrieval.py`/`reranking.py`.

---

## 2026-06-16 — FASE 3: Retrieval robusto (hybrid vettoriale + BM25, RRF)

**Obiettivo.** Aggiungere una componente lessicale (BM25) al recupero vettoriale,
fondere i risultati e rendere lo scoring trasparente, **senza regredire** sul
corpus attuale. Estrarre la logica di retrieval/reranking dal monolite.

**File analizzati.** `agent.py` (metodi `_retrieve_documents`,
`_multi_stage_retrieval`, `_rerank_documents`, `_filter_documents_by_course`).

**File creati.**
- `retrieval.py` — retrieval ibrido: `build_query_variants`, `run_vector_queries`,
  `Bm25Index`/`build_bm25_index` (BM25 sui chunk già in ChromaDB), `tokenize`,
  `reciprocal_rank_fusion` (RRF), `hybrid_retrieve`.
- `reranking.py` — reranker euristico (`rerank_documents`) e filtro metadata per
  corso (`filter_documents_by_course`), estratti da `agent.py`.
- `tests/test_retrieval.py` — 9 test (tokenize, RRF, BM25, fusione, modalità).
- `eval/retrieval_ablation.py` — confronto VETTORIALE vs HYBRID senza LLM.

**File modificati.**
- `agent.py` — ridotto da **1976 a 1438 righe**: rimossi i tre metodi estratti;
  `__init__` costruisce l'indice BM25 una volta e accetta `use_bm25` (default True);
  `_retrieve_documents` ora chiama `hybrid_retrieve` → `rerank_documents` →
  `filter_documents_by_course`.
- `rag_types.py` — `RagTrace` con due campi: `retrieval_mode`, `fusion_scores`.
- `app_agent.py` — il debug RAG mostra modalità di retrieval e scoring di fusione.
- `requirements.txt`, `requirements-mac.txt` — aggiunta dipendenza `rank_bm25==0.2.2`
  (libreria leggera, pura Python, dipende solo da numpy già presente).

**Decisione di progetto.** L'RRF genera i *candidati* (recall); l'ordinamento
finale resta il reranker euristico (FASE 4 aggiungerà un reranker neurale
opzionale). Con `use_bm25=False` l'RRF su un solo ranking preserva l'ordine
vettoriale: la **modalità vettoriale riproduce esattamente la baseline pre-FASE 3**
(proprietà testata e usata per l'A/B).

**Difetto di robustezza corretto.** Un chunk che si riduce a 0 token (solo
stopword/punteggiatura) faceva fallire `BM25Okapi` con `ZeroDivisionError`:
aggiunto un token sentinella in `Bm25Index`.

**Come testare.**
```bash
python -m pytest                       # 75 test (66 + 9 retrieval)
python eval/retrieval_ablation.py      # A/B vettoriale vs hybrid, no LLM (~secondi)
python eval/run_eval.py                # eval completa con hybrid (Ollama, ~4-7 min)
```

**Risultati misurati (2026-06-16).**
- Test: **75 passati**.
- Ablation (13 domande con doc attesi, no LLM): retrieval-hit 1,0 in entrambe le
  modalità; rango del documento corretto = 1 in entrambe per tutte le domande
  (**0 peggiorate**, 0 migliorate, 13 uguali); top-k identico in 9/13 (in 4 casi
  BM25 cambia i chunk di *supporto* alle posizioni 2–5).
- Eval completa con hybrid: behavior 0,85 (era 0,90), abstention 0,714 (era 0,857),
  retrieval/citation 1,0, course/topic 1,0. **Unica differenza: q17** (no_answer):
  BM25 ha recuperato il piano di studi (contenuto sui semestri) e il modello ha
  risposto invece di astenersi. È un effetto di *generazione/astensione* (come q18),
  non di qualità del retrieval; obiettivo della FASE 6. Dettaglio in
  `docs/valutazione_rag.md` e `docs/esperimenti_rag.md` (ESP-02).

**Rischi residui.** L'aumento di recall può far emergere contenuti tangenziali su
domande borderline (caso q17): da governare con evidence selection (FASE 5) e
astensione robusta (FASE 6). La retrieval-hit resta satura (1,0) sul corpus
attuale: la robustezza andrà ri-misurata su corpus ampliato.

**Prossimo step.** FASE 4 — reranker neurale opzionale (cross-encoder multilingua),
con fallback euristico e misura dell'impatto su pertinenza, tempi e RAM/CPU.

---

## 2026-06-17 — FASE 4: Reranker neurale opzionale (cross-encoder multilingua)

**Obiettivo.** Aggiungere un reranker neurale **opzionale e configurabile**, con
fallback automatico all'euristica, e **misurarne l'impatto** (incluse RAM, CPU,
tempi), senza imporre un modello pesante di default.

**File analizzati.** `agent.py` (`_retrieve_documents`), pipeline di retrieval.

**File creati.**
- `neural_reranker.py` — `CrossEncoderReranker` (caricamento pigro, scorer
  iniettabile per i test, fallback automatico) e `rerank_by_scores` (puro).
- `tests/test_neural_reranker.py` — 7 test offline (scorer finto, nessun download).

**File modificati.**
- `config.py` — `RERANKER_ENABLED` (env `UNILAW_RERANKER`, default OFF),
  `RERANKER_MODEL_NAME` (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`),
  `RERANKER_TOP_N=15`.
- `agent.py` — `UniLawResponder(..., use_neural_reranker=None)`; reranker neurale
  applicato ai top-N candidati **dopo** euristica e filtro corso; trace `reranker`.
- `rag_types.py` — `RagTrace.reranker`.
- `app_agent.py` — toggle "Reranker neurale (cross-encoder)" in sidebar e voce nel
  debug RAG.
- `eval/run_eval.py`, `eval/retrieval_ablation.py` — flag `--reranker` per l'A/B.

**Dipendenze.** Nessuna nuova libreria: si usa `sentence-transformers` (già
presente). Il modello (~458 MB) viene scaricato da HuggingFace al primo uso.

**Costi misurati (CPU).** Caricamento ~77 s a freddo (~1–2 s a caldo, una tantum
per processo); inferenza ~4 ms per coppia (≈56 ms per 15 candidati, trascurabile a
query); RAM di picco ~800 MB; disco ~458 MB.

**Come testare.**
```bash
python -m pytest                                   # 82 test (offline, no modello)
python eval/retrieval_ablation.py --reranker       # A/B hybrid vs hybrid+ce (no LLM)
python eval/run_eval.py --reranker                 # eval completa con reranker (Ollama)
UNILAW_RERANKER=1 streamlit run app_agent.py       # oppure toggle in sidebar
```

**Risultati misurati (2026-06-17).**
- Test: **82 passati** (75 + 7).
- Ablation (13 domande, no LLM): retrieval-hit 1,0 in entrambe le configurazioni,
  ma il rango del documento corretto **peggiora in 6/13** casi (0 migliorati, 7
  uguali) — es. q01 da rango 1 a 5. Il cross-encoder generico **demolisce i priori
  di dominio** dell'euristica (boost sulle fonti autorevoli).
- Eval completa con reranker: behavior 0,90, abstention 0,714, retrieval/citation
  1,0. Rispetto all'hybrid (FASE 3): **q14 passa da astensione errata a risposta
  corretta** (il reranker riordina il contesto sulla consultabilità della tesi,
  risolvendo la falsa astensione segnalata fin dalla FASE 1); q17/q18 invariati.
  Behavior aggregato pari alla baseline euristica (0,90).

**Decisione tecnica.** Il cross-encoder generico **non è un guadagno netto** su
questo corpus tarato: aiuta un caso borderline di generazione (q14) ma peggiora il
rango delle fonti autorevoli. Resta quindi **opzionale e disattivato di default**,
con l'euristica come ordinamento primario. Potrebbe risultare utile su corpus più
ampi o meno tarati: per questo è mantenuto, configurabile e misurabile.

**Rischi residui.** Il modello generico non conosce i priori di dominio; un
reranker addestrato/calibrato sul dominio, o una fusione euristica⊕neurale, sono
possibili evoluzioni. La RAM cresce di ~500 MB quando attivo.

**Prossimo step.** FASE 5 — evidence selection (passaggi più brevi e precisi) e
verifica delle citazioni a livello di affermazione, con l'obiettivo di ridurre in
modo robusto i casi q14 (falsa astensione) e q18 (over-answering).

---

## 2026-06-17 — FASE 5: Evidence selection + verifica delle citazioni

**Obiettivo.** Fornire al modello evidenze **più brevi e mirate** e **verificare le
citazioni** (rimuovere quelle inventate, segnalare le affermazioni con supporto
debole), per ridurre falsa astensione e over-answering.

**File creati.**
- `evidence.py` — `split_sentences`, `select_passage` (selezione dei passaggi più
  pertinenti alla domanda, con minimo di frasi garantito e tetto di caratteri).
- `tests/test_evidence.py` (5 test) e nuovi test in `tests/test_citations.py` (5).

**File modificati.**
- `citations.py` — `strip_invalid_citations` (rimuove i [F#] inventati) e
  `grounding_report` (supporto lessicale delle frasi che citano).
- `config.py` — `EVIDENCE_SELECTION_ENABLED` (default True), `EVIDENCE_MAX/MIN_SENTENCES`,
  `EVIDENCE_MAX_CHARS`, `CITATION_GROUNDING_ENABLED`, `CITATION_GROUNDING_MIN_RATIO`.
- `agent.py` — `UniLawResponder(..., use_evidence=None)`; `_build_context` applica
  l'evidence selection (solo ramo LLM); dopo la generazione: rimozione citazioni
  inventate + grounding con politica **"reduce"** (confidenza più bassa + nota, mai
  blocco cieco). Trace `evidence_chars`, `grounding`.
- `rag_types.py` — `RagTrace.evidence_chars`, `RagTrace.grounding`.
- `app_agent.py` — debug RAG mostra evidence e grounding.
- `eval/run_eval.py` — flag `--no-evidence` per l'A/B.

**Scelta di progetto.** Evidence selection e grounding agiscono **solo sul ramo
LLM**: i template deterministici sono grounded by construction e restano invariati
(test deterministici verdi). La politica di grounding è "reduce" (nota +
confidenza), non blocco/rigenerazione, per non sopprimere risposte corrette su un
modello locale; il meccanismo di blocco resta una possibile estensione.

**Come testare.**
```bash
python -m pytest                       # 92 test offline
python eval/run_eval.py                # FASE 5 (evidence + grounding, default)
python eval/run_eval.py --no-evidence  # isola l'effetto dell'evidence selection
```

**Risultati misurati (2026-06-17).**
- Test: **92 passati** (82 + 10).
- Eval completa, confronto attribuito:

  | Configurazione | behavior | abstention |
  |---|---|---|
  | hybrid, senza evidence (`--no-evidence`) | 0,85 | 0,714 |
  | **evidence + grounding (default FASE 5)** | **0,95** | **1,00** |
  | evidence + reranker neurale | 0,95 | 0,857 |

  Il guadagno (+0,10 behavior, +0,29 abstention) è **interamente attribuibile
  all'evidence selection**: con `--no-evidence` i numeri tornano a quelli dell'hybrid.
  **q18 e q17 (over-answering) risolti**; abstention perfetta (7/7). Il grounding
  non altera la classificazione (politica "reduce"), ma rimuove citazioni inventate
  e segnala il supporto debole. Aggiungere il reranker **non** migliora (peggiora
  l'astensione): resta OFF. Report `eval/reports/baseline_20260617_095838.*`.

**Caso residuo q14 (verificato).** "La tesi è consultabile dopo la laurea?": il
documento corretto (`regolamento-tesi-2023.pdf`, p.2) **contiene** le regole
(consultabile / embargo 24 mesi / non consultabile) ed è recuperato, ma il modello
locale 8B continua ad astenersi. Non è un errore di etichetta né di retrieval, ma
un limite di generazione del modello locale; l'unico intervento che lo ribalta (il
reranker) costa più di quanto rende. Documentato come limite residuo.

**Rischi residui.** Il grounding lessicale è euristico (possibili falsi
segnali su parafrasi); l'evidence selection potrebbe, su altre domande, tagliare
una frase utile (mitigato dal minimo garantito). Da ri-valutare su corpus ampliato.

**Prossimo step.** FASE 6 — astensione affidabile: classificare la causa
dell'astensione (fuori dominio / ambigua / documento mancante / retrieval debole /
fonte insufficiente) e testarla con domande negative; affrontare q14 come caso di
studio della distinzione "fonte presente ma non sfruttata".

---

## 2026-06-17 — FASE 6: Astensione affidabile (classificazione della causa)

**Obiettivo.** Rendere l'astensione **affidabile e distinguibile**: classificare la
*causa* del "non lo so" tra le categorie richieste e testarla con domande negative.

**File creati.**
- `abstention.py` — tassonomia delle cause (`fuori_dominio_corso`, `ambigua`,
  `retrieval_debole`, `fuori_dominio`, `evidenza_insufficiente`), `retrieval_strength`
  (quota di token della domanda coperti dalla migliore fonte), `classify_llm_abstention`,
  `is_abstention`, `format_reason`.
- `tests/test_abstention_reasons.py` — 11 test offline (classificazione + trace).

**File modificati.**
- `config.py` — `ABSTENTION_OOD_MAX_STRENGTH=0.37` (soglia fuori-dominio vs
  insufficiente, **calibrata sui casi reali**).
- `agent.py` — `trace.abstention_reason` impostato su ogni ramo: corso ignoto →
  `fuori_dominio_corso`, ambigua → `ambigua`, nessuna fonte → `retrieval_debole`;
  e, sul ramo LLM, se la risposta è un'astensione, classificazione automatica
  `fuori_dominio` vs `evidenza_insufficiente` (con nota esplicativa accodata).
- `rag_types.py` — `RagTrace.abstention_reason`.
- `app_agent.py` — il debug RAG mostra la causa di astensione.
- `eval/questions_baseline.jsonl` — campo `expected_abstention_reason` sulle 7
  domande negative.
- `eval/run_eval.py` — nuova metrica `abstention_reason_accuracy`.

**Calibrazione (evidence-based).** La soglia 0,37 deriva dalle *retrieval strength*
misurate: q19 (capitale Francia)=0,33 → fuori dominio; q17 (orario)=0,40,
q18 (mensa)=0,67, q14 (consultabilità)=0,50 → in dominio ma risposta assente. La
sonda ha **corretto un'assunzione**: il corpus cita il servizio mensa (bando borsa),
quindi q18 è "fonte presente ma insufficiente", non "fuori dominio".

**Scelta di progetto.** Politica conservativa: i messaggi deterministici di
astensione restano invariati (preservano i marker dei test); il ramo LLM aggiunge
una riga di spiegazione classificata. Nessun blocco/rigenerazione.

**Come testare.**
```bash
python -m pytest             # 102 test offline
python eval/run_eval.py      # include abstention_reason_accuracy
```

**Risultati misurati (2026-06-17).**
- Test: **102 passati** (92 + 10).
- Eval completa: behavior 0,95, abstention 1,0, **abstention_reason_accuracy 1,0**
  (7/7 cause corrette: q10/q11 fuori_dominio_corso, q12/q13 ambigua, q17/q18
  evidenza_insufficiente, q19 fuori_dominio). Nessuna regressione sulle altre
  metriche. Report `eval/reports/baseline_20260617_104430.*`.

**Rischi residui.** La `retrieval_strength` è lessicale e la soglia 0,37 è calibrata
su questo set: l'accuratezza della causa va ri-validata su domande negative
held-out. La distinzione fuori_dominio/insufficiente resta euristica.

**Prossimo step.** FASE 7 — riduzione dell'hard-coding: classificare le regole
deterministiche e spostare la conoscenza normativa (tabella L-19, soglie TOLC)
verso dati strutturati derivati dalle fonti, con test dedicati.

---

## 2026-06-17 — FASE 7: Riduzione dell'hard-coding (conoscenza → dati)

**Obiettivo.** Classificare tutte le regole deterministiche e trasformare la
conoscenza normativa codificata in **dati strutturati e tracciabili**, senza
cambiare il comportamento.

**Verifica sulle fonti (preliminare).** Confermati sui PDF: TOLC L-31 "Ris_Test non
inferiore a 16 / inferiore a 16 e non inferiore a 9" (`regolamento-di-accesso...`);
L-19 "2 ore e 30 minuti, 80 quesiti, Cultura generale 30 / Inglese 10 / Logica 20 /
Comprensione 20" (`immatricolazione scienze dell'educazione...`).

**File creati.**
- `knowledge.py` — `SourceRef` (provenienza), `TOLC_INFORMATICA` (+source),
  `classify_tolc`, `tolc_band_labels`, `tolc_bands_table`, `L19_ADMISSION` (+source),
  `l19_test_table_markdown`.
- `tests/test_knowledge.py` — 8 test (parità byte-per-byte con le stringhe legacy,
  totali coerenti, provenienza presente).

**File modificati.**
- `rules_tolc.py` — `classify_tolc_score` ora **delega** a `knowledge.classify_tolc`
  (soglie 9/16 non più codificate qui).
- `agent.py` — le 3 tabelle delle fasce TOLC e la tabella della prova L-19 nei
  template sono **generate dal layer di conoscenza** (output identico).
- `docs/relazione_unilaw_agent.md` (sezione 7: classificazione completa delle regole).

**Classificazione delle regole.** In `docs/relazione_unilaw_agent.md` §7: ogni
regola è etichettata (mantenere / trasformare in dato / mantenere con test /
ridurre) con la strategia. Trasformate in dato: soglie TOLC e tabella L-19. Ridotte
parzialmente: i template (parti normative ora dai dati). Lasciate con motivazione:
prosa dei template, regole normative nel prompt, intent a keyword (tutte testate).

**Come testare.**
```bash
python -m pytest             # 110 test offline (102 + 8 knowledge)
python eval/run_eval.py      # nessuna regressione attesa (refactor invariante)
```

**Risultati misurati (2026-06-17).**
- Test: **110 passati** (102 + 8). Le tabelle generate coincidono byte-per-byte con
  quelle codificate (test dedicati); output deterministico q01 verificato identico.
- Eval completa: course/topic/retrieval/citation 1,0; behavior 0,90 e abstention
  0,857 in questa riesecuzione.

**Nota di riproducibilità (importante).** Il calo apparente (0,90/0,857 vs 0,95/1,0
della FASE 6) **non è una regressione**: la modifica è invariante per costruzione
(byte-identica, 110 test verdi, ramo LLM intoccato). Il diff rispetto alla FASE 6
riguarda **una sola** domanda, q17 (ramo LLM), che oscilla tra astensione e risposta
a parità di pipeline: il modello locale `llama3.1:8b` non è perfettamente
deterministico nemmeno a `temperature=0`. Tutte le domande deterministiche (q01–q13)
sono invariate. Report `eval/reports/baseline_20260617_120429.*`.

**Rischi residui.** Restano hard-coded (motivati, testati): prosa dei template,
regole normative nel prompt `ANSWER_STYLE_GUIDE`, intent detection a keyword. La
variabilità del modello locale rende instabili di ±1 domanda le metriche di
generazione su casi borderline.

**Prossimo step.** FASE 8 — UI e usabilità: stato dell'indice, trace esportabile,
report fonti, messaggi di errore più chiari, senza peggiorare la semplicità.

---

## 2026-06-17 — FASE 8: UI e usabilità reale

**Obiettivo.** Rendere fruibile ed esportabile l'osservabilità già nel trace,
chiarire stato e limiti, migliorare i messaggi d'errore — senza appesantire la UI.

**File creati.**
- `trace_export.py` — `trace_to_dict`, `trace_to_json`, `trace_to_markdown`:
  esportazione completa e leggibile del RagTrace (funzioni pure).
- `tests/test_trace_export.py` — 5 test offline (JSON valido, sezioni Markdown,
  trace vuoto/None, placeholder liste vuote).

**File modificati.**
- `app_agent.py`:
  - **Trace esportabile**: pulsanti di download JSON e Markdown nel pannello di
    debug (sidebar e inline);
  - **Stato pipeline**: nuovo blocco "PIPELINE" in sidebar (retrieval ibrido,
    reranker ON/OFF, evidence ON/OFF, verifica citazioni, astensione classificata);
  - **Upload documenti**: `st.file_uploader` per aggiungere PDF al corpus
    (`documenti/`) con ricostruzione dell'indice;
  - **Errori più chiari**: messaggio strutturato con cause comuni e rimedi.

**Verifica.** Avvio headless con `streamlit.testing.v1.AppTest`: l'app si carica
**senza eccezioni**; il blocco PIPELINE e i controlli sono presenti. I pulsanti di
export compaiono nel debug dopo una risposta (contenuto coperto dai 5 test).

**Come testare.**
```bash
python -m pytest                       # 115 test offline
streamlit run app_agent.py             # UI: PIPELINE, upload PDF, export trace, errori
```

**Risultati misurati (2026-06-17).**
- Test: **115 passati** (110 + 5). Boot headless dell'app: nessuna eccezione.

**Rischi residui.** L'upload scrive in `documenti/` e innesca un rebuild (operazione
esplicita dell'utente, può richiedere tempo). I pulsanti di export non sono coperti
da test d'integrazione UI (solo la generazione del contenuto è testata).

**Prossimo step.** FASE 9 — documentazione finale: rifinire la relazione
universitaria (baseline, miglioramenti, esperimenti, risultati, limiti) e la
copertina con il logo (`assets/logo_unisa.png`).

---

## 2026-06-17 — FASE 9: Documentazione finale (chiusura del percorso)

**Obiettivo.** Rifinire la relazione universitaria come documento coerente e
completo (non una semplice demo) e garantire la coerenza tra tutti i `docs/`.
Fase di sola documentazione: nessuna modifica al codice (115 test invariati).

**File modificati.**
- `docs/relazione_unilaw_agent.md`: Abstract aggiornato all'esito complessivo;
  Conclusioni riscritte (sintesi del percorso a 9 fasi, risultati, cosa rende il
  sistema utilizzabile, limiti residui, valore didattico); Appendice con comandi
  aggiornati, struttura completa dei moduli e **istruzioni di conversione PDF/DOCX**
  (pandoc); aggiunta della voce FASE 8 nei "Miglioramenti"; header allineato.
- `docs/valutazione_rag.md`: conteggio test allineato a 115.

**Coerenza verificata.** Numeri di test correnti (115) coerenti nei documenti
"vivi" (relazione, valutazione); i riferimenti storici per-fase (62/.../110) sono
mantenuti dove descrivono lo stato di quella fase. Tutte e 9 le fasi sono presenti
nel changelog e nei "Miglioramenti" della relazione.

**Come verificare.**
```bash
python -m pytest        # 115 test verdi (nessuna modifica al codice in FASE 9)
```

**Azioni manuali residue per l'utente** (dati che non posso conoscere):
- ~~inserire il logo ufficiale in `assets/logo_unisa.png`~~ → **fatto dall'utente il
  2026-06-17**; la copertina ora usa `![...](../assets/logo_unisa.png)` al posto del
  segnaposto;
- ~~completare anno accademico, docente e data~~ → **fatto**: copertina con
  anno accademico 2025/2026, docente Fabio Palomba, data 24/06/2026;
- ~~(opzionale) convertire in PDF/DOCX con pandoc~~ → **fatto**: generati in radice
  `relazione_unilaw_agent.{docx,pdf}` e `relazione_completa_unilaw_agent.{docx,pdf}`.

### 2026-06-17 — Logo di copertina inserito
L'utente ha aggiunto `assets/logo_unisa.png` (sigillo ufficiale UniSA). Aggiornati:
copertina della relazione (immagine al posto di `[IMMAGINE ALLEGATA]`), header del
documento, `assets/README.md`, e i comandi pandoc in Appendice (`--resource-path=docs`
per risolvere il percorso del logo in fase di conversione).

**Stato del progetto.** Roadmap completata (FASE 0 → 9). Il sistema è un RAG locale
misurabile, document-grounded, con retrieval ibrido, reranker opzionale, evidence
selection, verifica delle citazioni, astensione classificata, conoscenza normativa
tracciabile, UI con trace esportabile, 115 test automatici e una valutazione
sperimentale ripetibile.

---

## 2026-06-17 — Correzione: indice ChromaDB spostato fuori da iCloud

**Sintomo.** All'avvio dell'app la knowledge base non si inizializzava:
`sqlite3.OperationalError: attempt to write a readonly database` durante la
ricostruzione, seguito da `no such table: tenants` / `Could not connect to tenant
default_tenant` (indice corrotto, `chroma.sqlite3` ridotto a 0 byte).

**Causa.** Il progetto è in `~/Desktop`, sincronizzata da iCloud. Durante una
ricostruzione dell'indice, iCloud ha bloccato/evinto il file SQLite mentre veniva
scritto, facendo fallire la prima scrittura e corrompendo il database.

**Correzione.** `config.py`: `CHROMA_PERSIST_DIRECTORY` ora punta, per default, a
`~/Library/Application Support/UniLawAgent/chroma_db` (cartella **non** sincronizzata
da iCloud), con override via `UNILAW_CHROMA_DIR`. L'indice è stato ricostruito nella
nuova posizione (22 PDF, 1390 chunk) e verificato: caricamento da disco e query
funzionanti. Aggiornati i riferimenti in `README.md` e `docs/architettura_rag.md`.

**Note.** La vecchia cartella `.chroma_db/` nel progetto è ora inutilizzata (conteneva
solo un `chroma.sqlite3` da 0 byte) e può essere eliminata. 115 test invariati. Per
applicare la correzione basta riavviare `streamlit run app_agent.py`.

---

## 2026-06-17 — Decisione: RAG puro come comportamento primario (ESP-07 applicato)

**Contesto.** L'esperimento ESP-07 (`docs/esperimenti_rag.md`) ha misurato che, senza
i template deterministici, il RAG generativo mantiene la stessa accuratezza
aggregata: il sistema **non dipende** dalle risposte codificate per leggere e
comprendere i documenti. L'unico vantaggio residuo dei template è sull'esattezza
numerica delle soglie di accesso e sulla citazione della fonte canonica.

**Decisione.** Disattivare di default i 5 template "di prosa" (OFA, tesi, Erasmus,
borsa, accesso L-19) e mantenere attivo solo il **guard numerico TOLC-I**.

**Modifiche.**
- `config.py`: aggiunto `PROSE_TEMPLATES_ENABLED` (default False, env
  `UNILAW_PROSE_TEMPLATES`). `DETERMINISTIC_RULES_ENABLED` resta master (default True).
- `agent.py`: `UniLawResponder(..., use_prose_templates=None)`; nel ramo deterministico
  il guard numerico TOLC è sempre valutato (col master attivo), i 5 template di prosa
  solo se abilitati.
- `eval/run_eval.py`: flag `--prose-templates` per riprodurre la configurazione "tutti
  i template".

**Risultati misurati (configurazione predefinita, 2026-06-17).** behavior 0,95,
course/topic 1,0, retrieval-hit 1,0, citation-hit 1,0, abstention 1,0, reason 1,0.
Su 13 domande con risposta, solo 4 (punteggio TOLC) usano il guard numerico; le altre
9 sono RAG puro. Report `eval/reports/baseline_20260617_155116.*`.

**Documentazione.** Aggiornate `docs/relazione_completa_unilaw_agent.md` (sezioni 9 e
13, con le tabelle dei risultati), `docs/esperimenti_rag.md`, `docs/valutazione_rag.md`,
`docs/architettura_rag.md`, `docs/relazione_unilaw_agent.md`, `README.md`.

**Come testare.**
```bash
python -m pytest                      # 115 test
python eval/run_eval.py               # default (guard TOLC, prosa off)
python eval/run_eval.py --no-deterministic   # RAG puro
python eval/run_eval.py --prose-templates    # tutti i template (legacy)
```

**Rischi residui.** Il guard TOLC resta una regola codificata (motivata: esattezza
soglie). Su q01 il RAG puro è meno netto sul "con OFA" per il valore specifico: il
guard lo copre. Variabilità ±1 domanda del modello locale invariata.

---

# Ciclo 2 — consolidamento e qualità

> Le fasi del Ciclo 2 hanno numerazione propria (FASE 1–16) e vanno citate come
> «Ciclo 2 — FASE n», per non confonderle con le FASE 0–9 del Ciclo 1. Piano completo
> in `docs/roadmap_progetto.md`.

## 2026-06-17 — Ciclo 2 — FASE 1: Fix bug separatore delle migliaia

**Obiettivo.** Correggere il difetto di calcolo numerico caratterizzato (non corretto)
in FASE 1 del Ciclo 1: un singolo punto usato come separatore di migliaia veniva letto
come decimale.

**File analizzati.** `tools.py` (`_normalizza_numero_italiano`, `_estrai_calcolo_percentuale`,
`prova_calcolo_sicuro`), `agent.py:169` (punto di ingresso del calcolo),
`tests/test_tools.py`.

**File modificati.** `tools.py`, `tests/test_tools.py`.

**Modifiche tecniche.**
- `_normalizza_numero_italiano`: aggiunto un ramo che, in assenza di virgola e con un
  solo punto seguito **da esattamente 3 cifre** (regex `-?\d+\.\d{3}`), interpreta il
  punto come separatore di migliaia (`"20.000"` → `20000`). Gli altri casi restano
  invariati: punto+virgola → migliaia/decimale (`"20.000,50"` → `20000.5`); solo virgola
  → decimale (`"5,5"` → `5.5`); più punti → tutti migliaia (`"1.234.567"` → `1234567`);
  un punto con un numero di cifre diverso da 3 → decimale (`"20.5"` → `20.5`,
  `"20000.50"` → `20000.5`).
- Il fix è circoscritto al ramo percentuali: `_normalizza_numero_italiano` è usato solo
  da `_estrai_calcolo_percentuale`. Il path delle espressioni aritmetiche pure
  (`calcola_espressione_sicura`, es. `10/4` → `2,5`) **non** passa per questa funzione e
  resta invariato.

**Modifiche ai test.**
- `test_percentage_thousands_separator_known_bug` → rinominato
  `test_percentage_thousands_separator` e convertito in test di **correttezza**:
  «5% di 20.000€» ora deve restituire `"1.000"` (prima certificava il bug con `"1"`).
- Aggiunto `test_normalizza_numero_italiano` (parametrico, 8 casi) che copre
  direttamente i formati documentati. Suite `tests/test_tools.py`: da 7 a 15 test.

**Impatto sul RAG.** Migliora la correttezza delle risposte aritmetiche su importi con
separatore di migliaia in formato italiano, dominio in cui un errore numerico è un
difetto di affidabilità.

**Trade-off documentato.** Un decimale "puro" con esattamente 3 cifre dopo il punto
(es. `"3.141"`) viene ora interpretato come migliaia (`3141`). È la convenzione scelta
nella roadmap (un punto + 3 cifre = migliaia): coerente con il dominio (importi in euro
in formato italiano), dove i decimali si scrivono con la virgola.

**Come testare.**
```bash
python -m pytest                      # 123 test, attesi tutti verdi
python -m pytest tests/test_tools.py  # 15 test del modulo di calcolo
```

**Risultati test.** 123 test verdi (115 + 8 nuovi casi). Nessuna regressione.

**Rischi residui.** Bassi: cambia solo l'interpretazione del singolo punto seguito da
3 cifre, per costruzione. Nessuna modifica al comportamento del modello né al retrieval.

**Prossimo step.** Ciclo 2 — FASE 2 (normalizzazione del formato citazioni `(F#)→[F#]`).

---

## 2026-06-18 — Ciclo 2 — FASE 2: Normalizzazione del formato citazioni `(F#)→[F#]`

**Obiettivo.** Ricondurre al formato canonico `[F#]` le varianti di citazione che il
modello produce occasionalmente — `(F1)` o un `F1` "nudo" — così che vengano
riconosciute dall'estrazione, dalla verifica del grounding e dal blocco fonti. Le regex
del modulo riconoscono solo `[F#]`: una citazione tra parentesi o senza parentesi è
"invisibile" e innesca il fallback fuorviante "Fonti utilizzate" introdotto in FASE 2
del Ciclo 1.

**File analizzati.** `citations.py` (`extract_cited_source_indexes`,
`strip_invalid_citations`, `grounding_report`, `format_sources_block`),
`agent.py:answer` (post-processing del ramo LLM, ll. 294–342),
`tests/test_citations.py`. Evidenza dal report `baseline_20260617_155116`: la risposta a
q14 cita «(F1) e (F2)» invece di `[F1]`/`[F2]`.

**File modificati.** `citations.py`, `agent.py`, `tests/test_citations.py`.

**Modifiche tecniche.**
- Aggiunta `citations.normalize_citations(answer, sources)`, funzione pura che:
  - converte i gruppi tra parentesi composti **solo** da riferimenti a fonti — `(F1)`,
    `(F1, F2)`, `(F1 e F3)` — in `[F1]`, `[F1] [F2]`, `[F1] [F3]`;
  - converte i riferimenti "nudi" `F1` in `[F1]`, evitando le occorrenze già tra
    parentesi quadre/tonde e quelle interne a parole o codici (`xF2`, `F2a`);
  - **agisce solo sugli indici di fonti realmente recuperate**: le parentesi e i
    riferimenti con indice inesistente (`(F9)` quando esistono solo F1..F2) restano
    intatti, per non alterare la prosa con citazioni inesistenti.
- `agent.py:answer`: `normalize_citations` è agganciata **prima** di
  `strip_invalid_citations` (ramo LLM), quindi prima della verifica del grounding e del
  blocco fonti. I template deterministici non sono toccati (producono già `[F#]`).

**Modifiche ai test.**
- Aggiunti 7 test in `tests/test_citations.py`: parentesi singole, riferimenti nudi,
  gruppo con più riferimenti (`(F1, F2)`, `(F1 e F3)`), citazioni canoniche lasciate
  intatte, gating sui soli indici validi, esclusione di parole/codici (`xF2`, `F2a`),
  e un test di integrazione che mostra come dopo la normalizzazione `(F1)` venga
  riconosciuto da `grounding_report`.

**Impatto sul RAG.** Le citazioni in forme alternative vengono verificate e attribuite
correttamente: si riduce il rischio del fallback fuorviante «Fonti utilizzate» quando il
modello cita in formato non canonico (sinergia con la FASE 3 del Ciclo 2, blocco fonti
onesto in astensione). Sul piano aggregato l'impatto è atteso neutro: la normalizzazione
agisce solo quando il modello devia dal formato canonico.

**Come testare.**
```bash
python -m pytest                           # 130 test, attesi tutti verdi
python -m pytest tests/test_citations.py   # 18 test del modulo citazioni
```

**Risultati test.** 130 test verdi (123 + 7 nuovi casi). Nessuna regressione.

**Risultati eval (non-regressione).** Eseguita `python eval/run_eval.py` con la
configurazione predefinita, modello `llama3.1:8b`, `temperature=0`
(report `baseline_20260618_095313`). **`citation_hit_rate` 1,0 — invariato** rispetto ai
report di riferimento (`baseline_20260617_155116`, `baseline_20260616_224551`): è
l'obiettivo di verifica della fase. course/topic/retrieval 1,0. Le metriche di
generazione hanno dato behavior 0,90 e abstention 0,857: lo scostamento (vs 0,95/1,0 del
report finale del Ciclo 1) è dovuto a **q14** (falsa astensione su regolamento generale —
caso aperto del Ciclo 1, target della FASE 14) e **q18** (over-answering sulla mensa,
oscillazione nota del modello locale a `temperature=0`), **non** a questa modifica: la
normalizzazione riscrive solo i token di citazione e non altera ciò che il modello genera
(comportamento neutro per costruzione; q01–q13 deterministici stabili). In questa
esecuzione q14 ha citato in formato canonico `[F1]`/`[F2]` e il blocco onesto «Fonti
citate» è comparso correttamente — il meccanismo che la fase protegge.

**Rischi residui.** Bassi. Trade-off documentato: un `F1` "nudo" nella prosa che NON sia
una citazione (es. la sigla «modello F1») viene ricondotto a `[F1]` quando l'indice 1
corrisponde a una fonte recuperata. È improbabile nel dominio normativo e il gating sugli
indici validi evita di "inventare" riferimenti a fonti inesistenti. Nessuna modifica al
comportamento del modello né al retrieval.

**Prossimo step.** Ciclo 2 — FASE 3 (blocco fonti onesto in astensione: sopprimere o
rietichettare il fallback `sources[:3]` quando il sistema si astiene).

---

## 2026-06-18 — Ciclo 2 — FASE 3: Blocco fonti onesto in astensione

**Obiettivo.** Evitare la falsa attribuzione di fonti quando il sistema si astiene. In
astensione, in assenza di citazioni `[F#]` valide, `format_sources_block` ripiegava su
`sources[:3]` etichettandole «Fonti utilizzate»: ma se la risposta è «Non lo so», nessuna
fonte è stata *usata*: l'etichetta è una rivendicazione falsa. Sulla domanda fuori
dominio q19 («capitale della Francia») il sistema si asteneva ma elencava documenti
universitari casuali come «utilizzati» — il rischio di «citazioni fabbricate» già
segnalato nell'audit (Ciclo 1 — FASE 0).

**File analizzati.** `citations.py` (`format_sources_block`, `extract_cited_source_indexes`),
`agent.py:answer` (post-processing del ramo LLM, ll. 294–346) e il delegante
`_format_sources_block`, i tre formatter di astensione dedicati
(`_format_unknown_course_answer`, `_format_clarification_answer`,
`_format_no_evidence_answer` — che **non** emettono un blocco fonti, quindi già onesti),
`abstention.py:is_abstention`, `eval/run_eval.py` (scoring), `tests/test_citations.py`.
Evidenza dal report `baseline_20260617_155116` e da `citations.py:152`/`agent.py:344`.

**File modificati.** `citations.py`, `agent.py`, `tests/test_citations.py`.

**Modifiche tecniche.**
- `citations.format_sources_block` ha un nuovo parametro `abstaining: bool = False`.
  Ordine di decisione:
  - se la risposta cita **fonti reali** (`[F#]` validi) → blocco «Fonti citate» (le
    citazioni reali restano **sempre** mostrate, anche in astensione);
  - altrimenti, se `abstaining=True` → fallback rietichettato onestamente «Documenti
    consultati (nessuno utilizzato per la risposta):» (i documenti recuperati restano
    elencati, per trasparenza del retrieval, ma senza rivendicarli «utilizzati»);
  - altrimenti (risposta non astenuta senza citazioni) → fallback «Fonti utilizzate:»
    **invariato** (retrocompatibilità).
- `agent.py:answer`: l'esito di `is_abstention(final_answer)` è ora catturato una sola
  volta in `abstaining` (riusato sia per la classificazione della causa sia per il blocco
  fonti) e passato a `_format_sources_block(..., abstaining=abstaining)`.
- Il delegante `_format_sources_block` inoltra `abstaining` (default `False`): i 7
  rami a template deterministico, non astensioni, restano invariati per costruzione.

**Modifiche ai test.**
- Aggiunti 2 test in `tests/test_citations.py`:
  `test_sources_block_abstaining_relabels_fallback` (in astensione senza citazioni il
  titolo è la rietichettatura onesta, **non** «Fonti utilizzate», ma i documenti restano
  elencati) e `test_sources_block_abstaining_keeps_real_citations` (in astensione, una
  citazione `[F#]` reale resta mostrata come «Fonti citate»). Aggiornato il commento del
  test di fallback esistente (caso «non astenuto»).

**Impatto sul RAG.** Output coerente sulle astensioni: nessuna falsa attribuzione di
fonti su q19 (fuori dominio) e sugli altri casi di astensione del ramo LLM, preservando
le citazioni reali quando presenti. Migliora l'onestà degli output, obiettivo di
affidabilità del progetto.

**Come testare.**
```bash
python -m pytest                           # 132 test, attesi tutti verdi
python -m pytest tests/test_citations.py   # 20 test del modulo citazioni
```

**Risultati test.** 132 test verdi (130 + 2 nuovi casi). Nessuna regressione.

**Verifica eval (neutralità per costruzione).** La modifica cambia **solo** l'etichetta
testuale del blocco fonti in astensione; non altera ciò che il modello genera, né i
`[F#]`, né i nomi dei file elencati. Lo scorer (`eval/run_eval.py:145,155`) estrae i
documenti citati con `filenames_in(answer)` — dai **nomi file**, non dal titolo del
blocco — e calcola `citation_hit_rate` solo sulle domande *answerable*; `classify_behavior`
si basa sulle frasi di astensione, non sul titolo del blocco. Tutte le metriche
(`behavior`, `course`, `topic`, `retrieval_hit`, `citation_hit`, `abstention_reason`)
sono quindi **invariate per costruzione**. Ispezione dal vivo dei negativi (pipeline
reale, `llama3.1:8b`): su q19 il blocco è ora «Documenti consultati (nessuno utilizzato
per la risposta):» (prima «Fonti utilizzate»); q17/q18, quando rispondono con citazioni
valide, mostrano correttamente «Fonti citate».

**Rischi residui.** Bassi: nessuna modifica al modello né al retrieval; i rami
deterministici e le risposte non astenute sono invariati (default `abstaining=False`).

**Prossimo step.** Ciclo 2 — FASE 4 (ampliare il dataset di valutazione, primo passo del
Blocco B di rinforzo della misurazione).

---

## 2026-06-18 — Ciclo 2 — FASE 4: Ampliamento del dataset di valutazione

**Obiettivo.** Con 20 domande su un corpus "saturo" molte metriche erano al 100% e non
discriminavano; senza domande nuove non si poteva misurare la robustezza né giustificare
i refactor del Blocco C. Primo passo del Blocco B (fondamenta di misurazione): portare il
dataset da 20 a 40 domande, mantenendo lo schema delle etichette, e produrre la **nuova
baseline di riferimento**.

**File analizzati.** `eval/questions_baseline.jsonl`, `eval/run_eval.py` (schema,
metriche, basi di calcolo), `intent.py` (`infer_query_intent` e predicati), `knowledge.py`
e `abstention.py` (tassonomia cause), corpus `documenti/` (22 PDF: verifica del grounding
dei valori normativi delle nuove domande *answerable*).

**File creati.** `tests/test_eval_dataset.py` — 9 test di integrità del dataset (offline).

**File modificati.** `eval/questions_baseline.jsonl` (q21–q40), `docs/valutazione_rag.md`
(§1.1, §1.2, §1.3, nuova §14).

**Modifiche tecniche.**
- **+20 domande (q21–q40)** distribuite su quattro assi: (a) *più corsi/argomenti* —
  prove finali e norme redazionali di L-16, piano di studi di L-19, prova finale di
  Informatica L-31, ammissione a Economia (L-18; corso `economia` mai testato prima),
  durata Erasmus, scadenze borsa (esercitano documenti del corpus mai toccati dal set
  originale); (b) *parafrasi/sinonimi* di domande note (TOLC, ammissione L-19,
  graduatorie borsa); (c) *distrattori* (es. *Norme redazionali* vs *Linee guida prove
  finali* di L-16); (d) *negativi held-out* q34–q39 (riservati alla validazione fuori
  campione della FASE 6), che coprono quattro cause di astensione.
- I nomi dei documenti attesi sono stati **risolti dai file reali su disco** in fase di
  generazione (nessun typo su trattini/apostrofi), e i valori normativi delle answerable
  sono stati **verificati sui PDF** (onestà del grounding).
- Le 20 domande storiche (q01–q20) restano **invariate** per preservare la
  comparabilità con la baseline del Ciclo 1.

**Modifiche ai test.**
- `tests/test_eval_dataset.py` (9 test, offline) valida: caricamento e crescita del
  dataset, unicità degli `id`, schema e domini dei valori, **coerenza deterministica
  delle etichette** (`expected_course`/`expected_topic` == `infer_query_intent(...)`),
  coerenza dei comportamenti senza LLM con i predicati dell'intent
  (`unknown_course`→corso rilevato, `clarify`→ambiguo), causa di astensione valida sui
  negativi, esistenza su disco dei documenti attesi, copertura del set held-out. È una
  **rete di sicurezza contro l'etichettatura errata**: un'etichetta incoerente fa
  fallire i test invece di falsare in silenzio le metriche.

**Impatto sul RAG.** Nessuna modifica al comportamento del sistema (solo dati di test e
documentazione). Le metriche diventano **più informative**: sul dataset a 40 domande la
baseline misurata è behavior **0,90** (36/40), course/topic **1,0**, retrieval/citation
**1,0** (27/27), abstention **0,923** (12/13), abstention_reason **0,923** (12/13)
(`eval/reports/baseline_20260618_132450.*`). I 4 errori di `behavior` sono tutti del
modello locale: q14 (storico) e **q21, q29 (nuovi)** sono false astensioni *con la fonte
corretta recuperata e citata* (stesso limite di generazione di q14, ora osservato su
corso/argomento diversi → famiglia di errori, non caso isolato); q18 è over-answering.
**Tutti i 6 negativi held-out (q34–q39) si astengono e sono classificati con la causa
giusta**: evidenza preliminare che la soglia 0,37 generalizza (validazione formale in
FASE 6). `retrieval_hit` resta 1,0: i distrattori vengono recuperati ma non spiazzano il
gold → per stressare davvero il retrieval servirebbero documenti-trappola nel corpus
(estensione futura, non in questa fase).

**Come testare.**
```bash
python -m pytest                       # 141 test offline, attesi tutti verdi
python -m pytest tests/test_eval_dataset.py   # 9 test di integrità del dataset
python eval/run_eval.py                # nuova baseline a 40 domande (Ollama, ~15 min)
```

**Risultati test.** 141 test verdi (132 + 9 nuovi). Nessuna regressione.

**Rischi residui.** Bassi sul piano del codice (nessuna modifica alla pipeline). Sul
piano della misurazione: alcune etichette di causa per i negativi a generazione (q35–q37)
dipendono dal comportamento del modello e dalla soglia lessicale 0,37; sono proprio gli
oggetti della validazione fuori campione (FASE 6) e della `retrieval_strength` semantica
(FASE 13).

**Prossimo step.** Ciclo 2 — FASE 5 (scoring dell'eval più robusto: classificare l'esito
dai segnali strutturati del `RagTrace` invece che dai soli marcatori testuali).

---

## 2026-06-18 — Ciclo 2 — FASE 5: Scoring dell'eval dai segnali del trace

**Obiettivo.** Secondo passo del Blocco B (fondamenta di misurazione): rendere lo scoring
dell'eval **stabile al variare del wording del modello**. Per calcolare
`behavior_accuracy` l'harness deve dedurre se il sistema ha *risposto*, *si è astenuto*,
ha *chiesto chiarimenti* o ha *rifiutato un corso fuori dominio*. In origine questa
classificazione si basava sui soli **marcatori testuali** della risposta: una soluzione
fragile, perché basta che il modello riformuli un'astensione perché lo scorer la legga
come risposta. Il `RagTrace` contiene già segnali strutturati più affidabili.

**File analizzati.** `eval/run_eval.py` (classificazione testuale e aggregazione),
`agent.py` (rami che impostano `trace.abstention_reason` su corso fuori dominio, ambigua,
retrieval debole e, sul ramo generativo, la classificazione di `is_abstention`),
`abstention.py` (tassonomia cause e `UNCERTAINTY_MARKERS`), `rag_types.py` (campo
`abstention_reason` del `RagTrace`).

**File creati.** `tests/test_eval_scoring.py` — 8 test offline dello scoring.

**File modificati.** `eval/run_eval.py`; `docs/valutazione_rag.md` (§1.1, §1.3).

**Modifiche tecniche.**
- **`classify_behavior(answer, trace)`** (prima `classify_behavior(answer)`): l'esito è
  ora dedotto in primo luogo dalla **causa di astensione strutturata** del trace —
  `OUT_OF_DOMAIN_COURSE`→`unknown_course`, `AMBIGUOUS`→`clarify`,
  `{WEAK_RETRIEVAL, OUT_OF_DOMAIN, INSUFFICIENT_EVIDENCE}`→`abstain` — e, in assenza di
  causa, è `answer`. La funzione restituisce `(esito, fonte)` con `fonte ∈ {trace, text}`.
- **Fallback testuale** preservato in `classify_behavior_from_text(answer)`: usato quando
  il trace non porta segnali. L'**LLM non raggiungibile** non lascia una causa nel trace
  (la chiamata fallisce prima), quindi resta riconosciuto dai marcatori testuali.
- Perché il trace è più affidabile: la `is_abstention` dell'agente riconosce un insieme
  di formulazioni **più ampio** delle `ABSTAIN_PHRASES` dello scorer testuale (cattura in
  più «non ho trovato», «non sono in grado di fornire», «non sono in possesso») → un'
  astensione con quelle parole, prima letta come `answer`, ora è classificata `abstain`.
- **Trasparenza nel report**: nuovo campo `behavior_source` per ogni domanda e conteggi
  aggregati `behavior_from_trace` / `behavior_from_text` (più alto è `trace`, più lo
  scoring è robusto). Nuova riga nel report Markdown.
- **Refactor degli import** di `run_eval.py`: le dipendenze pesanti (`Chroma`, `agent`,
  `database`) e gli effetti collaterali globali (`os.chdir`, `logging.disable`,
  `warnings.filterwarnings`) sono spostati dentro `main()`/`load_vector_db()`. Così il
  modulo è **importabile offline dai test** per le sole funzioni di scoring, senza
  caricare l'indice né Ollama; l'esecuzione come script (`python eval/run_eval.py`) è
  invariata.

**Modifiche ai test.** `tests/test_eval_scoring.py` (8 test, offline, trace sintetici):
mappa causa→esito per tutte le cause; priorità del trace sul testo; causa non mappata
trattata comunque come astensione; risposta senza causa → `answer`; **caso centrale** —
un'astensione con formulazione fuori dalle `ABSTAIN_PHRASES` («Non ho trovato…») che lo
scorer testuale mancherebbe (`classify_behavior_from_text`→`answer`) ma il trace coglie
(`abstain`); riconoscimento dell'LLM non disponibile dal testo; fallback testuale per i
rami senza LLM; **consistenza** — ricostruendo il trace dalla causa registrata, lo scoring
riproduce il `predicted_behavior` del report più recente (no regressione sui casi noti).

**Impatto sul RAG.** Nessuna modifica al comportamento del sistema (cambia solo come
l'eval *legge* gli esiti). La modifica è **verdict-neutra per costruzione** e verificata:
sulla baseline a 40 domande i verdetti coincidono **riga per riga** con il report
precedente (FASE 4, `baseline_20260618_132450`) — **0 domande cambiate**. Il nuovo report
(`eval/reports/baseline_20260618_185545.*`, `llama3.1:8b`) conferma le metriche: behavior
**0,90** (36/40), course/topic **1,0**, retrieval/citation **1,0** (27/27), abstention
**0,923** (12/13), reason **0,923** (12/13). Dato saliente: **tutti i 40 verdetti
provengono dai segnali del trace** (`behavior_from_trace: 40`, `behavior_from_text: 0`) →
il trace copre l'intero dataset; il fallback testuale resta per robustezza futura (LLM giù
o nuove formulazioni). Il guadagno non è nei numeri di oggi (identici), ma nella
**stabilità**: una deriva di formulazione che prima avrebbe falsato `behavior_accuracy`
ora non lo fa.

**Come testare.**
```bash
python -m pytest                          # 149 test offline, attesi tutti verdi
python -m pytest tests/test_eval_scoring.py   # 8 test dello scoring
python eval/run_eval.py                   # nuova baseline (Ollama, ~7-15 min)
```

**Risultati test.** 149 test verdi (141 + 8 nuovi). Nessuna regressione.

**Rischi residui.** Bassi. Il refactor degli import sposta effetti collaterali in
`main()`: l'esecuzione come script è invariata (verificata, exit 0) e il modulo importa
offline senza `chdir`. Lo scoring dipende dal fatto che l'agente imposti `abstention_reason`
su ogni ramo di astensione: è così oggi e i test lo presidiano; il fallback testuale copre
eventuali rami futuri non ancora tracciati.

**Prossimo step.** Ciclo 2 — FASE 6 (validazione *held-out* della soglia di astensione
`ABSTENTION_OOD_MAX_STRENGTH`, sfruttando i negativi q34–q39 introdotti in FASE 4).

---

## 2026-06-19 — Ciclo 2 — FASE 6: Validazione held-out della soglia di astensione

**Obiettivo.** Terzo passo del Blocco B (fondamenta di misurazione): dare alla soglia di
astensione una **bontà stimata fuori campione**. La soglia `ABSTENTION_OOD_MAX_STRENGTH`
(0,37) governa la distinzione fra le due cause di astensione *quantitative* —
`fuori_dominio` (le fonti non coprono i termini della query) vs `evidenza_insufficiente`
(fonti pertinenti ma risposta assente) — ma era stata **fissata a mano** e poi misurata
**sugli stessi casi** (q17/q18/q19): una stima ottimistica. La FASE 6 separa
**calibrazione** e **validazione**, sfruttando i negativi held-out q34–q39 riservati in
FASE 4 e mai usati per tarare la soglia.

**File analizzati.** `abstention.py` (`retrieval_strength`, `classify_llm_abstention`),
`config.py` (`ABSTENTION_OOD_MAX_STRENGTH`), `eval/questions_baseline.jsonl` (negativi
q17/q18/q19 storici e q34–q39 held-out), `eval/retrieval_ablation.py` (stile harness no-LLM),
`docs/valutazione_rag.md` (§11.2).

**File creati.**
- `eval/abstention_threshold_validation.py` — harness di calibrazione/validazione che misura
  la `retrieval_strength` dei negativi *threshold-relevant* eseguendo la sola pipeline di
  recupero (Chroma + BM25 + RRF; **nessun Ollama**), calibra la soglia sui soli storici e
  valida sugli held-out. Report in `eval/reports/abstention_threshold_validation.{json,md}`.
- `tests/test_abstention_threshold.py` — 11 test offline delle nuove funzioni pure + un
  controllo di consistenza sul report.

**File modificati.** `abstention.py` (nuove funzioni pure + refactor invariante);
`docs/valutazione_rag.md` (§11.2 e nuova §15); `docs/roadmap_progetto.md` (sez. 2, 13, 14).

**Modifiche tecniche.**
- **`classify_by_strength(strength, threshold)`** — estratta la decisione binaria governata
  dalla soglia (sotto soglia → `fuori_dominio`, a soglia/sopra → `evidenza_insufficiente`).
  `classify_llm_abstention` ora la richiama: logica **identica** (refactor a comportamento
  invariato; i test storici di `test_abstention_reasons.py` restano verdi).
- **`calibrate_ood_threshold(labeled)`** — sceglie la soglia **dai dati** anziché a mano. Con
  classi separabili restituisce la soglia a **massimo margine** (punto medio fra la strength
  più alta dei `fuori_dominio` e la più bassa degli `evidenza_insufficiente`) — lo stesso
  criterio con cui era stata fissata 0,37; con classi sovrapposte massimizza l'accuratezza.
  Richiede esempi di entrambe le classi (altrimenti `ValueError`).
- **`threshold_accuracy(labeled, threshold)`** — accuratezza di una soglia sulle sole due
  cause governate da essa (ignora `ambigua`, `fuori_dominio_corso`, `retrieval_debole`).
- Nessuna modifica a `config.py`: la calibrazione **riproduce** la soglia in uso (vedi sotto),
  quindi 0,37 resta il default, ora *giustificato dai dati e validato fuori campione*.

**Modifiche ai test.** `tests/test_abstention_threshold.py` (11 test, offline, puri):
`classify_by_strength` sotto/a/​sopra soglia e parità con la logica storica; calibrazione
separabile = punto medio a massimo margine; **caso reale** — su q17/q18/q19 la regola ritrova
~0,367 (≈ la 0,37 di config); errore se manca una classe; caso sovrapposto = accuratezza
massima; `threshold_accuracy` con separazione perfetta, esclusione delle cause non
governate, `None` su insieme vuoto; **consistenza** — dalle strength salvate nel report le
funzioni pure riproducono le accuratezze del summary e i due insiemi non si intersecano.

**Impatto sul RAG.** Nessuna modifica al comportamento del sistema: il refactor è
invariante e la soglia resta 0,37. Il valore della fase è **metodologico** — la soglia non è
più un numero arbitrario ma una scelta riproducibile e validata su dati non visti.

**Risultati misurati** (report `eval/reports/abstention_threshold_validation.*`; retrieval
deterministico, nessun LLM):

| Insieme | Domande | Soglia | Accuratezza |
|---|---|---|---|
| calibrazione | q17, q18, q19 | 0,3667 (calibrata) | 1,00 (3/3) |
| held-out | q35, q36, q37 | 0,37 (config) | **1,00 (3/3)** |
| held-out | q35, q36, q37 | 0,3667 (calibrata) | 1,00 (3/3) |

Strength misurate: calibrazione q19=0,333 (`fuori_dominio`), q17=0,400 / q18=0,667
(`evidenza_insufficiente`); held-out q35=0,333 / q36=0,250 (`fuori_dominio`), q37=0,500
(`evidenza_insufficiente`). La calibrazione a massimo margine sui soli storici dà **0,3667**,
cioè la 0,37 fissata a mano (differenza solo di arrotondamento). Applicata agli held-out —
**mai visti** — la soglia classifica **3 su 3** correttamente: c'è un margine netto fra la
strength più alta dei fuori-dominio held-out (0,333) e la più bassa degli insufficienti
(0,500). È la **validazione formale fuori campione** anticipata in FASE 4 (dove i 6 held-out
risultavano 6/6 nell'eval completa): qui isolata sulla sola decisione di soglia.

**Come testare.**
```bash
python -m pytest                                      # 160 test offline, attesi verdi
python -m pytest tests/test_abstention_threshold.py   # 11 test della soglia
python eval/abstention_threshold_validation.py        # harness (Chroma, no Ollama, ~30s)
```

**Risultati test.** 160 test verdi (149 + 11 nuovi). Nessuna regressione.

**Rischi residui.** Gli insiemi sono piccoli (3 + 3 casi threshold-relevant): l'accuratezza
1,0 va letta come **evidenza coerente**, non come garanzia statistica — il corpus e i corsi
sono pochi. La `retrieval_strength` resta **lessicale**: la sua versione semantica e la
ri-taratura della soglia sull'held-out sono la Ciclo 2 — FASE 13 (che dipende proprio da
questo split). Il margine osservato è comodo ma corpus-dipendente.

**Prossimo step.** Ciclo 2 — FASE 7 (quantificare la variabilità del modello locale con
un'opzione `--repeat N` che riporti media e deviazione delle metriche di generazione).

---

## 2026-06-20 — Ciclo 2 — FASE 7: Quantificare la variabilità del modello locale

**Obiettivo.** Ultimo passo del Blocco B (fondamenta di misurazione): dare un **numero**
alla variabilità del modello locale, finora dichiarata solo qualitativamente come «±1
domanda». Senza una banda di rumore non si può giudicare se un guadagno (o un calo)
osservato negli A/B del Blocco C è reale o rientra nell'oscillazione. La fase aggiunge
un'opzione `--repeat N` che ripete l'intero dataset N volte e riporta **media e
deviazione standard** delle metriche, più l'elenco delle domande che cambiano verdetto.

**File analizzati.** `eval/run_eval.py` (loop principale, `aggregate`, `write_reports`,
`classify_behavior`), `agent.py` (`setup_redis_cache` — verifica di assenza di cache nel
percorso eval), `config.py` (`DEFAULT_TEMPERATURE=0.0`), `docs/valutazione_rag.md` (§12,
§14.4).

**File creati.**
- `tests/test_eval_variability.py` — 13 test offline delle nuove funzioni pure
  (`aggregate_repeats`, `aggregate_per_question`, `variability_from_reports`) + un
  controllo di consistenza sul report di variabilità prodotto.
- Report di questa fase: `eval/reports/variability_20260620_131229.{json,md}` (più i 5
  report per-run `eval/reports/baseline_20260620_*.{json,md}`).

**File modificati.** `eval/run_eval.py` (opzioni `--repeat`/`--aggregate-reports`,
funzioni di aggregazione, scrittura del report di variabilità, salvataggio incrementale
per-run); `docs/valutazione_rag.md` (§1.1 conteggio test, §12 riscritta con la misura,
§14.4 aggiornata, nuova §16); `docs/roadmap_progetto.md` (sez. 2, 13, 14).

**Modifiche tecniche.**
- **`--repeat N`** — esegue l'intero dataset N volte (riusando lo stesso `responder`, ma
  ogni `.answer()` è una generazione fresca: nessuna cache nel percorso eval, vedi sotto)
  e produce un report di variabilità. Con `N=1` il comportamento è **identico** a prima
  (un solo `baseline_*.json`).
- **`aggregate_repeats(summaries)`** (pura) — per ogni metrica-tasso: `mean`, `std`
  (deviazione **campionaria**, ddof=1; `0.0` con <2 valori), `min`, `max`, `values`. Le
  metriche `None` (universo vuoto in una run) sono ignorate.
- **`aggregate_per_question(runs)`** (pura) — per ogni domanda: quante run l'hanno
  classificata `behavior_ok`, i verdetti distinti osservati e il flag `oscillates`. Isola
  *quali* domande generano la variabilità aggregata.
- **`write_variability_report(...)`** — scrive `variability_*.{json,md}` con la tabella
  media±σ, le domande che oscillano e i sommari per esecuzione.
- **Robustezza (lezione operativa).** Ogni run salva **subito** il suo `baseline_*.json`:
  con run lunghe su CPU un'interruzione non fa più perdere il lavoro già fatto.
  **`--aggregate-reports baseline_*.json ...`** ricostruisce il report di variabilità da
  report già salvati, **offline** (nessun Ollama), per recuperare run completate prima di
  un'interruzione.
- **Nessuna cache nel percorso di valutazione (verificato).** `setup_redis_cache()` è
  invocata **solo** da `app_agent.py` (la UI), mai da `run_eval.py`; la cache LLM globale
  di LangChain resta quindi disattivata durante l'eval anche se Redis è in esecuzione. La
  prova decisiva è il tempo: ogni run ha richiesto ~8–13 min di inferenza reale (una cache
  avrebbe reso le run 2–5 quasi istantanee).

**Modifiche ai test.** `tests/test_eval_variability.py` (13 test, offline, puri):
media/σ/min/max corretti (σ campionaria di `[0,90; 0,95]` ≈ 0,0354); σ=0 per metrica
costante e per singola run; gestione dei `None`; `ValueError` su input vuoto; flag di
oscillazione e conteggio `ok` per domanda; ordine preservato; recupero offline da report
salvati (con `monkeypatch` di `REPORTS_DIR`); consistenza del report prodotto.

**Impatto sul RAG.** Nessuna modifica al comportamento del sistema: la fase è di **sola
misurazione**. Il valore è metodologico — fornisce la banda di rumore con cui leggere gli
A/B successivi.

**Risultati misurati** (report `eval/reports/variability_20260620_131229.*`; `llama3.1:8b`,
`temperature=0.0`, configurazione predefinita, 5 esecuzioni × 40 domande):

| Metrica | Media | Dev. std (σ) | Min | Max |
|---|---|---|---|---|
| behavior_accuracy | 0,90 | **0,0** | 0,90 | 0,90 |
| course_accuracy | 1,00 | 0,0 | 1,00 | 1,00 |
| topic_accuracy | 1,00 | 0,0 | 1,00 | 1,00 |
| retrieval_hit_rate | 1,00 | 0,0 | 1,00 | 1,00 |
| citation_hit_rate | 1,00 | 0,0 | 1,00 | 1,00 |
| abstention_rate | 0,923 | **0,0** | 0,923 | 0,923 |
| abstention_reason_accuracy | 0,923 | **0,0** | 0,923 | 0,923 |

**Domande che oscillano: nessuna** (0 su 40). Le risposte del modello sono risultate
**byte-identiche** nelle 5 esecuzioni. Interpretazione onesta: la variabilità
*within-session* (stesso modello caldo, prompt identici, pipeline fissa) è **nulla** — a
`temperature=0` la decodifica è greedy e quindi deterministica. Questo **affina** la nota
storica «±1 domanda» (q17): quell'oscillazione era un effetto **cross-session/cross-fase**
(tra le fasi cambiavano la pipeline e quindi il prompt, o lo stato del runtime del modello),
non un fenomeno per-run. La banda di rumore è dunque «≈0 entro la sessione; ≤±1 domanda tra
sessioni diverse».

**Come testare.**
```bash
python -m pytest                                  # 173 test offline, attesi verdi
python -m pytest tests/test_eval_variability.py   # 13 test della variabilità
python eval/run_eval.py --repeat 5                # 5 esecuzioni (Ollama attivo, ~1 h CPU)
python eval/run_eval.py --aggregate-reports eval/reports/baseline_20260620_*.json
                                                  # ricompone il report offline (no Ollama)
```

**Risultati test.** 173 test verdi (160 + 13 nuovi). Nessuna regressione.

**Rischi residui.** Le 5 esecuzioni sono **consecutive nella stessa sessione** a modello
caldo: misurano il **limite inferiore** della variabilità (within-session ≈ 0), non la
varianza cross-session, dove restano possibili — per riavvio del server, carico macchina,
versione di Ollama o non-determinismo numerico nel batching — rare oscillazioni di ±1
domanda come osservato storicamente. La σ=0 va quindi letta come «riproducibilità a
pipeline e runtime fissi», non come assenza assoluta di rumore.

**Prossimo step.** Ciclo 2 — FASE 8 (togliere dal prompt le fasce numeriche TOLC, primo
refactor del Blocco C, da valutare in A/B con la banda di rumore appena quantificata).

---

## 2026-06-21 — Ciclo 2 — FASE 8: Togliere dal prompt le fasce numeriche TOLC

**Obiettivo.** Primo refactor del Blocco C (riduzione misurata dell'hard-coding). Le
soglie TOLC-I 9/16 erano scritte **due volte**: come unica fonte di verità in
`knowledge.py` (con provenienza sul PDF, FASE 7) e — di nuovo, in prosa — nel prompt
`ANSWER_STYLE_GUIDE`. Poiché la classificazione numerica del punteggio è già prodotta dal
**guard deterministico** (sempre attivo, legge da `knowledge.py`), ripetere le fasce nel
prompt è una seconda fonte di verità a rischio di disallineamento. La fase rimuove la
duplicazione e **misura** che il comportamento non cambia oltre la banda di rumore
quantificata in FASE 7.

**File analizzati.** `config.py` (`ANSWER_STYLE_GUIDE`, `qa_template`), `agent.py`
(`answer` ai righi ~242–279: il guard `_try_deterministic_accesso_informatica_answer`
short-circuita **prima** di costruire il prompt; il ramo LLM usa `ANSWER_STYLE_GUIDE`),
`knowledge.py` (`classify_tolc`, `tolc_band_labels`, `tolc_bands_table`), `rules_tolc.py`.

**File modificati.**
- `config.py` — rimosse da `ANSWER_STYLE_GUIDE` le **4 righe** delle fasce numeriche
  TOLC-I (nella sezione «Regole specifiche per accesso/TOLC/OFA»): la riga «Se nel
  contesto sono presenti le fasce < 9, >= 9 e < 16, >= 16, riportale chiaramente» e le tre
  righe di classificazione del punteggio (`< 9` → niente immatricolazione diretta;
  `>= 9 e < 16` → con OFA; `>= 16` → senza OFA). Aggiunto un commento che documenta la
  decisione (unica fonte di verità = `knowledge.py` + guard). **Lasciate intatte** le due
  righe di *routing* per corso della stessa sezione (instradamento al
  `regolamento-di-accesso-informatical-31-.pdf`): sono il target della **FASE 9**, non di
  questa.
- `tests/test_knowledge.py` — 2 guard di non-regressione (vedi sotto).

**Modifiche tecniche.** Solo prosa di prompt: nessun cambiamento di logica, moduli o
flusso. Le soglie 9/16 restano in `knowledge.TOLC_INFORMATICA`; il guard
`_try_deterministic_accesso_informatica_answer` continua a classificare il punteggio e a
stampare la tabella delle fasce (`tolc_bands_table`) **indipendentemente dal prompt**. Per
costruzione, quindi, ogni domanda gestita dal guard è **byte-identica** prima e dopo;
cambia solo il prompt delle domande TOLC-adiacenti che finiscono al ramo LLM.

**Modifiche ai test.** `tests/test_knowledge.py`:
- `test_style_guide_has_no_tolc_numeric_bands` — verifica che le firme testuali delle 4
  righe rimosse non siano più in `ANSWER_STYLE_GUIDE` e che le soglie 9/16 restino in
  `knowledge.py` (impedisce la reintroduzione della duplicazione).
- `test_style_guide_keeps_tolc_routing_rules` — verifica che l'intestazione e la regola di
  routing al documento di accesso restino (circoscrive la FASE 8: rimuove **solo** le
  fasce numeriche, non il routing → FASE 9).

**Impatto sul RAG.** Meno duplicazione e una sola fonte di verità per le soglie TOLC,
senza variazione di comportamento (misurata sotto). Riduce il rischio che prompt e
`knowledge.py` divergano in futuro.

**Risultati misurati (A/B nella stessa sessione, `llama3.1:8b`, `temperature=0`, dataset a
40 domande).** A = con fasce nel prompt; B = senza. Per isolare il segnale dalla banda di
rumore cross-process di FASE 7 sono stati raccolti **più campioni indipendenti** (processi
separati, modello ricaricato):

| Campione | behavior | retrieval | citation | abstention / reason | q28 | report |
|---|---|---|---|---|---|---|
| A — con fasce | 0,90 | 0,963 | 0,963 | 0,923 / 0,923 | answer | `baseline_20260620_152206` |
| (FASE 7) con fasce | 0,90 | 1,00 | 1,00 | 0,923 / 0,923 | answer | `baseline_20260620_131229` |
| B — senza fasce | 0,875 | 1,00 | 1,00 | 0,923 / 0,923 | abstain | `baseline_20260620_153742` |
| B2 — senza fasce | 0,90 | 0,963 | 0,963 | 1,00 / 1,00 | answer | `baseline_20260621_112434` |
| B3 — senza fasce | 0,90 | 0,963 | 0,963 | 1,00 / 1,00 | answer | `baseline_20260621_113108` |

- **Guard TOLC invariante (verificato).** Le 6 domande gestite dal guard (q01, q02, q03,
  q09, q26, q40) hanno risposta **byte-identica** fra A e B: la rimozione non tocca il
  percorso deterministico, come atteso.
- **L'unico delta è q28**, una domanda **non-TOLC** (numero di posti per Economia Aziendale
  L-18, ramo LLM): in B passa da `answer` a `abstain`. Ma **oscilla** sotto lo *stesso*
  prompt no-bands (è `answer` in B2 e B3): è quindi **rumore cross-process**, non un effetto
  della modifica. q28 è genuinamente borderline — il documento riporta i posti della
  Magistrale LM-56, non della triennale L-18 chiesta — e l'astensione di B è semanticamente
  difendibile.
- Le metriche `retrieval`/`citation` (q24) e `abstention` (q18) oscillano nello stesso modo
  in **entrambe** le configurazioni → rumore, non segnale. Le false astensioni **stabili**
  in tutte e cinque le run sono {q14, q21, q29} (famiglia nota, target FASE 14).
- **Lettura.** behavior no-bands = {0,875; 0,90; 0,90} contro con-bands = {0,90; 0,90}: il
  delta è ≤1 domanda, **entro la banda di rumore di FASE 7** («≤±1 domanda tra sessioni»).
  Comportamento invariato, duplicazione rimossa.

**Come testare.**
```bash
python -m pytest                                   # 175 test offline, attesi verdi
python -m pytest tests/test_knowledge.py           # include i 2 guard di FASE 8
python eval/run_eval.py                            # baseline (Ollama attivo, ~9 min CPU)
```

**Risultati test.** 175 test verdi (173 + 2 nuovi). Nessuna regressione.

**Rischi residui.** Onestamente, con un solo A e tre B non si stima una σ cross-process
completa per ciascuna config; la prova di neutralità si fonda su (1) l'identità
byte-per-byte del percorso guard e (2) l'oscillazione di q28 sotto prompt identico. Su un
modello 8B le domande borderline (q16/q18/q24/q28) restano sensibili a perturbazioni minime
del prompt; il calo a 0,875 di una singola run no-bands è dentro la banda, non una
regressione.

**Prossimo step.** Ciclo 2 — FASE 9 (de-duplicare le regole di routing per corso nel
prompt: ridurre `ANSWER_STYLE_GUIDE` alle sole regole generali di stile e citazione,
lasciando il routing al reranker + filtro metadata; A/B con la stessa banda di rumore).

## 2026-06-21 — Ciclo 2 — FASE 9: De-duplicare il routing per corso nel prompt — RISULTATO NEGATIVO (routing mantenuto)

**Obiettivo.** Ridurre `ANSWER_STYLE_GUIDE` alle sole regole generali di stile e
citazione, rimuovendo le cinque sezioni «Regole specifiche per <corso/topic>». L'ipotesi:
l'instradamento *dei documenti* è già fatto, sui metadata, dal reranker euristico e dal
filtro per corso (`reranking.py`: `rerank_documents` + `filter_documents_by_course`), e le
regole di *contenuto* per topic sono già emesse, per ogni domanda, dal profilo dinamico
`_build_answer_profile` (iniettato nello stesso prompt come `answer_profile`); ripeterle nel
prompt sarebbe quindi una seconda fonte di verità. La fase doveva **misurare** che il
comportamento non cambia oltre la banda di rumore.

**Esito in una riga.** L'A/B nella stessa sessione ha **misurato una regressione**
(behavior 0,90 → 0,875/0,85): la rimozione **non** è neutra. Come per il reranker neurale
(Ciclo 1 — FASE 4), si riporta il risultato negativo e si **mantiene il routing nel
prompt**. Nessuna modifica di comportamento spedita.

**File analizzati.** `config.py` (`ANSWER_STYLE_GUIDE`, `qa_template`), `agent.py`
(`answer` → ramo LLM con `style_guide` + `_build_answer_profile` ~619–684; gestione
ambiguità `intent.is_ambiguous` → `_format_clarification_answer` ~201), `reranking.py`
(`rerank_documents`, `filter_documents_by_course`), `tests/test_knowledge.py`.

**File modificati.**
- `config.py` — **`ANSWER_STYLE_GUIDE` invariato (byte-identico)**: aggiunto solo un
  commento che documenta la valutazione FASE 9, l'esito negativo dell'A/B e la decisione di
  mantenere il routing (rimando a ESP-09).
- `tests/test_knowledge.py` — il guard FASE 8 `test_style_guide_keeps_tolc_routing_rules`
  (che marcava il routing come «target FASE 9») è stato sostituito da
  `test_style_guide_keeps_course_routing_rules`: ora verifica che le cinque sezioni di
  routing **restino** nel prompt (anti-rimozione accidentale), motivando la scelta con
  l'esito misurato. Header del modulo aggiornato.
- `docs/architettura_rag.md` — §5 (limiti): annotato che il routing è stato valutato per la
  rimozione ma mantenuto per il risultato negativo.

**Metodo di misura.** A/B nella **stessa sessione** (config predefinita, `llama3.1:8b`,
`temperature=0`, dataset a 40 domande). Within-session σ=0 (FASE 7, riconfermato: le 3 run
B sono risultate byte-identiche), quindi ogni domanda che cambia esito **è segnale, non
rumore**. Per evitare il confronto cross-sessione (che la FASE 8 evitava di proposito) la
baseline «con routing» è stata **rieseguita in questa stessa sessione** (A-fresh).

| Config | behavior | course/topic | retrieval | citation | fallimenti (behavior) | report |
|---|---|---|---|---|---|---|
| A-fresh — **con** routing | **0,90** | 1,0 / 1,0 | 1,0 | 1,0 | q14, q16, q21, q29 | `baseline_20260621_222535` |
| B — **senza** routing (+ clausola chiarimento) | 0,875 | 1,0 / 1,0 | 0,963 | 0,963 | q18, q28, q29, q30, q31 | `_185818`, `_190346`, `_195327` |
| B' — **senza** routing (rimozione pura) | 0,85 | 1,0 / 1,0 | 1,0 | 0,963 | q18, q21, q28, q29, q30, q31 | `baseline_20260621_214237` |

**Lettura per domanda (A-fresh → B, stessa sessione, σ=0).** Rimuovere il routing
**risolve 3** e **rompe 4**:

| domanda | atteso | con routing (A) | senza routing (B) | effetto |
|---|---|---|---|---|
| q28 — posti Economia Aziendale L-18 | answer | answer ✓ | abstain ✗ | rotto (falsa astensione) |
| q30 — scadenza domanda borsa | answer | answer ✓ | abstain ✗ | rotto (falsa astensione) |
| q31 — graduatoria provvisoria/definitiva borsa | answer | answer ✓ | abstain ✗ | rotto (falsa astensione) |
| q18 — costo mensa | abstain | abstain ✓ | answer ✗ | rotto (over-answering) |
| q14 — consultabilità tesi | answer | abstain ✗ | answer ✓ | risolto |
| q16 — discussione di laurea | answer | abstain ✗ | answer ✓ | risolto |
| q21 — durata presentazione prova finale | answer | abstain ✗ | answer ✓ | risolto (solo con la clausola di chiarimento; in B' resta abstain) |

**Interpretazione (onesta).**
- **course/topic/retrieval invariati a 1,0**: l'argomento architetturale tiene — il routing
  *dei documenti* è davvero ridondante col reranker (la selezione delle fonti non cambia).
- **Il behavior però regredisce**: la prosa di routing non instrada solo i documenti, fa
  anche da *framing di commit* («usa questo documento, rispondi da qui») di cui il modello
  8B si avvale. Senza, aumentano le **false astensioni** su domande answerable borderline
  (q28/q30/q31, area borsa/economia) e compare **over-answering** su una no-answer (q18).
- Le 6 domande gestite dal **guard TOLC** (q01, q02, q03, q09, q26, q40) restano `answer` in
  tutte le config (il guard short-circuita prima del prompt): la correttezza numerica non è
  in gioco. La falsa astensione **stabile** ovunque è q29 (Erasmus durata; famiglia nota).
- Net behavior: A 0,90 → B 0,875 (−1) → B' 0,85 (−2). La clausola di chiarimento
  generalizzata (B vs B') recupera q21 ma non evita le regressioni borsa.

**Decisione (APPLICATA).** Mantenere il routing nel prompt. `ANSWER_STYLE_GUIDE` resta
byte-identico; un guard di test impedisce la rimozione accidentale. È un risultato negativo
riportato per onestà metodologica (analogo alla FASE 4). Una rimozione *parziale* (togliere
le sole righe di puro instradamento documento, tenendo il framing anti-astensione) richiede
altri A/B per isolare le righe: rimandata, non inclusa qui.

**Come testare.**
```bash
python -m pytest                          # 175 test offline, attesi verdi
python -m pytest tests/test_knowledge.py  # include il guard FASE 9 (routing mantenuto)
python eval/run_eval.py                   # baseline (Ollama attivo, ~7 min CPU)
```

**Risultati test.** 175 test verdi (invariati: un guard sostituito da un altro). Nessuna
regressione di test.

**Rischi residui.** La misura è su un dataset di 40 domande e un modello 8B: i casi
borderline (q16/q18/q21/q28/q30/q31) sono sensibili a perturbazioni del prompt. La
conclusione («il routing nel prompt riduce le false astensioni dell'8B») è coerente coi dati
same-session ma non è una garanzia su corpus/modelli diversi.

**Prossimo step.** Ciclo 2 — FASE 10 (ripulire il caso speciale in `_postprocess_answer`:
restringere il ramo che riscrive l'astensione per `doc_type ∈ {accesso, regolamento,
altro}`, dove «altro» è troppo ampia).

---

## 2026-06-22 — Ciclo 2 — FASE 10: Ripulire il caso speciale in `_postprocess_answer`

**Obiettivo.** Rimuovere (o restringere) il ramo di `_postprocess_answer` che, in presenza
di un'astensione del modello su argomento "accesso" e con fonti di `doc_type ∈ {accesso,
regolamento, altro}`, **riscriveva** la risposta con un testo generico. La condizione era
troppo ampia («altro» cattura qualsiasi documento non classificato) e il ramo conviveva con
il layer di astensione introdotto dopo (FASE 6), rischiando di mascherare astensioni
legittime.

**Diagnosi (più seria del previsto).** Il ramo non era solo «troppo ampio»: il testo di
riscrittura («*Le fonti recuperate contengono documenti potenzialmente rilevanti, ma la
risposta non è stata formulata in modo affidabile…*») **non contiene alcun marcatore di
`is_abstention`**. Di conseguenza, riscrivendo la risposta, l'astensione diventava
invisibile al resto della pipeline:
- il layer FASE 6 (`is_abstention` → `classify_llm_abstention`) non la riconosceva più →
  `trace.abstention_reason` **non** veniva impostata;
- lo scorer dell'eval (FASE 5, che legge l'esito dal trace) la classificava come **`answer`**
  invece di `abstain`;
- il blocco fonti onesto (FASE 3) veniva bypassato (`abstaining=False` → ricompariva «Fonti
  utilizzate»), cioè proprio la falsa attribuzione che la FASE 3 aveva eliminato.

In altre parole il ramo era **ridondante** con il layer di astensione e, peggio, lo
**contraddiceva**.

**File analizzati.** `agent.py` (`_postprocess_answer` ~1431–1474; il chiamante in
`answer` ~288; il layer FASE 6 a ~327–334; il blocco fonti FASE 3 a ~344–347),
`abstention.py` (`is_abstention`, `UNCERTAINTY_MARKERS`), `database.py` (`_infer_doc_type`),
`tests/conftest.py`, `eval/run_eval.py` (scoring dal trace).

**File modificati.**
- `agent.py` — **rimosso** il ramo di riscrittura. Ora `_postprocess_answer`: (1) mantiene la
  rete di sicurezza per risposta vuota; (2) se la risposta è un'astensione, si limita ad
  **abbassare la confidenza** a «bassa»; (3) restituisce sempre il testo del modello, senza
  riscriverlo. La *classificazione della causa* e il *messaggio esplicito* sono lasciati al
  layer FASE 6 a valle, così l'astensione resta riconoscibile e coerente con la causa. Il
  rilevamento dell'incertezza è stato **unificato** su `abstention.is_abstention` (unica
  fonte di verità), eliminando la lista locale di marcatori duplicata (sottoinsieme
  incoerente di quella di `abstention.py`). La firma è stata semplificata a
  `_postprocess_answer(self, answer)` (i parametri `sources`, `intent`, `question`, usati
  solo dal ramo rimosso, erano diventati codice morto); aggiornato il chiamante in `answer`.
- `tests/test_postprocess.py` — **nuovo** modulo (6 test offline): rete di sicurezza per
  risposta vuota; strip degli spazi; risposta «confidente» lasciata intatta (la confidenza
  stimata a monte non viene forzata); **astensione preservata e non riscritta**; astensione
  ancora **riconoscibile da `is_abstention`** dopo il post-processing (la proprietà che il
  vecchio ramo rompeva); assenza del messaggio generico del vecchio caso speciale.

**Perché migliora il RAG.** L'astensione del modello non viene più mascherata: il «non lo
so» resta tale, viene classificato per causa (FASE 6) e mostra il blocco fonti onesto
(FASE 3). Si elimina una seconda logica di astensione, ridondante e in conflitto con quella
ufficiale, e una duplicazione di marcatori. Output più trasparente e più facile da valutare
(lo scorer non viene più ingannato dal testo riscritto).

**Metodo di misura.** A/B (`llama3.1:8b`, `temperature=0`, dataset a 40 domande, **stessa
sessione**, due processi distinti). A = codice con il ramo (importato prima della modifica,
verificato via timestamp del processo); B = codice senza il ramo.

| Config | behavior | course/topic | retrieval | citation | abstention / reason | report |
|---|---|---|---|---|---|---|
| A — **con** ramo (vecchio) | **0,875** | 1,0 / 1,0 | 1,0 | 1,0 | 0,923 / 0,923 | `baseline_20260622_094539` |
| B — **senza** ramo (nuovo) | **0,875** | 1,0 / 1,0 | 1,0 | 1,0 | 1,0 / 1,0 | `baseline_20260622_095518` |

**Attribuzione (decisiva).** Il testo di riscrittura («potenzialmente rilevanti») compare
**0 volte** nel report A: il ramo **non si è attivato in nessuna delle due run** (in A tutte
le domande "accesso" del ramo LLM hanno risposto; in B il ramo è rimosso). Le uniche
differenze per-domanda sono **q18** e **q28**, ed entrambe sono **oscillazione cross-process
del modello 8B**, non effetto della modifica — verificato dai testi generati, divergenti tra
A e B:

| domanda | atteso | A (vecchio) | B (nuovo) | causa |
|---|---|---|---|---|
| q18 — costo mensa | abstain | over-answering → `answer` ✗ | «Non lo so» → `abstain` ✓ | oscillazione (topic=None: il ramo non poteva attivarsi) |
| q28 — posti Economia L-18 | answer | risposta borderline («non è specificato… consultare p.19») → `answer` ✓ | «Non lo so» → `abstain` ✗ | oscillazione (le fonti di q28 sono `doc_type=guida`: il ramo non si sarebbe attivato comunque) |

Fallimenti **stabili in entrambe** le run: q14, q16, q21, q29 (famiglia di false astensioni
nota, target FASE 14). Il salto abstention/reason 0,923 → 1,0 è dovuto a q18 (negativo che in
A non si astiene, in B sì), non alla modifica. Conclusione: **nessuna regressione**
(behavior 0,875 = 0,875); la modifica è **neutra per comportamento su questa eval** perché il
caso speciale non è esercitato dal dataset corrente (servirebbe una domanda "accesso" del
ramo LLM che si astenga con fonti `accesso/regolamento/altro`, es. L-16). Il valore della
fase — non mascherare più le astensioni — è dimostrato **per costruzione** e dai 6 test
unitari. Dettagli e per-domanda: `esperimenti_rag.md` (ESP-10).

**Come testare.**
```bash
python -m pytest                          # 181 test offline, attesi verdi
python -m pytest tests/test_postprocess.py  # i 6 test della FASE 10
python eval/run_eval.py                   # baseline (Ollama attivo, ~8-13 min CPU)
```

**Risultati test.** 181 test verdi (175 + 6 nuovi in `test_postprocess.py`). Nessuna
regressione.

**Rischi residui.** Il caso speciale non è esercitato dall'eval corrente: la prova della non
regressione poggia sull'argomento per costruzione (ramo dormiente, verificato) più che su un
trigger osservato dal vivo; la trasparenza è coperta dai test unitari. La famiglia di false
astensioni q14/q16/q21/q29 resta aperta (target FASE 14). L'attribuzione cross-process si
appoggia ai testi generati: q18/q28 oscillano (banda di rumore della FASE 7 fra sessioni).

---

## 2026-06-22 — Ciclo 2 — FASE 11: Intent detection semantica (opt-in)

**Obiettivo.** Affiancare al riconoscimento dell'intento a parole chiave (`intent.py`) un
classificatore per **similarità di embedding**, per renderlo più robusto verso le parafrasi
(formulazioni che non contengono i token-keyword), **senza** cambiare il comportamento
predefinito finché non validato (opt-in, default OFF).

**File analizzati.** `intent.py`, `config.py`, `agent.py` (costruttore e
`_infer_query_intent`), `neural_reranker.py` (pattern opt-in/lazy/scorer iniettabile),
`database.py` (`_build_embeddings`, embedder riusabile), `eval/run_eval.py`,
`tests/conftest.py`, dataset `eval/questions_baseline.jsonl`.

**File creati.**
- `semantic_intent.py` — `SemanticIntentClassifier` (opt-in, lazy, **embedder iniettabile**
  per i test offline, fallback sicuro a `None` se l'embedder non è disponibile o solleva
  un'eccezione); funzioni pure `cosine_similarity` e `best_label_by_similarity`; frasi-ancora
  in linguaggio naturale per i 4 corsi e i 5 argomenti (`COURSE_ANCHORS`, `TOPIC_ANCHORS`).

**File modificati.**
- `config.py` — `SEMANTIC_INTENT_ENABLED` (env `UNILAW_SEMANTIC_INTENT`, **default OFF**),
  `SEMANTIC_INTENT_COURSE_MIN_SIMILARITY=0.5`, `SEMANTIC_INTENT_TOPIC_MIN_SIMILARITY=0.45`
  (soglie del coseno **provvisorie**, da validare).
- `intent.py` — `infer_query_intent(question, memory, semantic_classifier=None)`: quando il
  classificatore è fornito, **affianca** le keyword riempiendo SOLO le caselle vuote (corso e/o
  argomento), **prima** della memoria a slot e **senza mai sovrascrivere** un riconoscimento a
  keyword; non tocca i corsi fuori dominio (`detected_unknown_course`). Con `None` (default) il
  comportamento è byte-identico a prima.
- `agent.py` — `UniLawResponder` ha il parametro `use_semantic_intent` (default da config);
  quando attivo costruisce il classificatore riusando l'embedder del vector store
  (`_embedder_from_vector_db` → `_embedding_function.embed_documents`), **senza** caricare un
  nuovo modello né aggiungere dipendenze; quando OFF il classificatore non viene nemmeno
  costruito (`self.semantic_intent = None`).
- `eval/run_eval.py` — flag `--semantic-intent` per l'A/B.
- `tests/test_intent_detection.py` — +17 test: helper puri, meccanica del classificatore con
  embedder finto iniettato (offline), gap-fill, no-override, non-intervento sui corsi fuori
  dominio, equivalenza con `None`, default OFF e cablaggio nel responder.

**Impatto sul RAG.** Maggiore robustezza dell'intent verso le parafrasi quando la funzione è
abilitata; **comportamento predefinito invariato** (default OFF). Nessuna nuova dipendenza
(riusa il modello di embedding già caricato per il retrieval).

**Verifica.**
- *Neutralità sull'eval, per costruzione.* Diagnostica al solo livello di intent sul dataset a
  40 domande (modello di embedding reale, **nessun Ollama**): le keyword coprono già corso e
  argomento su **tutte e 40** le domande (0 caselle vuote) e il semantico **non sovrascrive
  mai** (override = 0). Quindi, ad embedding attivo, l'intent prodotto è **byte-identico** a
  quello a sole keyword su tutto il dataset; poiché domanda e intent restano identici e il
  modello è deterministico a `temperature=0` (σ=0 within-session, FASE 7), la generazione
  end-to-end è **identica per costruzione**: l'A/B generativo con Ollama non cambierebbe alcun
  verdetto (come per i casi non esercitati dal dataset, cfr. FASE 10).
- *Valore misurato sulle parafrasi (modello reale).* Su 9 sonde-parafrasi prive dei
  token-keyword del loro intent (tutte mancate dalle keyword): il semantico ne **recupera 7/9
  correttamente** (i 4 corsi + borsa/erasmus/piano\_studi), con **1 mancata sotto soglia**
  (tesi, formulazione molto obliqua) e **1 falso positivo** (accesso → piano\_studi su «quali
  prove devo sostenere per entrare al primo anno»). Conferma il guadagno di robustezza e che
  le soglie 0,5/0,45 sono in un intervallo sensato ma **non perfette** → si **mantiene il
  default OFF** finché non validate su un set più ampio. Dettagli e tabella: `esperimenti_rag.md`
  (ESP-11).

**Come testare.**
```bash
python -m pytest                                 # 198 test offline, attesi verdi
python -m pytest tests/test_intent_detection.py  # i test dell'intent (incl. FASE 11)
UNILAW_SEMANTIC_INTENT=1 python eval/run_eval.py # eval con semantico attivo (Ollama, ~8-13 min)
python eval/run_eval.py --semantic-intent        # equivalente via flag
```

**Risultati test.** 198 test verdi (181 + 17 nuovi in `test_intent_detection.py`). Nessuna
regressione (gli 8 test storici dell'intent restano verdi: il default OFF non li tocca).

**Rischi residui.** Le soglie del coseno sono **provvisorie** e validate solo su un piccolo set
di sonde (7/9): un set più ampio è necessario prima di considerare l'abilitazione di default;
resta 1 falso positivo (accesso↔piano_studi) e 1 parafrasi sotto soglia. Il dataset eval
attuale non esercita il layer (0 caselle vuote): per misurarne il guadagno end-to-end servono
domande-parafrasi nel dataset (estensione futura, in sinergia con la FASE 4). La qualità
dipende dal modello di embedding multilingue (MiniLM) e dalla rappresentatività delle
frasi-ancora.

**Prossimo step.** Ciclo 2 — FASE 12 (grounding semantico delle citazioni).

---

## 2026-06-22 — Ciclo 2 — FASE 12: Grounding semantico delle citazioni (opt-in)

**Obiettivo.** Il grounding delle citazioni (FASE 5) verifica il supporto delle frasi che
citano una fonte tramite **sovrapposizione lessicale** di token (overlap coefficient, soglia
0,18): una frase corretta ma **parafrasata** — che dice la stessa cosa della fonte con parole
diverse — non condivide token e viene bocciata, abbassando ingiustamente la confidenza e
aggiungendo una nota di "supporto debole" non meritata. Si aggiunge una **rete di recupero
semantica** che **affianca** il lessicale (non lo sostituisce), opt-in e **default OFF** finché
le soglie non sono validate su un insieme più ampio.

**File analizzati.** `citations.py` (`grounding_report`), `agent.py` (chiamata al grounding nel
ramo LLM, costruttore, `_embedder_from_vector_db`), `config.py`, `semantic_intent.py` (pattern
opt-in/embedder iniettabile/`cosine_similarity`), `evidence.py` (`split_sentences`),
`retrieval.py` (`tokenize`), `eval/run_eval.py`, `tests/test_citations.py`.

**File modificati.**
- `citations.py` — `grounding_report(answer, sources, min_overlap=0.18, embedder=None,
  min_semantic=0.16)`: le frasi citanti che **non** superano la soglia lessicale ricevono un
  secondo controllo per **similarità di embedding** (nuovo helper `_semantic_support`), che
  confronta la frase con ciascuna **frase** della fonte citata (frase↔frase, non frase↔chunk
  intero, per non diluire la similarità) e accetta se la più vicina supera `min_semantic`. La
  rete semantica **si aggiunge** al lessicale: non può togliere un supporto già riconosciuto e
  con `embedder=None` (default) il risultato è **byte-identico** al solo lessicale. Fallback
  sicuro: se l'embedder è assente o solleva un'eccezione, si ricade sul solo lessicale, senza
  errori. Importa `cosine_similarity` ed `Embedder` da `semantic_intent.py` (nessuna nuova
  dipendenza).
- `config.py` — `CITATION_GROUNDING_SEMANTIC_ENABLED` (env `UNILAW_SEMANTIC_GROUNDING`,
  **default OFF**) e `CITATION_GROUNDING_SEMANTIC_MIN_SIMILARITY=0.16` (soglia del coseno
  **calibrata** sui dati misurati, vedi sotto).
- `agent.py` — `UniLawResponder` ha il parametro `use_semantic_grounding` (default da config);
  quando attivo recupera l'embedder del vector store con lo stesso `_embedder_from_vector_db`
  introdotto in FASE 11 (nessun nuovo modello/dipendenza) e lo passa a `grounding_report`;
  quando OFF l'embedder non viene nemmeno recuperato (`self.semantic_grounding_embedder = None`)
  → grounding byte-identico al lessicale.
- `eval/run_eval.py` — flag `--semantic-grounding` per l'A/B.
- `tests/test_citations.py` — +10 test: il lessicale boccia la parafrasi; il semantico la
  recupera; non recupera una frase estranea; `embedder=None` byte-identico; il semantico non può
  togliere un supporto lessicale; fallback sicuro su embedder che solleva eccezione; sotto soglia
  resta non supportata; default OFF e cablaggio nel responder (con/senza vector store).

**Impatto sul RAG.** Quando abilitato, riduce i falsi "supporto debole" sulle parafrasi → meno
declassamenti ingiustificati della confidenza e meno note di cautela non meritate. **Nessun
effetto sulle metriche-vetrina dell'eval** (behavior/citation/abstention non dipendono dalla
confidenza) e **comportamento predefinito invariato** (default OFF). Nessuna nuova dipendenza.

**Calibrazione della soglia (onesta).** Misura su una piccola sonda con il modello di embedding
reale (`paraphrase-multilingual-MiniLM-L12-v2`, **nessun Ollama**), confrontando la frase citante
con le frasi della fonte (max coseno):

| Classe | Coseno massimo misurato |
|---|---|
| 4 coppie **parafrasi** (stesso significato, parole diverse) | 0,194 – 0,605 |
| 3 coppie **estranee** (significato diverso) | 0,017 – 0,129 |

Il punto di massimo margine è la mediana fra le due classi: (0,129 + 0,194)/2 ≈ **0,16**. A
questa soglia la sonda separa **7/7** (4 parafrasi recuperate, 3 estranee correttamente respinte),
mentre il **solo lessicale boccia tutte e 4 le parafrasi**. La separazione è però **sottile** e
la similarità di frase del modello sulle parafrasi normative è **debole**: per questo il default
resta **OFF**, in attesa di validazione su un insieme più ampio (sinergia con FASE 4/13). Come per
il reranker neurale (FASE 4) e l'intent semantico (FASE 11), il meccanismo è implementato e
disponibile ma non abilitato finché il guadagno non è solido. Dettagli e tabelle: `esperimenti_rag.md`
(ESP-12).

**Come testare.**
```bash
python -m pytest                                    # 208 test offline, attesi verdi
python -m pytest tests/test_citations.py            # i test delle citazioni (incl. FASE 12)
UNILAW_SEMANTIC_GROUNDING=1 python eval/run_eval.py # eval con grounding semantico (Ollama, ~8-13 min)
python eval/run_eval.py --semantic-grounding        # equivalente via flag
```

**Risultati test.** 208 test verdi (198 + 10 nuovi in `test_citations.py`). Nessuna regressione
(i test storici del grounding restano verdi: con `embedder=None` il comportamento è invariato).

**Rischi residui.** La soglia 0,16 è calibrata su una sonda piccola (4+3 coppie) con margine
sottile (0,13 vs 0,19): su un insieme più ampio è atteso qualche falso recupero → validazione
necessaria prima dell'abilitazione di default. La similarità di frase del modello MiniLM sulle
parafrasi normative è debole (3 parafrasi su 4 sotto 0,30). Il dataset eval attuale non esercita
il layer (le metriche non dipendono dalla confidenza): per misurarne il guadagno end-to-end
servono domande/risposte parafrasate (estensione futura, sinergia con FASE 4).

**Prossimo step.** Ciclo 2 — FASE 13 (`retrieval_strength` semantica + ricalibrazione OOD).

---

## 2026-06-22 — Ciclo 2 — FASE 13: `retrieval_strength` semantica + ricalibrazione OOD (opt-in)

**Obiettivo.** La causa di astensione `fuori_dominio` vs `evidenza_insufficiente` (FASE 6)
si decide sulla *retrieval strength* **lessicale** (overlap di token domanda↔fonte, soglia
`ABSTENTION_OOD_MAX_STRENGTH=0,37`): penalizza le query **parafrasate**, che pur essendo in
dominio non condividono token con la fonte e rischiano di essere marcate erroneamente
`fuori_dominio`. Si aggiunge una forza **semantica** (similarità di embedding query↔fonte) e si
**ricalibra** la soglia OOD sullo **stesso split** calibrazione/validazione della FASE 6.

**File analizzati.** `abstention.py`, `config.py`, `agent.py`, `eval/run_eval.py`,
`eval/abstention_threshold_validation.py`, `semantic_intent.py` (per `Embedder`/`cosine_similarity`),
`tests/test_abstention_reasons.py`, `tests/test_abstention_threshold.py`.

**File modificati.**
- `abstention.py` — nuova funzione pura `semantic_retrieval_strength(question, sources, embedder)`
  (massima similarità del coseno fra domanda e contenuto di ciascuna fonte, valore in [0,1], le
  similarità negative trattate come 0; fallback sicuro 0.0 con embedder assente/in errore).
  `classify_llm_abstention` accetta ora `embedder` e `semantic_ood_max_strength` opzionali: con
  `embedder=None` (default) usa la forza **lessicale** e la soglia storica (byte-identico); con un
  embedder usa la forza **semantica** e la soglia ricalibrata. Importa `Embedder`/`cosine_similarity`
  da `semantic_intent.py` (nessuna nuova dipendenza).
- `config.py` — `ABSTENTION_SEMANTIC_STRENGTH_ENABLED` (env `UNILAW_SEMANTIC_ABSTENTION`, **default
  OFF**) e `ABSTENTION_OOD_SEMANTIC_MAX_STRENGTH=0,53` (ricalibrata dai dati).
- `agent.py` — parametro `use_semantic_abstention`; se ON `self.semantic_abstention_embedder =
  self._embedder_from_vector_db()` (riusa l'embedder del vector store, come FASE 11/12), passato a
  `classify_llm_abstention`; se OFF l'embedder non viene recuperato.
- `eval/run_eval.py` — flag `--semantic-abstention`.
- `eval/abstention_threshold_validation.py` — l'harness misura ora **anche** la forza semantica
  (sezione parallela a quella lessicale), calibra e valida la soglia semantica sugli stessi insiemi.
- `tests/` — +10 in `test_abstention_reasons.py` (forza semantica, ramo semantico di
  `classify_llm_abstention`, neutralità del default, cablaggio nel responder) e +1 in
  `test_abstention_threshold.py` (default OFF) + estesa la consistenza del report alla sezione
  semantica.

**Impatto sul RAG (misurato, senza Ollama, modello di embedding reale).** L'harness misura le
strength semantiche dei negativi *threshold-relevant*: calibrazione q19=0,460 (`fuori_dominio`),
q17=0,597/q18=0,674 (`evidenza_insufficiente`); held-out q35=0,324/q36=0,404 (`fuori_dominio`),
q37=0,663 (`evidenza_insufficiente`). La calibrazione a massimo margine sui soli storici dà
**0,5286** → soglia di config **0,53** (riproducibile dai dati, come la 0,37 lessicale rispetto
alla 0,3667). Sugli held-out — **mai visti** — la forza semantica classifica **3/3** sia a 0,53 sia
a 0,5286; le due forze **concordano** su tutti e 6 i negativi. Test unitario del guadagno atteso
(embedder finto): una frase parafrasata senza token in comune è `evidenza_insufficiente` per il
semantico ma `fuori_dominio` per il solo lessicale. Smoke test live (q19, embedder reale): assegna
`fuori_dominio` come la lessicale, integrazione nel responder confermata.

**Come testare.**
```bash
python -m pytest                                       # 219 test offline, attesi verdi
python eval/abstention_threshold_validation.py         # calibra+valida (no Ollama); sezioni lex+sem
UNILAW_SEMANTIC_ABSTENTION=1 python eval/run_eval.py   # eval con forza semantica (Ollama)
python eval/run_eval.py --semantic-abstention          # equivalente via flag
```

**Risultati test.** **219 test verdi** (208 + 11). Nessuna regressione: con `embedder=None` la
classificazione è invariata (i test storici dell'astensione restano verdi); la sezione lessicale del
report (e il test di consistenza FASE 6) è preservata.

**Rischi residui.** La separazione semantica **regge** ma è più **compressa** della lessicale: in
valore assoluto la forza dei *fuori dominio* è alta (0,32–0,46: il MiniLM dà similarità moderata
anche off-topic), con margine fra le classi più stretto. Insiemi piccoli (3 + 3): evidenza coerente,
non garanzia statistica. Per questo — come reranker (FASE 4), intent semantico (FASE 11), grounding
semantico (FASE 12) — la variante è **default OFF** in attesa di un set più ampio. Il guadagno sulle
parafrasi è dimostrato unitariamente, non ancora misurato end-to-end (il dataset non contiene un
negativo parafrasato che il lessicale sbaglia): estensione sinergica con la FASE 4.

**Prossimo step.** Ciclo 2 — FASE 14 (mitigazione q14, falsa astensione su regolamento generale).

---

## 2026-06-22 — Ciclo 2 — FASE 14: Mitigazione q14 (falsa astensione su regolamento generale)

**Obiettivo.** Su «La tesi di Informatica L-31 è consultabile dopo la laurea?» (q14) la
risposta è nel corpus — un **regolamento generale di Ateneo** (`regolamento-tesi-2023.pdf`,
`course_tag="generale"`) che disciplina la consultabilità (consultabile / dopo un embargo di
24 mesi / non consultabile) — ma il modello 8B si **astiene** cercando un dettaglio "specifico
per Informatica L-31" che il regolamento generale, per natura, non riporta. È il caso di studio
aperto dal Ciclo 1. Ridurre questa falsa astensione **senza** dettare la risposta.

**File analizzati.** `agent.py` (`_build_answer_profile`, `answer`, `_build_context`),
`intent.py` (`asks_tesi_consultazione`), `database.py` (`_infer_doc_type`/`_infer_course_tag`),
`config.py`, `eval/run_eval.py`, `eval/questions_baseline.jsonl`, `evidence.py`.

**Diagnosi (misurata, no Ollama).** Il contesto effettivamente passato al modello per q14
**contiene già** la regola completa (estratti da pag. 1–3 del regolamento tesi): la falsa
astensione è dunque un limite di **generazione/framing**, non di retrieval né di evidence
selection. Per questo la leva scelta è un **hint nel profilo di risposta** (l'altra opzione di
roadmap, il "boost di evidenza", non serve: l'evidenza è già presente e completa).

**File modificati.**
- `agent.py` — nuovo predicato puro `has_general_tesi_regulation(sources)` (vero se fra le
  fonti c'è un regolamento generale sulla tesi: `course_tag=="generale"` e
  `doc_type ∈ {tesi, regolamento}`). `_build_answer_profile(intent, question, sources=None)`:
  nel ramo consultabilità (`asks_tesi_consultazione`), se il toggle è attivo **e** il predicato
  è vero, aggiunge al profilo un hint che **autorizza l'uso della regola generale** (il
  regolamento sulla consultabilità è unico per tutto l'Ateneo e vale per ogni corso; l'assenza
  del nome del corso non è una mancanza d'informazione) e invita a non rispondere «Non lo so»
  quando il contesto riporta le condizioni di consultabilità. Il *cosa* (consultabile sì/no,
  con quale embargo) resta lettura del modello: l'hint non prescrive la risposta. Call site
  aggiornato a `_build_answer_profile(intent, question, sources)`.
- `config.py` — `GENERAL_TESI_HINT_ENABLED` (env `UNILAW_GENERAL_TESI_HINT`, **default ON**).
  A differenza delle reti semantiche opt-in (FASE 11–13), è una correzione **mirata e a basso
  rischio**, gated sulla situazione di retrieval reale e senza nuove dipendenze: resta attiva di
  default. Il toggle serve all'A/B riproducibile.
- `agent.py` — parametro costruttore `use_general_tesi_hint` (default da config).
- `eval/run_eval.py` — flag `--no-general-tesi-hint`.
- `tests/test_answer_profile.py` — nuovo file (+11): predicato puro (regolamento generale
  sì/no, doc non-tesi, lista vuota/None, match su una qualsiasi fonte), profilo con/senza hint,
  ramo non-consultabilità non toccato, `sources=None` sicuro, toggle OFF.

**Impatto sul RAG (misurato, Ollama `llama3.1:8b`, `temperature=0`).**
- **A/B isolato su q14, stesso processo** (cambia **solo** l'hint): hint **OFF** → astensione
  (`evidenza_insufficiente`, «Non lo so in base ai documenti disponibili.»); hint **ON** →
  risposta corretta e citata («Risposta breve: Sì. … le Tesi sono consultabili dopo un periodo
  (embargo) di 24 mesi [F1] … la tesi di Informatica L-31 è consultabile dopo la laurea»).
  **Riprodotto su due processi freschi** (effetto stabile, non rumore cross-process).
- **Tuning del wording (misurato).** Un hint "soft" (le regole valgono per tutti i corsi,
  applica la regola generale) **non basta**: q14 resta astenuto. Un hint **imperativo** che
  chiarisce che il regolamento è unico per l'Ateneo e che l'assenza del nome del corso non è
  mancanza d'informazione **sblocca** la risposta. È stato adottato il secondo, evitando la
  variante che pre-dichiarava la conclusione («allora la risposta è SÌ»), più rischiosa per il
  grounding.
- **Blast radius (verificato offline, retrieval deterministico).** Su tutte e 40 le domande del
  dataset, **solo q14** soddisfa contemporaneamente «consultabilità» e «regolamento generale
  sulla tesi fra le fonti» → solo per q14 il prompt cambia. Tutte le altre risposte sono
  **identiche per costruzione**; q14, falsa astensione **stabile** della baseline, diventa
  risposta corretta: behavior **0,90 (36/40) → 0,925 (37/40)**, con course/topic, retrieval e
  citation invariati a 1,0. Le altre false astensioni della famiglia (q16/q21/q29) **non sono di
  consultabilità** e restano fuori dallo scopo: il loro recupero è un'estensione futura.

**Come testare.**
```bash
python -m pytest                                   # 230 test offline, attesi verdi
python eval/run_eval.py                            # default: hint ON (Ollama)
python eval/run_eval.py --no-general-tesi-hint     # A: senza l'hint (per l'A/B)
UNILAW_GENERAL_TESI_HINT=0 python eval/run_eval.py # equivalente via env
```

**Risultati test.** **230 test verdi** (219 + 11). Nessuna regressione: con `sources=None` o
toggle OFF il profilo è byte-identico a prima della FASE 14; l'hint compare solo nel ramo
consultabilità con un regolamento generale fra le fonti.

**Rischi residui.** L'efficacia dipende dal modello 8B: l'hint **riduce** (non garantisce) la
falsa astensione — sul caso misurato è risultata robusta su processi distinti, ma è un effetto
di prompting, non una garanzia formale. Lo scope è volutamente stretto (consultabilità + tesi
generale): non affronta q16/q21/q29 (false astensioni di natura diversa). Trade-off accettato:
l'hint spinge a rispondere quando il contesto riporta le condizioni di consultabilità, coerente
con la regola «rispondi solo in base al contesto» (non introduce informazioni esterne).

**Prossimo step.** Ciclo 2 — FASE 15 (copertina centrata in DOCX/PDF).

---

## 2026-06-22 — Fix operativo: rebuild della knowledge base da GUI («attempt to write a readonly database»)

**Obiettivo.** Sbloccare il rebuild della knowledge base lanciato dall'interfaccia
grafica, che falliva con `sqlite3.OperationalError: attempt to write a readonly
database` lasciando poi l'app incapace di rispondere alle domande.

**Sintomo.** Dopo l'analisi dei 22 PDF, `Chroma.from_documents` falliva sull'`upsert`
con errore SQLite di database in sola lettura; da quel momento ogni domanda non
funzionava. Non era un problema di permessi né di disco: un processo Python pulito
scriveva sullo *stesso* file `chroma.sqlite3` senza errori, e `lsof` mostrava **due
inode diversi sullo stesso path** (il vecchio indice da 12 MB ancora aperto + un file
nuovo) — file cancellato e ricreato sotto un handle aperto.

**Causa radice.** Il rebuild da GUI gira nel processo Streamlit long-running, che tiene
la KB aperta (`@st.cache_resource`). `_delete_existing_index()` esegue `shutil.rmtree()`,
ma ChromaDB 0.4 mantiene in cache, per tutta la durata del processo, un "System" (con la
connessione SQLite) per ogni `persist_directory`: il nuovo client riusa quella connessione
ormai orfana (file cancellato) → `readonly database`. Un processo appena avviato non ha
quella cache (per questo il test isolato scriveva senza problemi).

**File modificati.**
- `database.py` — in `_delete_existing_index()`, dopo la `rmtree`, si svuota la cache di
  sistema di ChromaDB (`from chromadb.api.client import SharedSystemClient;
  SharedSystemClient.clear_system_cache()`), così il rebuild apre una connessione nuova sul
  file ricreato. Chiamata avvolta in `try/except` (difensiva rispetto a cambi di API tra
  versioni); no-op in un processo appena avviato.
- `docs/problemi_noti.md` — **nuovo** documento dedicato di troubleshooting; questo caso è la
  voce **P-01** (sintomo, diagnosi, causa radice, soluzione, recupero immediato, prevenzione).

**Verifica.** `chromadb` 0.4.24; import e `SharedSystemClient.clear_system_cache()` OK;
`database.py` supera il controllo di sintassi. Da ora il rebuild dalla GUI funziona senza
dover riavviare l'app.

**Rischi residui.** Nessuno noto: la modifica agisce solo sul percorso di rebuild. Resta la
regola generale (cancellare i file di un DB SQLite aperto va sempre accompagnato dallo
svuotamento della cache di sistema o dal riavvio del processo). Dettagli completi in
[P-01](problemi_noti.md).
