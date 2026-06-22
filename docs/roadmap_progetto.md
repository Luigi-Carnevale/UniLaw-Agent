# Roadmap di progetto — UniLaw Agent

Documento unico che **argomenta l'intero percorso di lavoro su UniLaw Agent, dall'inizio
alla fine**: le fasi già svolte (con il problema affrontato, l'approccio scelto, l'esito
misurato e il modo in cui si è proseguito) e le fasi ancora da svolgere (già argomentate
con motivazione, intervento, verifica, rischio e dipendenze).

Serve da **ponte tra le sessioni di lavoro** e da riferimento condiviso (anche per il
lavoro in cowork) per mantenere allineata tutta la documentazione.

- **Creato:** 2026-06-17 (rinominato da `roadmap_ciclo2.md`, esteso a tutto il progetto).
- **Due cicli di lavoro.**
  - **Ciclo 1 — costruzione del RAG misurabile (FASE 0–9 + ESP-07): SVOLTO.**
  - **Ciclo 2 — consolidamento e qualità (FASE 1–16): DA SVOLGERE.**

> **Convenzione di numerazione.** I due cicli hanno numerazioni distinte. Le fasi del
> Ciclo 1 sono «FASE 0–9» (storiche, già nei documenti). Le fasi del Ciclo 2 sono
> «FASE 1–16» e vanno **sempre** citate come «**Ciclo 2 — FASE n**» (o «C2-FASE n»),
> mai come «FASE n» semplice, per non confonderle con quelle del Ciclo 1.

---

## 1. Come usare questo documento

- È la **fonte di verità del percorso**: la Parte I racconta cosa è stato fatto e
  perché; la Parte II argomenta cosa resta da fare.
- Le tabelle di stato sono **vive**: alla fine di ogni fase del Ciclo 2 si aggiorna lo
  stato (sezione 13) e si aggiunge una riga al *changelog del documento* (in fondo).
- Una fase si considera **completa** solo quando sono coerenti e aggiornati: **codice,
  test e documentazione** collegata (relazioni, `changelog_tecnico.md`,
  `valutazione_rag.md`, `esperimenti_rag.md`, `architettura_rag.md`, `readme.md` dove
  pertinente). È il contratto di documentazione del progetto.
- **Onestà dei risultati.** I numeri delle fasi svolte sono quelli **effettivamente
  misurati** (con i report citati). Per le fasi non svolte i numeri sono *attesi*, mai
  dichiarati.

## 2. Sintesi del percorso (colpo d'occhio)

| Ciclo | Fase | Titolo | Stato | Esito chiave |
|---|---|---|---|---|
| 1 | 0 | Audit tecnico | ✅ Svolto | Roadmap e criteri di qualità; ~85% di `agent.py` hard-coded |
| 1 | 1 | Test + baseline | ✅ Svolto | 62 test; baseline behavior 0,90 / abstention 0,857 |
| 1 | 2 | Refactoring leggero | ✅ Svolto | `agent.py` 2369→1976 righe; comportamento invariato |
| 1 | 3 | Retrieval ibrido | ✅ Svolto | BM25+RRF; recall ↑, nessuna regressione di retrieval |
| 1 | 4 | Reranker neurale opzionale | ✅ Svolto | Trade-off, non guadagno: **default OFF** |
| 1 | 5 | Evidence selection + citazioni | ✅ Svolto | behavior 0,90→**0,95**, abstention →**1,00** |
| 1 | 6 | Astensione affidabile | ✅ Svolto | causa classificata; reason-accuracy **1,0 (7/7)** |
| 1 | 7 | Riduzione hard-coding | ✅ Svolto | `knowledge.py` unica fonte; invariante per costruzione |
| 1 | 8 | UI e usabilità | ✅ Svolto | trace esportabile, PIPELINE, upload PDF; 115 test |
| 1 | 9 | Documentazione finale | ✅ Svolto | relazioni + deliverable DOCX/PDF |
| 1 | — | ESP-07 + decisione finale | ✅ Svolto | template di prosa **OFF**; default 0,95 / 1,0 |
| 2 | 1 | Fix bug calcolo migliaia | ✅ Svolto | «5% di 20.000€» → 1.000; 123 test verdi |
| 2 | 2 | Normalizza citazioni `(F#)→[F#]` | ✅ Svolto | `(F1)`/`F1`→`[F1]` (solo indici validi); 130 test verdi |
| 2 | 3 | Fonti oneste in astensione | ✅ Svolto | niente «Fonti utilizzate» in astensione; 132 test verdi |
| 2 | 4 | Ampliare dataset eval | ✅ Svolto | 20→40 domande; nuova baseline behavior 0,90; 141 test verdi |
| 2 | 5 | Scoring eval robusto | ✅ Svolto | esito dai segnali del trace; verdetti stabili al wording; 149 test |
| 2 | 6 | Held-out soglia astensione | ✅ Svolto | Soglia 0,37 riprodotta dai dati e validata su held-out: 3/3; 160 test verdi |
| 2 | 7 | Variabilità del modello | ✅ Svolto | `--repeat N` (media±σ); σ=0 within-session su 5 run; 173 test |
| 2 | 8 | Togliere fasce TOLC dal prompt | ✅ Svolto | Duplicazione rimossa; comportamento invariato entro la banda; 175 test |
| 2 | 9 | De-dup routing nel prompt | ✅ Svolto | **Risultato negativo**: rimozione regredisce behavior 0,90→0,875; routing **mantenuto**; 175 test verdi |
| 2 | 10 | Ripulire `_postprocess_answer` | ✅ Svolto | Rimosso il ramo che riscriveva (mascherava) l'astensione; behavior invariato 0,875; 181 test |
| 2 | 11 | Intent detection semantica | ✅ Svolto | Affianca le keyword (opt-in, default OFF); parafrasi 7/9; eval neutra per costruzione; 198 test |
| 2 | 12 | Grounding semantico | ✅ Svolto | Rete semantica opt-in (default OFF); sonda 7/7 a soglia calibrata 0,16; 208 test |
| 2 | 13 | `retrieval_strength` semantica | ✅ Svolto | Forza semantica opt-in (default OFF); soglia ricalibrata 0,53, held-out 3/3; 219 test |
| 2 | 14 | Mitigazione q14 | ✅ Svolto | hint "regola generale" (default ON, gated): q14 astensione→risposta; behavior 0,90→0,925; 230 test |
| 2 | 15 | Copertina centrata DOCX/PDF | ✅ Svolto | Copertine DOCX centrate (PDF già centrati); 6 deliverable rigenerati |
| 2 | 16 | Documentazione finale Ciclo 2 | ✅ Svolto | Narrativa Ciclo 2 (FASE 1–14) integrata nelle relazioni; deliverable consegnabili; 230 test, behavior 0,925 |

## 3. Stato attuale del sistema

UniLaw Agent è un RAG locale **misurabile, document-grounded e tracciabile**: retrieval
ibrido vettoriale + BM25 (fusione RRF), reranker neurale opzionale (default OFF),
evidence selection, verifica delle citazioni, astensione classificata per causa,
conoscenza normativa tracciabile (`knowledge.py`), UI con trace esportabile. La
configurazione predefinita disattiva i 5 template "di prosa" e mantiene solo il *guard
numerico TOLC-I* (ESP-07); dal Ciclo 2 — FASE 8 le soglie TOLC 9/16 non sono più ripetute
nel prompt (unica fonte di verità in `knowledge.py`); dal Ciclo 2 — FASE 10 il
post-processing non riscrive più le astensioni (la causa è classificata dal layer FASE 6 e
le fonti sono mostrate in modo onesto dalla FASE 3); dal Ciclo 2 — FASE 11 il riconoscimento
dell'intento può essere affiancato da un classificatore semantico **opzionale**
(`semantic_intent.py`, default OFF) che recupera corso/argomento sulle parafrasi; dal
Ciclo 2 — FASE 12 anche il grounding delle citazioni ha una **rete di recupero semantica
opzionale** (`grounding_report(..., embedder)`, default OFF, soglia calibrata 0,16) che
recupera le frasi parafrasate riducendo i falsi "supporto debole"; dal Ciclo 2 — FASE 13
anche la *retrieval strength* dell'astensione ha una **variante semantica opzionale**
(`semantic_retrieval_strength`, default OFF) con soglia **ricalibrata dai dati** (0,53) e
**validata 3/3 sull'held-out** della FASE 6; dal Ciclo 2 — FASE 14 il profilo di risposta,
sulle domande di **consultabilità della tesi** in cui è recuperato un **regolamento generale
di Ateneo**, riceve un hint (`has_general_tesi_regulation`, default **ON**, gated sulla
situazione di retrieval) che autorizza l'uso della regola generale e riduce la falsa
astensione (caso q14).
**230 test offline verdi.**
Comportamento predefinito sul dataset a 40 domande (ampliato nel Ciclo 2 — FASE 4): l'effetto
misurato della FASE 14 è il **recupero di q14** (da falsa astensione a risposta corretta «Sì…
consultabile dopo embargo di 24 mesi [F1]»), **isolato** a quella sola domanda — unica di
consultabilità che recupera un regolamento generale sulla tesi (verificato offline) — e
riprodotto end-to-end su processi distinti. Il dato robusto è un **+1 esatto su q14**: a parità
di tutto il resto, behavior **0,90 → 0,925** sulla baseline within-session di riferimento (σ=0,
Ciclo 2 — FASE 7); course/topic, retrieval, citation restano 1,0. Il *livello assoluto* di una
singola run oscilla di ±1 domanda tra sessioni (una run fresca senza l'hint ha dato 0,875 per
oscillazione di q28 e un retrieval miss): è la banda di rumore della FASE 7, ortogonale all'hint.
Il calo da 0,95 (20 domande del Ciclo 1) a 0,90/0,925 è dovuto al test più ampio e severo, non a
una regressione.

---

# PARTE I — Ciclo 1: costruzione del RAG misurabile (SVOLTO)

**Obiettivo del ciclo.** Trasformare una versione iniziale funzionante ma fortemente
basata su regole codificate in un RAG *production-like*, **valutabile**,
document-grounded, con retrieval ibrido, citazioni verificate, astensione affidabile,
test e documentazione universitaria. Il metodo è stato un ciclo ripetuto *misura →
intervento → ri-misura*, a piccoli passi verificabili, con la documentazione aggiornata
a ogni fase.

Tutte le metriche sotto sono **effettivamente eseguite** con `llama3.1:8b`
(`temperature=0`); i report grezzi sono in `eval/reports/`.

## FASE 0 — Audit tecnico (2026-06-16)
- **Perché.** Prima di modificare, capire cosa fosse realmente RAG e cosa fosse
  hard-coded, e dove stessero i rischi: senza questa mappa ogni intervento sarebbe stato
  cieco.
- **Come è stato affrontato.** Lettura sistematica di `app_agent.py`, `agent.py`,
  `database.py`, `config.py`, `tools.py`, README e corpus (22 PDF). Nessuna modifica al
  codice.
- **Esito (diagnosi).** Parti genuinamente RAG: ingest → chunking → embeddings →
  ChromaDB; retrieval vettoriale con MMR; deduplicazione; generazione grounded;
  estrazione citazioni `[F#]`; manifest SHA-256. Criticità: ~85% di `agent.py`
  hard-coded e tarato sui filename attuali (reranking euristico ~370 righe; 6 template
  deterministici ~850 righe che bypassano LLM e retrieval); conoscenza normativa
  duplicata nel prompt e nel codice; rischio di "allucinazione autorevole" dai template,
  fragilità verso nuovi PDF, citazioni potenzialmente fabbricate dal fallback
  `sources[:3]`.
- **Come siamo andati avanti.** La diagnosi ha prodotto la roadmap a fasi e i criteri di
  qualità; il primo passo necessario era una **rete di misura** (FASE 1), perché senza
  baseline nessun miglioramento sarebbe stato dimostrabile.

## FASE 1 — Test automatici e baseline misurata (2026-06-16)
- **Perché.** Mancava qualsiasi test e qualsiasi misura oggettiva: i refactoring
  avrebbero rischiato regressioni silenziose e i "miglioramenti" sarebbero stati
  opinioni.
- **Come è stato affrontato.** Creata una suite di **62 test offline** di
  *characterization* (fotografano il comportamento esistente, difetti inclusi); un
  dataset di **20 domande etichettate** (facili, difficili, ambigue, fuori dominio,
  senza risposta, sinonimi, mal formulate); un harness `eval/run_eval.py` con report
  JSON/Markdown; la documentazione tecnica iniziale.
- **Esito (misurato).** Baseline: **behavior 0,90**, course/topic **1,0**,
  retrieval-hit **1,0**, citation-hit **1,0**, **abstention 0,857**
  (`baseline_20260616_224551`). Individuato e *caratterizzato* (non corretto) il bug del
  separatore migliaia in `tools.py`. Due fallimenti chiave: **q14** (falsa astensione su
  evidenza buona) e **q18** (over-answering).
- **Come siamo andati avanti.** Con la baseline congelata, i due fallimenti sono
  diventati obiettivi misurabili per le fasi 5–6, e si è potuto procedere al refactoring
  (FASE 2) avendo una rete di sicurezza.

## FASE 2 — Refactoring leggero (2026-06-16)
- **Perché.** `agent.py` era un monolite di 2369 righe che concentrava intent,
  retrieval, reranking, regole, generazione, citazioni e confidenza: difficile estendere
  e testare in isolamento.
- **Come è stato affrontato.** Estrazione delle responsabilità **pure e già testate** in
  moduli dedicati (`rag_types`, `intent`, `rules_tolc`, `confidence`, `citations`); i
  metodi della classe sono diventati **deleganti sottili**, preservando la superficie
  pubblica (retrocompatibilità degli import). Lasciate in `agent.py` le parti da
  riscrivere poi (retrieval, reranking, template), per evitare churn rischioso.
- **Esito (misurato).** `agent.py` 2369→**1976** righe; **66 test** (62+4); eval
  `--limit 9` **identico alla baseline** → comportamento preservato.
- **Come siamo andati avanti.** Su moduli isolati è diventato sicuro intervenire sul
  retrieval (FASE 3).

## FASE 3 — Retrieval ibrido vettoriale + BM25 (2026-06-16)
- **Perché.** Il recupero era solo semantico: termini esatti, codici e sigle potevano
  sfuggire; mancava una misura della robustezza.
- **Come è stato affrontato.** Creato `retrieval.py` (BM25 via `rank_bm25` sui chunk già
  in ChromaDB + **Reciprocal Rank Fusion** + `hybrid_retrieve`) ed estratto
  `reranking.py` (`agent.py` sceso a **1438** righe). Toggle `use_bm25` per l'A/B;
  scoring di fusione nel trace. Aggiunta la sola dipendenza leggera `rank_bm25==0.2.2`.
- **Esito (misurato).** Ablation senza LLM: retrieval-hit 1,0 in entrambe le modalità,
  documento corretto al rango 1 (**0 peggiorati**); BM25 aggiunge recall in 4/13 casi.
  Eval completa: cambia **una sola** domanda, q17, dove il maggior recall porta il
  modello a rispondere invece di astenersi (behavior 0,90→**0,85**, abstention
  0,857→**0,714**): effetto di *generazione*, non di retrieval. `use_bm25=False`
  riproduce esattamente la baseline. Corretto un crash di BM25 su chunk a 0 token.
  (`baseline_20260616_235334`).
- **Come siamo andati avanti.** Il caso q17 ha reso evidente che servivano evidenze più
  mirate (FASE 5) e un'astensione più robusta (FASE 6). Prima, però, si è verificato se
  un reranker neurale potesse migliorare l'ordinamento (FASE 4).

## FASE 4 — Reranker neurale opzionale (2026-06-17)
- **Perché.** Verificare, con onestà, se un cross-encoder migliora la pertinenza
  rispetto all'euristica — senza imporre un modello pesante di default.
- **Come è stato affrontato.** `neural_reranker.py`: cross-encoder multilingua
  `mmarco-mMiniLMv2-L12-H384-v1` via `sentence-transformers` (nessuna nuova dipendenza),
  caricamento pigro, scorer iniettabile per i test, **fallback automatico**
  all'euristica, applicato ai primi 15 candidati; **opt-in**, default OFF.
- **Esito (misurato).** Costi: caricamento ~77 s a freddo, ~4 ms/coppia, RAM ~800 MB,
  disco ~458 MB. Ablation: il rango del documento corretto **peggiora in 6/13** casi
  (q01 da 1 a 5) perché il cross-encoder generico ignora i priori di dominio
  dell'euristica. Eval: risolve un caso borderline (q14) ma il behavior aggregato resta
  **0,90**, pari alla baseline euristica. (`baseline_20260617_092819`).
- **Come siamo andati avanti.** **Decisione: default OFF**, euristica come ordinamento
  primario; il reranker resta disponibile e configurabile per corpus futuri. È un
  risultato negativo, riportato comunque: l'onestà metodologica è parte del valore.

## FASE 5 — Evidence selection e verifica delle citazioni (2026-06-17)
- **Perché.** Il modello riceveva chunk interi (rumorosi): favoriva l'over-answering
  (q17, q18) e non impediva citazioni inventate.
- **Come è stato affrontato.** `evidence.py` seleziona i passaggi più pertinenti alla
  domanda (min 3 / max 6 frasi, ≤700 caratteri per fonte). `citations.py` rimuove i
  `[F#]` inventati e misura il supporto lessicale con politica **"reduce"** (abbassa la
  confidenza e aggiunge una nota, mai blocco cieco). Applicate solo al ramo LLM.
- **Esito (misurato).** **behavior 0,90→0,95**, **abstention 0,857→1,00**; **q17 e q18
  risolti**. L'A/B con `--no-evidence` (0,85/0,714) attribuisce il guadagno
  **interamente all'evidence selection**. Il reranker, anche con evidence, **non** aiuta
  (0,95/0,857): confermato OFF. (`baseline_20260617_095838`).
- **Come siamo andati avanti.** Resta q14 (la fonte corretta è recuperata e contiene la
  regola, ma il modello 8B si astiene): limite di *generazione*, caso di studio per la
  FASE 6, che ha reso l'astensione esplicita e classificata.

## FASE 6 — Astensione affidabile (2026-06-17)
- **Perché.** L'astensione esisteva ma non era classificata né misurabile: il sistema
  diceva "non lo so" senza distinguere il motivo.
- **Come è stato affrontato.** `abstention.py`: tassonomia delle cause
  (`fuori_dominio_corso`, `ambigua`, `retrieval_debole`, `fuori_dominio`,
  `evidenza_insufficiente`) e classificazione automatica. La distinzione "fuori dominio"
  vs "fonte presente ma insufficiente" usa la *retrieval strength* (sovrapposizione
  lessicale domanda↔fonti) con soglia **0,37** calibrata sui casi reali. Aggiunta la
  metrica `abstention_reason_accuracy` e il campo `expected_abstention_reason` sui 7
  negativi.
- **Esito (misurato).** behavior **0,95**, abstention **1,0**,
  **`abstention_reason_accuracy` 1,0 (7/7)**. La calibrazione ha corretto un'assunzione:
  il corpus cita il servizio mensa, quindi q18 è *insufficienza di evidenza*, non fuori
  dominio. (`baseline_20260617_104430`).
- **Come siamo andati avanti.** Caveat dichiarato: soglia e accuratezza calibrate sullo
  stesso piccolo set → da ri-validare su held-out (è un obiettivo esplicito del Ciclo 2,
  FASE 6). Con l'affidabilità coperta, si è affrontata la riduzione dell'hard-coding.

## FASE 7 — Riduzione dell'hard-coding (2026-06-17)
- **Perché.** La conoscenza normativa (soglie TOLC 9/16, tabella prova L-19) era
  **codificata e duplicata** in classificatore, template e prompt: difficile da
  verificare, a rischio di disallineamento.
- **Come è stato affrontato.** Creato `knowledge.py` come **unica fonte di verità**, con
  **provenienza** (citazione testuale verificata sui PDF): TOLC 9/16 dal regolamento di
  accesso, L-19 80/2h30/30-10-20-20 dal bando di immatricolazione. `rules_tolc` e le
  tabelle dei template leggono da qui. Classificate **tutte** le regole deterministiche
  (relazione §7: mantenere / trasformare in dato / mantenere con test / ridurre).
- **Esito (misurato).** **110 test** (102+8); tabelle generate **byte-identiche** a
  quelle codificate (test di parità) → **comportamento invariato per costruzione**. La
  riesecuzione dell'eval ha dato 0,90/0,857 per il **solo** q17: non una regressione, ma
  **non-determinismo del modello locale** a `temperature=0`. (`baseline_20260617_120429`).
- **Come siamo andati avanti.** Restano hard-coded, con motivazione e test: prosa dei
  template, regole nel prompt (`ANSWER_STYLE_GUIDE`), intent a keyword — tutti target del
  Ciclo 2. Scoperta importante (variabilità ±1 domanda) poi quantificata come obiettivo
  del Ciclo 2, FASE 7.

## FASE 8 — UI e usabilità reale (2026-06-17)
- **Perché.** L'osservabilità raccolta nel trace non era esportabile, lo stato della
  pipeline non era visibile e mancavano upload e messaggi d'errore chiari.
- **Come è stato affrontato.** `trace_export.py` (esportazione del trace in
  JSON/Markdown, funzioni pure). In `app_agent.py`: pulsanti di download del trace,
  blocco "PIPELINE" in sidebar, `st.file_uploader` per aggiungere PDF + ricostruzione
  dell'indice, messaggi d'errore con cause e rimedi.
- **Esito (misurato).** **115 test** (110+5); avvio headless verificato con
  `AppTest` senza eccezioni.
- **Come siamo andati avanti.** Con il sistema completo e osservabile, si è chiuso il
  ciclo con la documentazione finale.

## FASE 9 — Documentazione finale (2026-06-17)
- **Perché.** Una fase del progetto non è "finita" finché la documentazione non è
  coerente e consegnabile.
- **Come è stato affrontato.** Rifinita la relazione universitaria (Abstract e
  Conclusioni riscritti sull'esito complessivo, Appendice con comandi e istruzioni di
  conversione pandoc), valutazione allineata a 115 test, copertina completa (logo UniSA,
  autore Luigi Carnevale, corso Fondamenti di IA, anno 2025/2026, docente Fabio Palomba,
  data 24/06/2026). Generati i deliverable `relazione_unilaw_agent.{docx,pdf}`.
- **Come siamo andati avanti.** Subito dopo è emersa una verifica importante (ESP-07).

## ESP-07 e decisione finale — RAG puro come comportamento primario (2026-06-17)
- **Perché.** Verificare se il sistema **dipende** dalle risposte codificate o sa davvero
  leggere e comprendere i documenti.
- **Come è stato affrontato.** Flag `use_deterministic` / `PROSE_TEMPLATES_ENABLED` per
  disattivare i template; confronto A/B nella stessa sessione.
- **Esito (misurato).** RAG puro (nessuna regola): behavior 0,90, citation 0,923,
  retrieval 1,0. Con i template: behavior 0,90–0,95, citation 1,0. **Togliere le regole
  non fa calare l'accuratezza aggregata.** (`_153822` puro, `_153222` con prosa).
- **Decisione applicata.** I 5 template di prosa **disattivati di default**; resta solo
  il **guard numerico TOLC-I** (esattezza delle soglie + fonte canonica). Nuovo default
  misurato: behavior **0,95**, retrieval/citation/abstention **1,0**; su 13 domande con
  risposta solo 4 (TOLC) usano il guard, le altre 9 sono RAG puro
  (`baseline_20260617_155116`).

## Correzioni successive al Ciclo 1 (2026-06-17)
- **Indice ChromaDB fuori da iCloud.** L'app crashava (`attempt to write a readonly
  database`) perché iCloud bloccava il file SQLite in `~/Desktop`. `CHROMA_PERSIST_DIRECTORY`
  ora punta per default a `~/Library/Application Support/UniLawAgent/chroma_db` (override
  `UNILAW_CHROMA_DIR`); indice ricostruito e verificato.
- **Revisione di coerenza della documentazione.** Rimosse affermazioni stantie nella
  relazione a fasi (es. "reranker e evidence non ancora implementati" in §5; "astensione
  classificata non ancora coperta" in §12), allineato il README e l'architettura,
  aggiunta la configurazione predefinita finale (ESP-07) alla relazione; rigenerati i
  deliverable della relazione a fasi (19 pagine).

## Esito complessivo del Ciclo 1

Evoluzione delle metriche (modello `llama3.1:8b`, dataset di 20 domande):

| Configurazione | behavior | abstention | retrieval / citation | test |
|---|---|---|---|---|
| FASE 1 — baseline euristica | 0,90 | 0,857 | 1,0 / 1,0 | 62 |
| FASE 3 — retrieval ibrido | 0,85 | 0,714 | 1,0 / 1,0 | 75 |
| FASE 4 — + reranker neurale | 0,90 | 0,714 | 1,0 / 1,0 | 82 |
| FASE 5 — + evidence selection | 0,95 | 1,00 | 1,0 / 1,0 | 92 |
| FASE 6 — + astensione classificata | 0,95 | 1,00 | 1,0 / 1,0 (reason 1,0) | 102 |
| FASE 7 — riduzione hard-coding (invariante) | 0,95* | 1,00* | 1,0 / 1,0 | 110 |
| **Default finale (ESP-07)** | **0,95** | **1,00** | **1,0 / 1,0** (reason 1,0) | **115** |

\* refactor a comportamento invariato; una riesecuzione ha dato 0,90/0,857 per il solo
q17 a causa del non-determinismo del modello locale.

**Cosa rende il sistema utilizzabile.** Non la sola fluidità delle risposte, ma:
citazioni verificate, astensione affidabile e motivata, tracciabilità esportabile,
conoscenza normativa auditabile con provenienza. **Limiti residui** (che motivano il
Ciclo 2): dipendenza dal corpus, variabilità del modello locale, dataset di valutazione
contenuto, soglia di astensione non validata fuori campione, segnali ancora lessicali,
regole residue nel prompt.

---

# PARTE II — Ciclo 2: consolidamento e qualità (DA SVOLGERE)

**Obiettivo del ciclo.** Non riaprire le scelte architetturali consolidate, ma
intervenire su: difetti residui di **correttezza**, **pulizia degli output**, **solidità
della misurazione** e una riduzione ulteriore — ma *misurata* — dell'hard-coding.

## 4. Analisi: i problemi che motivano il Ciclo 2

Dall'analisi del codice e degli **output reali** (report `baseline_20260617_155116`):

- **Bug di calcolo (separatore migliaia).** `tools.py:_normalizza_numero_italiano` legge
  `"20.000"` come `20.0`: «5% di 20.000€» → «1» invece di «1.000». Già caratterizzato da
  un test che oggi *fotografa il bug*.
- **Fonti fuorvianti in astensione.** `citations.py:format_sources_block` ripiega su
  `sources[:3]` etichettati «Fonti utilizzate» quando non ci sono `[F#]` validi. Caso
  reale q19 («capitale della Francia»): si astiene ma elenca documenti casuali come
  "utilizzati" (rischio "citazioni fabbricate" già segnalato nell'audit FASE 0).
- **Deriva del formato citazione.** Il modello a volte scrive `(F1)` invece di `[F1]`
  (q14): le regex non lo riconoscono → grounding non verifica la frase e scatta il
  fallback misleading.
- **Duplicazione nel prompt.** `ANSWER_STYLE_GUIDE` (~45 righe) ripete sia le fasce
  numeriche TOLC (già in `knowledge.py` + guard) sia regole di instradamento per corso
  (già fatte da reranker + filtro metadata).
- **Caso speciale ampio.** `agent.py:_postprocess_answer` riscrive l'astensione per
  `doc_type ∈ {accesso, regolamento, altro}`: «altro» è troppo larga e può mascherare
  astensioni legittime.
- **Misurazione fragile/limitata.** Dataset di 20 domande; soglia di astensione
  calibrata sullo stesso set; scoring dell'eval dedotto da marcatori testuali;
  variabilità «±1 domanda» non quantificata.
- **Segnali lessicali.** Grounding e `retrieval_strength` basati su sovrapposizione di
  parole, non su similarità semantica: penalizzano le parafrasi.
- **Limite di generazione (q14).** Fonte corretta recuperata ma il modello 8B si astiene
  cercando un dettaglio "specifico per il corso".
- **Deliverable.** Copertina di DOCX/PDF allineata a sinistra (pandoc scarta i
  `<div align="center">`).

Esclusi **per scelta** (fuori dai vincoli): ciclo ReAct completo, sostituzione di
Ollama/ChromaDB, librerie pesanti non motivate.

## 5. Principio di ordinamento del Ciclo 2

1. **Prima le correzioni rapide** (Blocco A): difetti isolati, a basso rischio, che
   migliorano subito correttezza e onestà degli output e non richiedono il modello.
2. **Poi si rinforza la misurazione** (Blocco B) **prima** di toccare comportamenti: non
   ha senso valutare refactor rischiosi con un metro fragile o un dataset piccolo.
3. **Quindi i refactor di qualità** (Blocco C), ciascuno con confronto A/B su un metro
   affidabile.
4. **Infine** mitigazioni mirate, polish del deliverable e documentazione (Blocco D).

Si lavora **una fase alla volta**, con verifica (`python -m pytest`, ed eval dove
indicato) prima di passare alla successiva.

## 6. Blocco A — Correzioni di correttezza

### Ciclo 2 — FASE 1 — Fix bug calcolo separatore migliaia
- **Perché.** Una risposta numerica errata in un dominio normativo è un difetto di
  affidabilità; oggi un test *certifica* il bug invece di correggerlo.
- **Evidenza.** `tools.py:52`; «5% di 20.000€» → «1»; test
  `test_percentage_thousands_separator_known_bug`.
- **Intervento.** Euristica robusta: un `.` seguito da esattamente 3 cifre (senza virgola
  decimale) è separatore di migliaia; altrimenti decimale. Coprire `20.000`,
  `20.000,50`, `20000.50`, `1.234.567`, `20.5` (resta decimale). Convertire il test del
  bug in test di **correttezza**.
- **Impatto atteso.** Calcolo corretto; un test "vero" in più.
- **File.** `tools.py`, `tests/test_tools.py`.
- **Verifica.** `python -m pytest` (offline). **Rischio.** Basso. **Dipende da.** —

### Ciclo 2 — FASE 2 — Normalizzazione formato citazioni `(F#)→[F#]`
- **Perché.** Lo stile chiede `[F1]`, ma il modello a volte produce `(F1)`: una
  citazione tra parentesi diventa "invisibile" a verifica e blocco fonti, innescando il
  fallback fuorviante della FASE 3.
- **Evidenza.** Risposta a q14: «(F1) e (F2)…»; regex in `citations.py:26,48,74`.
- **Intervento.** Normalizzare in ingresso `(F1)`/`F1`→`[F1]` prima della verifica, senza
  alterare la prosa oltre il necessario.
- **Impatto atteso.** Citazioni riconosciute e verificate; q14 mostrerà «Fonti citate».
- **File.** `citations.py`, agganciata in `agent.py:answer`; `tests/test_citations.py`.
- **Verifica.** `pytest` + eval A/B (citation-hit invariato). **Rischio.** Basso.
  **Dipende da.** —

### Ciclo 2 — FASE 3 — Blocco fonti onesto in astensione
- **Perché.** In astensione elencare documenti come «Fonti utilizzate» è contraddittorio
  e mina la fiducia; su una domanda fuori dominio è palesemente errato.
- **Evidenza.** q19 si astiene ma elenca `guida-studente…`; `citations.py:114`; append
  incondizionato in `agent.py:338`.
- **Intervento.** Se `is_abstention`, sopprimere il fallback `sources[:3]` o rietichettarlo
  («Documenti consultati — nessuno utilizzato»); le citazioni reali restano mostrate.
- **Impatto atteso.** Output coerente su q17/q18/q19; nessuna falsa attribuzione.
- **File.** `agent.py:answer`, `citations.py:format_sources_block`;
  `tests/test_citations.py`.
- **Verifica.** `pytest` + eval (ispezione negativi). **Rischio.** Basso. **Dipende da.**
  FASE 2 (sinergica).

## 7. Blocco B — Fondamenta di misurazione

### Ciclo 2 — FASE 4 — Ampliare il dataset di valutazione
- **Perché.** Con 20 domande e un corpus "saturo" molte metriche sono al 100% e non
  discriminano; senza domande nuove (e *documenti-trappola*) non si può misurare la
  robustezza né giustificare i refactor del Blocco C.
- **Evidenza.** `eval/questions_baseline.jsonl` (20 domande); retrieval-hit 1,0
  attribuito all'adattamento al corpus.
- **Intervento.** Aggiungere domande con distrattori, più corsi/argomenti, parafrasi e
  nuovi negativi, mantenendo lo schema delle etichette.
- **Impatto atteso.** Metriche più informative (alcuni valori scenderanno: è atteso).
- **File.** `eval/questions_baseline.jsonl`; `docs/valutazione_rag.md`.
- **Verifica.** Nuovo report eval di riferimento. **Rischio.** Medio. **Dipende da.** —

### Ciclo 2 — FASE 5 — Scoring dell'eval più robusto
- **Perché.** Dedurre answer/abstain/clarify da marcatori testuali è fragile; il
  `RagTrace` contiene segnali strutturati più affidabili.
- **Evidenza.** Classificazione testuale in `eval/run_eval.py`; campi
  `abstention_reason`, `deterministic_rule`, citazioni nel trace.
- **Intervento.** Classificare dai segnali del trace, con i marcatori come fallback.
- **Impatto atteso.** Verdetti stabili al variare del wording.
- **File.** `eval/run_eval.py`; `docs/valutazione_rag.md`.
- **Verifica.** Confronto verdetti pre/post sui casi noti. **Rischio.** Medio.
  **Dipende da.** —

### Ciclo 2 — FASE 6 — Validazione held-out della soglia di astensione
- **Perché.** Una soglia calibrata e valutata sullo stesso set sovrastima la propria
  bontà: punto debole dichiarato del layer di affidabilità.
- **Evidenza.** `ABSTENTION_OOD_MAX_STRENGTH=0,37` (`config.py:670`); nota in
  `valutazione_rag.md` §11.2.
- **Intervento.** Separare calibrazione e validazione; ri-tarare su uno, riportare
  l'accuratezza sull'altro (negativi non visti).
- **Impatto atteso.** Soglia con bontà stimata fuori campione.
- **File.** `eval/questions_baseline.jsonl`, `eval/run_eval.py`; doc di valutazione.
- **Verifica.** `abstention_reason_accuracy` su held-out. **Rischio.** Medio.
  **Dipende da.** FASE 4.

### Ciclo 2 — FASE 7 — Quantificare la variabilità del modello locale
- **Perché.** Il «±1 domanda» è dichiarato senza numeri; per leggere gli A/B del Blocco C
  serve la banda di rumore.
- **Evidenza.** Oscillazione di q17 a parità di pipeline (ESP-06/ESP-07).
- **Intervento.** Opzione `--repeat N` che riporta media e deviazione delle metriche di
  generazione.
- **Impatto atteso.** Una banda di rumore con cui giudicare i guadagni reali.
- **File.** `eval/run_eval.py`; `docs/valutazione_rag.md`.
- **Verifica.** Report con media±σ su N run. **Rischio.** Basso. **Dipende da.** FASE 5.

## 8. Blocco C — Riduzione hard-coding e refactor di qualità

### Ciclo 2 — FASE 8 — Togliere dal prompt le fasce numeriche TOLC
- **Perché.** Le soglie 9/16 vivono già in `knowledge.py` (con provenienza) e sono
  applicate dal guard: ripeterle nel prompt è una seconda fonte di verità.
- **Evidenza.** `config.py:696–699`.
- **Intervento.** Rimuovere quelle righe dallo `ANSWER_STYLE_GUIDE`.
- **Impatto atteso.** Meno duplicazione, comportamento invariato.
- **File.** `config.py`; `architettura_rag.md`, relazioni.
- **Verifica.** Eval A/B: behavior non cala (entro la banda di FASE 7). **Rischio.**
  Medio. **Dipende da.** FASE 5, 7.

### Ciclo 2 — FASE 9 — De-duplicare le regole di routing nel prompt
- **Perché.** L'instradamento per corso è già nel reranker + filtro metadata: averlo
  anche nel prompt accoppia due meccanismi.
- **Evidenza.** `config.py:693–718`; `reranking.py`.
- **Intervento.** Ridurre `ANSWER_STYLE_GUIDE` alle sole regole generali di stile e
  citazione; il routing resta nel codice.
- **Impatto atteso.** Prompt più snello, comportamento stabile.
- **File.** `config.py`; `architettura_rag.md`, relazioni.
- **Verifica.** Eval A/B (course/topic/retrieval invariati). **Rischio.** Medio.
  **Dipende da.** FASE 8.
- **Esito (misurato, 2026-06-21) — RISULTATO NEGATIVO, routing mantenuto.** L'A/B nella
  stessa sessione (within-session σ=0) ha smentito l'impatto atteso: `course/topic/retrieval`
  restano invarianti a 1,0 (il routing dei documenti È ridondante col reranker), ma il
  *behavior* **regredisce** da 0,90 (con routing) a 0,875/0,85 (senza), perché la prosa fa
  anche da framing di commit per il modello 8B → più false astensioni su domande answerable
  borderline (q28/q30/q31). Come per il reranker (FASE 4), si riporta il negativo e si
  **mantiene il routing nel prompt** (`ANSWER_STYLE_GUIDE` byte-identico; guard di
  non-rimozione nei test). Dettagli: ESP-09 e changelog Ciclo 2 — FASE 9.

### Ciclo 2 — FASE 10 — Ripulire il caso speciale in `_postprocess_answer`
- **Perché.** Il ramo che riscrive l'astensione per `doc_type ∈ {accesso, regolamento,
  altro}` è troppo ampio e convive con il layer di astensione introdotto dopo: rischia di
  mascherare astensioni legittime.
- **Evidenza.** `agent.py:1452–1463`.
- **Intervento.** Restringere la condizione o rimuovere il ramo se ridondante; preservare
  i marker usati dai test.
- **Impatto atteso.** Astensioni più trasparenti e coerenti con la causa.
- **File.** `agent.py`; `tests/`.
- **Verifica.** Eval A/B + test deterministici. **Rischio.** Medio. **Dipende da.**
  FASE 5.

### Ciclo 2 — FASE 11 — Intent detection semantica (opt-in)
- **Perché.** Il riconoscimento corso/argomento è a liste di keyword, fragile verso
  formulazioni nuove.
- **Evidenza.** `intent.py`; limiti in `architettura_rag.md` §5.
- **Intervento.** Classificatore per similarità di embedding (riusa il modello caricato)
  che **affianca** le keyword; configurabile, default invariato finché non validato.
- **Impatto atteso.** Maggiore robustezza su parafrasi, a parità di casi noti.
- **File.** `intent.py`, `config.py`; `tests/test_intent_detection.py`.
- **Verifica.** Test su parafrasi + eval. **Rischio.** Medio-alto. **Dipende da.**
  FASE 4.

### Ciclo 2 — FASE 12 — Grounding semantico delle citazioni
- **Perché.** Il grounding lessicale boccia frasi corrette ma parafrasate, riducendo
  ingiustamente la confidenza.
- **Evidenza.** `citations.py:grounding_report` (overlap, soglia 0,18).
- **Intervento.** Supporto frase↔fonte anche per similarità di embedding, soglia
  ricalibrata; mantenere il lessicale come componente.
- **Impatto atteso.** Meno falsi "supporto debole".
- **File.** `citations.py`, `config.py`; `tests/test_citations.py`.
- **Verifica.** `pytest` + eval. **Rischio.** Alto. **Dipende da.** FASE 2.

### Ciclo 2 — FASE 13 — `retrieval_strength` semantica + ricalibrazione OOD
- **Perché.** La distinzione "fuori dominio" vs "fonte insufficiente" usa solo overlap
  lessicale; una misura semantica la rende più affidabile.
- **Evidenza.** `abstention.py:retrieval_strength`; soglia 0,37.
- **Intervento.** Calcolare la forza anche via similarità embedding query↔fonte e
  **ri-tarare** la soglia sull'held-out di FASE 6.
- **Impatto atteso.** Causa di astensione più precisa, validata fuori campione.
- **File.** `abstention.py`, `config.py`; `tests/test_abstention_reasons.py`.
- **Verifica.** `abstention_reason_accuracy` su held-out. **Rischio.** Alto.
  **Dipende da.** FASE 6, 12.

## 9. Blocco D — Mitigazioni mirate e deliverable

### Ciclo 2 — FASE 14 — Mitigazione q14 (falsa astensione su regolamento generale)
- **Perché.** Su «la tesi è consultabile dopo la laurea?» la fonte corretta (regolamento
  tesi *generale*) è recuperata ma il modello si astiene cercando un dettaglio specifico
  per il corso. È il caso di studio aperto del Ciclo 1.
- **Evidenza.** q14 (`regolamento-tesi-2023.pdf`).
- **Intervento.** Quando la domanda è "è consultabile/accessibile X?" e la fonte è un
  regolamento generale, dare un *hint* nel profilo di risposta (o un piccolo boost di
  evidenza) che autorizzi l'uso della regola generale.
- **Impatto atteso.** Riduzione (non garanzia) della falsa astensione.
- **File.** `agent.py:_build_answer_profile` / evidence.
- **Verifica.** q14 su più run (FASE 7); onesti: dipende dal modello 8B. **Rischio.**
  Medio. **Dipende da.** FASE 7.

### Ciclo 2 — FASE 15 — Copertina centrata in DOCX/PDF
- **Perché.** I deliverable sono parte della consegna; la copertina allineata a sinistra
  è un difetto di presentazione noto (pandoc scarta i `<div align="center">`).
- **Evidenza.** Caveat nelle Appendici delle relazioni.
- **Intervento.** Title-block pandoc (metadati `title/author/date`) e/o `reference-doc`
  DOCX + template LaTeX per centrare la copertina; rigenerare i quattro deliverable.
- **Impatto atteso.** Copertina centrata in DOCX e PDF.
- **File.** intestazioni delle relazioni / template; root (`relazione_*.{docx,pdf}`).
- **Verifica.** Ispezione visiva. **Rischio.** Basso. **Dipende da.** —

### Ciclo 2 — FASE 16 — Documentazione finale del Ciclo 2
- **Perché.** Il ciclo non è chiuso finché la documentazione non riflette, con onestà, ciò
  che è stato fatto e misurato.
- **Intervento.** Integrare i risultati delle FASE 1–15 (con tabelle e report) in:
  entrambe le relazioni, `changelog_tecnico.md`, `valutazione_rag.md`,
  `esperimenti_rag.md`, `architettura_rag.md`, `readme.md`; aggiornare questo documento
  (stato → completato); rigenerare i deliverable.
- **Impatto atteso.** Documentazione coerente e pronta da consegnare.
- **File.** tutti i `docs/` pertinenti, `readme.md`, deliverable in root.
- **Verifica.** Sweep di coerenza + `pytest` verde. **Rischio.** Basso. **Dipende da.**
  tutte le fasi precedenti.

## 10. Mappa delle dipendenze (Ciclo 2)

```
A (1, 2, 3)        indipendenti tra loro; 3 dopo 2 (sinergica)
                   1 e 2 eseguibili in qualsiasi momento

B  4 ──► 6
   4 ──► 11
   5 ──► 6, 7, 8, 9, 10
   7 ──► 8, 9, 14

C  8 ──► 9
   2 ──► 12 ──► 13
   6 ──► 13

D  15  indipendente
   16  dopo tutte
```

Fasi avviabili da subito senza prerequisiti: **1, 2, 15** (e, a seguire, **3**).

---

## 11. Vincoli e non-obiettivi (validi per tutto il progetto)

- **Vincoli.** Stack local-first (Ollama + ChromaDB + embedding locali) non sostituibile
  senza approvazione; niente librerie pesanti non motivate; non eliminare `documenti/` o
  l'indice senza chiedere; modifiche a **piccoli step verificabili** (`python -m pytest`);
  documentazione sempre aggiornata.
- **Non-obiettivi.** Ciclo ReAct completo; cambio di modello o di vector store;
  multiutente; reranking neurale acceso di default.

## 12. Allineamento con la documentazione esistente

Per ogni fase del Ciclo 2, al completamento, aggiornare:

- **`docs/changelog_tecnico.md`** — voce datata (Ciclo 2 — FASE n): file, modifiche,
  impatto misurato, come testare, rischi, prossimo step.
- **`docs/valutazione_rag.md`** — quando cambiano dataset, metriche o soglie (FASE 4–13).
- **`docs/esperimenti_rag.md`** — per ogni A/B con numeri (FASE 8–13).
- **`docs/architettura_rag.md`** — quando cambiano moduli o flusso (FASE 9–13).
- **Relazioni** — in FASE 16, con sintesi e tabelle.
- **`readme.md`** — solo se cambiano installazione, avvio, dipendenze, struttura o uso.
- **Questo documento** — tabella di stato (sezione 13) e changelog (sezione 14).

## 13. Stato di avanzamento del Ciclo 2

| Fase | Titolo | Blocco | Rischio | Stato |
|---|---|---|---|---|
| 1 | Fix bug calcolo separatore migliaia | A | Basso | ✅ Completata (2026-06-17) |
| 2 | Normalizzazione formato citazioni `(F#)→[F#]` | A | Basso | ✅ Completata (2026-06-18) |
| 3 | Blocco fonti onesto in astensione | A | Basso | ✅ Completata (2026-06-18) |
| 4 | Ampliare il dataset di valutazione | B | Medio | ✅ Completata (2026-06-18) |
| 5 | Scoring dell'eval più robusto | B | Medio | ✅ Completata (2026-06-18) |
| 6 | Validazione held-out della soglia di astensione | B | Medio | ✅ Completata (2026-06-19) |
| 7 | Quantificare la variabilità del modello | B | Basso | ✅ Completata (2026-06-20) |
| 8 | Togliere dal prompt le fasce numeriche TOLC | C | Medio | ✅ Completata (2026-06-21) |
| 9 | De-duplicare le regole di routing nel prompt | C | Medio | ✅ Completata (2026-06-21) — *risultato negativo*: routing mantenuto |
| 10 | Ripulire il caso speciale in `_postprocess_answer` | C | Medio | ✅ Completata (2026-06-22) |
| 11 | Intent detection semantica (opt-in) | C | Medio-alto | ✅ Completata (2026-06-22) |
| 12 | Grounding semantico delle citazioni | C | Alto | ✅ Completata (2026-06-22) — opt-in, default OFF; soglia calibrata 0,16 |
| 13 | `retrieval_strength` semantica + ricalibrazione OOD | C | Alto | ✅ Completata (2026-06-22) — opt-in, default OFF; soglia ricalibrata 0,53, held-out 3/3 |
| 14 | Mitigazione q14 (falsa astensione) | D | Medio | ✅ Completata (2026-06-22) — hint "regola generale" default ON, gated; q14 risolto, behavior 0,90→0,925 |
| 15 | Copertina centrata in DOCX/PDF | D | Basso | ✅ Completata (2026-06-22) |
| 16 | Documentazione finale del Ciclo 2 | D | Basso | ✅ Completata (2026-06-22) |

## 14. Changelog di questo documento

- **2026-06-22** — **Ciclo 2 — FASE 15 e 16 completate + allineamento finale (FASE 13–14).
  CICLO 2 CONCLUSO.** Eseguite le ultime due fasi documentali e allineata tutta la
  documentazione allo stato attuale del codice (230 test, behavior predefinito **0,925**).
  Allineamento FASE 13–14 (i `.md` vivi `changelog_tecnico.md`, `esperimenti_rag.md`,
  `architettura_rag.md` erano già aggiornati da chi ha sviluppato il codice; il conteggio di
  `valutazione_rag.md` e le relazioni erano rimasti indietro): `valutazione_rag.md` §1.1
  **208 → 230** (+11 FASE 13, +11 FASE 14) e nuova nota in §14.4 (q14 risolto, default
  0,925, tabella FASE 4 mantenuta come storica); relazioni con conteggio **→ 230**, behavior
  predefinito **0,925**, e i paragrafi di FASE 13 (`retrieval_strength` semantica) e FASE 14
  (mitigazione q14). **FASE 15 — copertina centrata DOCX:** i DOCX delle relazioni sono ora
  post-processati per centrare i paragrafi di copertina (i PDF erano già centrati via
  WeasyPrint), eliminando il caveat noto. **FASE 16 — documentazione finale:** integrata la
  narrativa completa del Ciclo 2 (FASE 1–14) nelle due relazioni, aggiornata la riga di stato
  delle relazioni a «Ciclo 2 concluso», la presentazione a «FASE 1–16 completate» e questo
  documento (sez. 2 e 13: FASE 15 e 16 ✅). Rigenerati i **sei deliverable** DOCX/PDF.
  `readme.md` non modificato (installazione/struttura invariate). **Nota onesta:** restano,
  come limite di generazione del modello 8B, le false astensioni q16/q21/q29 (fuori dallo
  scopo della FASE 14) e le quattro estensioni semantiche (reranker, intent, grounding,
  `retrieval_strength`) sono **opt-in/default OFF** in attesa di validazione su un set più
  ampio: il Ciclo 2 è concluso ma questi sono i naturali punti di partenza di un eventuale
  Ciclo 3.
- **2026-06-22** — **Ciclo 2 — FASE 14 completata** (mitigazione q14: falsa astensione su
  regolamento generale). Aggiornate le tabelle di stato (sez. 2 e 13), la sez. 3 (conteggio
  test 219 → **230** e comportamento predefinito 0,90 → **0,925**). Su una domanda di
  consultabilità della tesi che nomina un corso preciso (q14, «La tesi di Informatica L-31 è
  consultabile dopo la laurea?») la regola vive in un **regolamento generale di Ateneo**
  (`regolamento-tesi-2023.pdf`, `course_tag="generale"`), non in un documento specifico del
  corso; il contesto recuperato **contiene già la regola completa** (consultabile / dopo
  embargo di 24 mesi / non consultabile), quindi la falsa astensione è un limite di
  **generazione**, non di retrieval. Codice: nuovo predicato puro
  `agent.has_general_tesi_regulation(sources)` (riconosce un regolamento generale sulla tesi
  fra le fonti); `_build_answer_profile(intent, question, sources)` — nel ramo consultabilità,
  se il predicato è vero e il toggle è attivo, aggiunge al profilo un hint che **autorizza
  l'uso della regola generale** e scoraggia l'astensione quando il contesto riporta le
  condizioni di consultabilità (senza dettare la risposta: il *cosa* resta lettura del
  modello). `config.GENERAL_TESI_HINT_ENABLED` (env `UNILAW_GENERAL_TESI_HINT`, **default
  ON** — è una correzione mirata e a basso rischio, gated sulla situazione di retrieval, non
  una rete semantica sperimentale); `agent.py`: parametro `use_general_tesi_hint`;
  `eval/run_eval.py`: flag `--no-general-tesi-hint` (per l'A/B). Test: nuovo
  `tests/test_answer_profile.py` (+11) → **230 test verdi** (219 + 11). **Misura (Ollama,
  `llama3.1:8b`, `temperature=0`).** A/B **isolato su q14** nello stesso processo (cambia solo
  l'hint): con hint **OFF** q14 si astiene (`evidenza_insufficiente`), con hint **ON** risponde
  correttamente («Risposta breve: Sì… consultabile dopo un periodo di embargo di 24 mesi [F1]»),
  **riprodotto su due processi freschi**. La calibrazione del wording è misurata: l'hint "soft"
  iniziale non bastava (il modello restava astenuto), un hint imperativo che chiarisce che il
  regolamento generale vale per tutti i corsi sblocca la risposta. **Blast radius verificato
  offline**: su tutte e 40 le domande **solo q14** attiva l'hint (è l'unica di consultabilità
  che recupera un regolamento generale sulla tesi) → tutto il resto è **identico per
  costruzione** e q14, falsa astensione stabile della baseline, diventa risposta corretta:
  behavior **0,90 (36/40) → 0,925 (37/40)**. Le altre false astensioni della famiglia
  (q16/q21/q29) **non** sono di consultabilità: restano fuori dallo scopo di questa fase.
  Doc: `changelog_tecnico.md` (voce dedicata), `esperimenti_rag.md` (ESP-14 con tabelle di
  tuning e A/B), `architettura_rag.md` (§3 profilo di risposta / flusso, §5 limite q14
  mitigato). `valutazione_rag.md` non toccata (nessun cambio di dataset/soglie; il bump del
  conteggio test e l'integrazione narrativa nelle relazioni restano la FASE 16). Relazioni e
  deliverable non rigenerati: per contratto (§12) l'integrazione avviene in FASE 16. Prossimo
  step: Ciclo 2 — FASE 15 (copertina centrata in DOCX/PDF).
- **2026-06-22** — **Ciclo 2 — FASE 13 completata** (`retrieval_strength` semantica +
  ricalibrazione OOD, opt-in). Aggiornate le tabelle di stato (sez. 2 e 13) e il conteggio
  test in sez. 3 (208 → **219**). La causa di astensione `fuori_dominio` vs
  `evidenza_insufficiente` (FASE 6) si decideva sulla *retrieval strength* **lessicale**
  (overlap di token, soglia 0,37): penalizza le query **parafrasate**, in dominio ma prive di
  token in comune con la fonte. Codice: nuova funzione pura
  `abstention.semantic_retrieval_strength` (massima similarità di embedding query↔fonte, [0,1],
  fallback sicuro); `classify_llm_abstention` accetta `embedder` + `semantic_ood_max_strength`
  opzionali (con `embedder=None`, default, comportamento **byte-identico** al lessicale; riusa
  l'embedder del vector store, nessuna nuova dipendenza). `config.py`:
  `ABSTENTION_SEMANTIC_STRENGTH_ENABLED` (env `UNILAW_SEMANTIC_ABSTENTION`, **default OFF**) e
  `ABSTENTION_OOD_SEMANTIC_MAX_STRENGTH=0,53`; `agent.py`: parametro `use_semantic_abstention`;
  `eval/run_eval.py`: flag `--semantic-abstention`; l'harness
  `eval/abstention_threshold_validation.py` misura, calibra e valida ora **anche** la sezione
  semantica. Test: +11 (208 → **219 test verdi**). **Misura (no Ollama, embedder reale):** sui
  negativi threshold-relevant la forza semantica calibra a massimo margine a **0,5286** (≈ la
  0,53 di config, riproducibile dai dati come la 0,37 lessicale rispetto alla 0,3667) e valida
  **3/3 sull'held-out** q35/q36/q37 (mai visti) sia a 0,53 sia a 0,5286; le due forze
  **concordano** su tutti e 6 i negativi. Guadagno atteso dimostrato unitariamente (embedder
  finto: una parafrasi senza token in comune è `evidenza_insufficiente` per il semantico ma
  `fuori_dominio` per il solo lessicale) e smoke test live (q19 → `fuori_dominio` come la
  lessicale, integrazione confermata). **Esito onesto:** la separazione semantica regge ma è
  più **compressa** (forza dei fuori-dominio alta in assoluto, 0,32–0,46) e gli insiemi sono
  piccoli → come reranker (FASE 4), intent semantico (FASE 11) e grounding semantico (FASE 12),
  **default OFF** in attesa di un set più ampio. **Neutra sull'eval** per costruzione (OFF:
  l'embedder non viene costruito; ON: cambierebbe solo l'etichetta di causa sui threshold-relevant,
  su cui i due segnali concordano → `abstention_reason_accuracy` invariata). Doc:
  `changelog_tecnico.md` (voce dedicata), `esperimenti_rag.md` (ESP-13 con tabelle),
  `architettura_rag.md` (§2 `abstention.py`, §3 flusso, §5 limite mitigato, §6 evoluzione),
  `valutazione_rag.md` (nuova §17 + aggiornato il riferimento avanti in §15.3). Relazioni e
  deliverable non rigenerati: per contratto (§12) l'integrazione narrativa avviene in FASE 16.
  Prossimo step: Ciclo 2 — FASE 14 (mitigazione q14, falsa astensione su regolamento generale).
- **2026-06-22** — **Allineamento di coerenza dei deliverable alla FASE 12 (su
  richiesta).** Dopo la FASE 12 (208 test), che per contratto (§12) aveva aggiornato i `.md`
  vivi `changelog_tecnico.md`, `esperimenti_rag.md` (ESP-12) e `architettura_rag.md` (§2, §3,
  §5, §6), restavano disallineati il conteggio test di `valutazione_rag.md` (lasciato a 198
  per questo passo), le due relazioni, la presentazione e i deliverable (fermi a FASE 11).
  Questo passo li riallinea **senza** anticipare l'integrazione narrativa completa (FASE 16).
  Interventi: (1) `valutazione_rag.md` §1.1 — conteggio **198 → 208** con il «+10 grounding
  semantico delle citazioni opt-in (Ciclo 2 — FASE 12)»; (2) `relazione_unilaw_agent.md` e
  `relazione_completa_unilaw_agent.md` — conteggio test **198 → 208** (abstract, §10/§13.1,
  Appendice e nota di consolidamento) e aggiunta della FASE 12 (controllo di grounding per
  similarità di embedding, additivo al lessicale, soglia 0,16 calibrata, sonda 7/7,
  **opt-in/default OFF**, neutro sulle metriche perché agisce solo sulla confidenza), con
  nuova riga nella tabella test della relazione completa; (3) `UniLaw_Agent_presentazione.pptx`
  — 198 → **208 test** e roadmap a «FASE 1–12 completate»; (4) **rigenerati i sei
  deliverable** DOCX/PDF (DOCX via pandoc; PDF via WeasyPrint con copertina centrata, indice
  con numeri di pagina e intestazione corrente). Metriche headline invariate (0,90). `readme.md`
  non toccato. Resta un **allineamento di coerenza**, non l'integrazione narrativa del Ciclo 2
  (FASE 16); copertina DOCX a sinistra (centratura = FASE 15).
- **2026-06-22** — **Ciclo 2 — FASE 12 completata** (grounding semantico delle citazioni,
  opt-in). Aggiornate le tabelle di stato (sez. 2 e 13) e il conteggio test in sez. 3
  (198 → **208**). Il grounding delle citazioni (FASE 5) misurava il supporto delle frasi
  citanti per **sola sovrapposizione lessicale** di token (soglia 0,18): le frasi corrette ma
  **parafrasate** non condividono token e venivano bocciate, abbassando ingiustamente la
  confidenza. Codice: `citations.py:grounding_report` accetta ora un `embedder` opzionale e una
  soglia `min_semantic`; le frasi bocciate dal lessicale ricevono un **secondo controllo per
  similarità di embedding** (frase citante ↔ frasi della fonte, helper `_semantic_support`) e
  sono recuperate se la più vicina supera la soglia. La rete semantica **si aggiunge** al
  lessicale (non lo sostituisce e non può togliere un supporto già riconosciuto); con
  `embedder=None` (default) il risultato è **byte-identico** al solo lessicale; fallback sicuro
  se l'embedder è assente o solleva eccezioni (nessuna nuova dipendenza: riusa l'embedder del
  vector store via `_embedder_from_vector_db`, come in FASE 11). `config.py`:
  `CITATION_GROUNDING_SEMANTIC_ENABLED` (env `UNILAW_SEMANTIC_GROUNDING`, **default OFF**) e
  `CITATION_GROUNDING_SEMANTIC_MIN_SIMILARITY`; `agent.py`: parametro `use_semantic_grounding`;
  `eval/run_eval.py`: flag `--semantic-grounding`. Test: +10 in `tests/test_citations.py` →
  **208 test verdi** (198 + 10). **Calibrazione della soglia (dai dati, no Ollama, modello di
  embedding reale):** su una sonda di coppie (frase citante, fonte) le **parafrasi** danno coseno
  0,194–0,605 e le **estranee** 0,017–0,129; il punto di massimo margine è la mediana
  (0,129+0,194)/2 ≈ **0,16** (soglia adottata). A 0,16 la sonda separa **7/7** (4 parafrasi
  recuperate — che il solo lessicale boccia tutte — e 3 estranee respinte). **Esito onesto:** la
  separazione è **sottile** e la similarità di frase del MiniLM sulle parafrasi normative è
  **debole** (3 su 4 sotto 0,30) → come per il reranker (FASE 4) e l'intent semantico (FASE 11),
  la funzione è implementata e disponibile ma **default OFF** finché non validata su un insieme
  più ampio. **Neutralità sull'eval per costruzione:** il grounding agisce solo sulla confidenza
  (politica "reduce"), che non entra nelle metriche-vetrina → behavior/citation/abstention
  invariate. Doc: `changelog_tecnico.md` (voce dedicata con tabella di calibrazione),
  `esperimenti_rag.md` (ESP-12 con tabelle), `architettura_rag.md` (§2 modulo `citations.py`, §3
  flusso, §5 limite citazioni mitigato, §6 evoluzione). `valutazione_rag.md` non toccata (nessun
  cambio di dataset, metriche-vetrina o soglie di astensione; il bump 198 → 208 è allineamento di
  coerenza, FASE 16). Relazioni e deliverable non rigenerati: per contratto (§12) l'integrazione
  narrativa avviene in FASE 16. Prossimo step: Ciclo 2 — FASE 13 (`retrieval_strength` semantica
  + ricalibrazione OOD).
- **2026-06-22** — **Allineamento di coerenza dei deliverable alla FASE 11 (su
  richiesta).** Dopo la FASE 11 (198 test), che per contratto (§12) aveva aggiornato i `.md`
  vivi `changelog_tecnico.md`, `esperimenti_rag.md` (ESP-11) e `architettura_rag.md` (§2,
  §5, §6), restavano disallineati il conteggio test di `valutazione_rag.md` (lasciato a 181
  apposta per questo passo), le due relazioni, la presentazione e i deliverable (fermi a
  FASE 10). Questo passo li riallinea **senza** anticipare l'integrazione narrativa completa
  (FASE 16). Interventi: (1) `valutazione_rag.md` §1.1 — conteggio **181 → 198** con il «+17
  intent detection semantica opt-in (Ciclo 2 — FASE 11)»; (2) `relazione_unilaw_agent.md` e
  `relazione_completa_unilaw_agent.md` — conteggio test **181 → 198** (abstract, §10/§13.1,
  Appendice e nota di consolidamento) e aggiunta della FASE 11 (modulo `semantic_intent.py`
  che affianca le keyword in gap-fill senza override, **opt-in/default OFF**, neutro per
  costruzione sull'eval, parafrasi 7/9), con nuova riga nella tabella test della relazione
  completa; (3) `UniLaw_Agent_presentazione.pptx` — 181 → **198 test** e roadmap a «FASE
  1–11 completate»; (4) **rigenerati i sei deliverable** DOCX/PDF (DOCX via pandoc; PDF via
  WeasyPrint con copertina centrata, indice con numeri di pagina e intestazione corrente).
  Metriche headline invariate (0,90, baseline rappresentativa): il semantico è OFF e non
  cambia i verdetti. `readme.md` non toccato. Resta un **allineamento di coerenza**, non
  l'integrazione narrativa del Ciclo 2 (FASE 16); copertina DOCX a sinistra (centratura =
  FASE 15).
- **2026-06-22** — **Ciclo 2 — FASE 11 completata** (intent detection semantica opt-in).
  Aggiornate le tabelle di stato (sez. 2 e 13) e il conteggio test in sez. 3 (181 → **198**).
  Nuovo modulo `semantic_intent.py`: `SemanticIntentClassifier` per **similarità di embedding**
  che **affianca** il riconoscimento a keyword (`intent.py`) — riempie SOLO le caselle
  (corso/argomento) lasciate vuote dalle keyword, **prima** della memoria a slot e **senza mai
  sovrascrivere** un riconoscimento a keyword; non tocca i corsi fuori dominio. Riusa l'embedder
  già caricato nel vector store (nessuna nuova dipendenza, nessun modello aggiuntivo); embedder
  iniettabile e fallback sicuro. **Opt-in, default OFF** (`config.SEMANTIC_INTENT_ENABLED`,
  env `UNILAW_SEMANTIC_INTENT`, param `use_semantic_intent`, flag eval `--semantic-intent`);
  soglie del coseno provvisorie 0,5 (corso) / 0,45 (argomento). `infer_query_intent` ha un terzo
  parametro `semantic_classifier=None` (con `None` il comportamento è byte-identico). Test:
  +17 in `tests/test_intent_detection.py` (helper puri, meccanica con embedder finto, gap-fill,
  no-override, default OFF, cablaggio) → **198 test verdi** (181 + 17). **Verifica.** (1)
  *Neutralità sull'eval, per costruzione*: sul dataset a 40 domande le keyword coprono già
  corso/argomento su **tutte e 40** (0 caselle vuote) e l'override è **0** → con il semantico
  attivo l'intent è byte-identico a quello a keyword; essendo identici domanda e intent, e il
  modello deterministico a `temperature=0` (σ=0 within-session, FASE 7), la generazione
  end-to-end è identica per costruzione (l'A/B generativo non cambierebbe alcun verdetto, come i
  casi non esercitati dal dataset, FASE 10). (2) *Parafrasi (modello reale)*: su 9 sonde prive
  dei token-keyword, tutte mancate dalle keyword, il semantico ne **recupera 7/9** (i 4 corsi +
  borsa/erasmus/piano_studi), con 1 sotto soglia (tesi) e 1 falso positivo (accesso↔piano_studi).
  Decisione: classificatore implementato e disponibile, **default OFF** finché le soglie non sono
  validate su un set più ampio (in sinergia con la FASE 4). Doc: `changelog_tecnico.md` (voce
  dedicata), `esperimenti_rag.md` (ESP-11), `architettura_rag.md` (§2 nuovo modulo, §5 limite
  dell'intent a keyword mitigato, §6 evoluzione). `valutazione_rag.md` non toccata (nessun cambio
  di dataset, metriche-vetrina o soglie di astensione; il bump 181 → 198 è allineamento di
  coerenza, FASE 16). Relazioni e deliverable non rigenerati: per contratto (§12) l'integrazione
  narrativa avviene in FASE 16. Prossimo step: Ciclo 2 — FASE 12 (grounding semantico delle
  citazioni).
- **2026-06-22** — **Allineamento di coerenza dei deliverable alla FASE 10 (su
  richiesta).** Dopo la FASE 10 (181 test), che per contratto (§12) aveva aggiornato i `.md`
  vivi `changelog_tecnico.md`, `esperimenti_rag.md` (ESP-10) e `architettura_rag.md`,
  restavano disallineati il conteggio test di `valutazione_rag.md` (fermo a 175), le due
  relazioni, la presentazione e i deliverable (fermi a FASE 9). Questo passo li riallinea
  **senza** anticipare l'integrazione narrativa completa (FASE 16). Interventi: (1)
  `valutazione_rag.md` §1.1 — conteggio **175 → 181** con il «+6 pulizia del post-processing
  dell'astensione (Ciclo 2 — FASE 10)»; (2) `relazione_unilaw_agent.md` e
  `relazione_completa_unilaw_agent.md` — conteggio test **175 → 181** (abstract, §10/§13.1,
  Appendice e nota di consolidamento) e aggiunta della FASE 10 (rimosso il ramo di
  `_postprocess_answer` che riscriveva e mascherava l'astensione, comportamento invariato,
  ramo dormiente sul dataset), con nuova riga nella tabella test della relazione completa;
  (3) `UniLaw_Agent_presentazione.pptx` — 175 → **181 test** e roadmap a «FASE 1–10
  completate»; (4) **rigenerati i sei deliverable** DOCX/PDF (DOCX via pandoc; PDF via
  WeasyPrint con copertina centrata, indice con numeri di pagina e intestazione corrente).
  Le **metriche headline restano 0,90** (la baseline rappresentativa within-session): lo
  0,875 dell'A/B FASE 10 è oscillazione cross-process entro la banda di rumore della FASE 7,
  non un nuovo valore di riferimento. `readme.md` non toccato. Resta un **allineamento di
  coerenza**, non l'integrazione narrativa del Ciclo 2 (FASE 16); copertina DOCX a sinistra
  (centratura = FASE 15).
- **2026-06-22** — **Ciclo 2 — FASE 10 completata** (ripulire il caso speciale in
  `_postprocess_answer`). Aggiornate le tabelle di stato (sez. 2 e 13) e il conteggio test in
  sez. 3 (175 → **181**). Rimosso il ramo che, su argomento "accesso" e fonti `doc_type ∈
  {accesso, regolamento, altro}`, **riscriveva** l'astensione con un testo generico: la
  condizione era troppo ampia («altro») e — riscrivendo la risposta con un testo privo di
  marcatori — rendeva l'astensione invisibile a `is_abstention`, scavalcando la
  classificazione della causa (FASE 6) e il blocco fonti onesto (FASE 3). Codice (`agent.py`):
  `_postprocess_answer` ora rileva solo l'incertezza (abbassa la confidenza) e restituisce il
  testo del modello senza riscriverlo; rilevamento unificato su `abstention.is_abstention`
  (eliminata la lista locale duplicata); firma semplificata (rimossi `sources/intent/question`,
  divenuti codice morto). Nuovo `tests/test_postprocess.py` (6 test offline) → **181 test
  verdi** (175 + 6). **A/B misurato** (`llama3.1:8b`, `temperature=0`, 40 domande, stessa
  sessione, due processi): A con ramo (`baseline_20260622_094539`) e B senza ramo
  (`baseline_20260622_095518`) danno **entrambi behavior 0,875**, course/topic/retrieval/
  citation 1,0; il testo di riscrittura compare **0 volte** in A ⇒ il ramo era **dormiente in
  entrambe** le run. Le sole differenze (q18, q28) sono **oscillazione cross-process** del
  modello (testi generati divergenti), non effetto della modifica; fallimenti stabili in
  entrambe: q14/q16/q21/q29 (famiglia nota, target FASE 14). **Nessuna regressione**; la
  modifica è neutra sull'eval (caso speciale non esercitato dal dataset) e il valore —
  astensioni non più mascherate — è dimostrato per costruzione e dai 6 test unitari. Doc:
  `changelog_tecnico.md` (voce dedicata con tabelle A/B e per-domanda), `esperimenti_rag.md`
  (ESP-10), `architettura_rag.md` (nota sul post-processing nel flusso). Relazioni e
  deliverable non rigenerati: per contratto (§12) l'integrazione narrativa avviene in FASE 16.
  Prossimo step: Ciclo 2 — FASE 11 (intent detection semantica opt-in).
- **2026-06-21** — **Allineamento di coerenza dei deliverable alla FASE 9 (su
  richiesta).** La FASE 9 è un **risultato negativo** senza variazione del conteggio test
  (175 invariati): per contratto (§12) aveva aggiornato i `.md` vivi `changelog_tecnico.md`,
  `esperimenti_rag.md` (ESP-09) e `architettura_rag.md` §5, mentre relazioni, presentazione
  e deliverable non riflettevano ancora l'esito. Questo passo li riallinea **senza**
  anticipare l'integrazione narrativa completa (FASE 16). Interventi: (1)
  `relazione_unilaw_agent.md` e `relazione_completa_unilaw_agent.md` — aggiunta della FASE 9
  (valutata la rimozione delle regole di *routing* per corso dal prompt; l'A/B nella stessa
  sessione misura una regressione 0,90 → 0,875, quindi **routing mantenuto**, come per il
  reranker neurale della FASE 4; conteggio test **invariato a 175**); (2)
  `UniLaw_Agent_presentazione.pptx` — roadmap a «FASE 1–9 completate» (175 test invariati);
  (3) **rigenerati i sei deliverable** DOCX/PDF (DOCX via pandoc; PDF via WeasyPrint con
  copertina centrata, indice con numeri di pagina e intestazione corrente). `valutazione_rag.md`
  non toccata (nessun cambio di metriche o conteggio); `readme.md` non toccato. Resta un
  **allineamento di coerenza**, non l'integrazione narrativa del Ciclo 2 (FASE 16);
  copertina DOCX ancora a sinistra (centratura = FASE 15).
- **2026-06-21** — **Ciclo 2 — FASE 9 completata — RISULTATO NEGATIVO (routing
  mantenuto).** Aggiornate le tabelle di stato (sez. 2 e 13). Si è **valutato** di rimuovere
  da `ANSWER_STYLE_GUIDE` le cinque sezioni «Regole specifiche per <corso/topic>» di
  instradamento, perché il routing *dei documenti* è già fatto sui metadata dal reranker +
  filtro per corso (`reranking.py`) e le regole di contenuto per topic sono già emesse dal
  profilo dinamico `_build_answer_profile`. L'A/B **nella stessa sessione** (config
  predefinita, `llama3.1:8b`, `temperature=0`, 40 domande, within-session σ=0 della FASE 7)
  ha però **misurato una regressione**: behavior **0,90** (con routing, A-fresh
  `baseline_20260621_222535`) → **0,875** (senza + clausola di chiarimento,
  `_185818`/`_190346`/`_195327`) → **0,85** (rimozione pura, `_214237`). `course/topic/
  retrieval` restano invarianti a 1,0 (il routing dei documenti È ridondante), ma la prosa fa
  anche da *framing di commit* di cui il modello 8B si avvale: senza, aumentano le false
  astensioni su domande answerable borderline (q28/q30/q31, area borsa/economia) e compare
  over-answering su q18; la rimozione risolve q14/q16/q21 ma ne rompe quattro (net −1/−2).
  **Decisione: mantenere il routing nel prompt**, come per il reranker neurale (Ciclo 1 —
  FASE 4): risultato negativo riportato per onestà. Codice: `ANSWER_STYLE_GUIDE`
  **byte-identico** (solo un commento documenta la valutazione e l'esito); in
  `tests/test_knowledge.py` il guard FASE 8 `test_style_guide_keeps_tolc_routing_rules` è
  sostituito da `test_style_guide_keeps_course_routing_rules` (verifica che il routing
  **resti**, anti-rimozione accidentale) → **175 test verdi** (invariati). Doc:
  `changelog_tecnico.md` (voce dedicata con tabelle A/B e per-domanda), `esperimenti_rag.md`
  (ESP-09), `architettura_rag.md` §5 (routing valutato e mantenuto). Relazioni e deliverable
  non rigenerati (per contratto §12 l'integrazione narrativa avviene in FASE 16). Prossimo
  step: Ciclo 2 — FASE 10 (ripulire il caso speciale in `_postprocess_answer`).
- **2026-06-21** — **Allineamento di coerenza dei deliverable alla FASE 8 (su
  richiesta).** Dopo la FASE 8 (175 test), che per contratto (§12) aveva aggiornato i `.md`
  vivi `changelog_tecnico.md` ed `esperimenti_rag.md` (ESP-08), restavano disallineati il
  conteggio test di `valutazione_rag.md` (fermo a 173), le due relazioni, la presentazione e
  i deliverable (fermi a FASE 7, 173 test). Questo passo li riallinea **senza** anticipare
  l'integrazione narrativa completa (FASE 16). Interventi: (1) `valutazione_rag.md` §1.1 —
  conteggio **173 → 175** con il «+2 guard di non-regressione sulle fasce TOLC tolte dal
  prompt (Ciclo 2 — FASE 8)»; (2) `relazione_unilaw_agent.md` e
  `relazione_completa_unilaw_agent.md` — conteggio test **173 → 175** (§10/§13.1, Appendice
  e nota di consolidamento) e aggiunta della FASE 8 (fasce TOLC rimosse dalla duplicazione
  nel prompt, comportamento invariato entro la banda di rumore della FASE 7), con nuova riga
  nella tabella test della relazione completa; (3) `UniLaw_Agent_presentazione.pptx` — 173 →
  **175 test** e roadmap a «FASE 1–8 completate»; (4) **rigenerati i sei deliverable**
  DOCX/PDF (DOCX via pandoc; PDF via WeasyPrint con copertina centrata, indice con numeri di
  pagina e intestazione corrente). `readme.md` e `architettura_rag.md` non toccati: la FASE 8
  ha rimosso righe di prompt in `config.py`, senza cambiare installazione, struttura né
  moduli. Resta un **allineamento di coerenza**, non l'integrazione narrativa del Ciclo 2
  nelle relazioni (FASE 16); copertina DOCX ancora a sinistra (centratura = FASE 15).
- **2026-06-21** — **Ciclo 2 — FASE 8 completata** (togliere dal prompt le fasce numeriche
  TOLC). Aggiornate le tabelle di stato (sez. 2 e 13) e il conteggio test in sez. 3
  (173 → **175**). Primo refactor del Blocco C. Le soglie TOLC-I 9/16 erano scritte due
  volte: come unica fonte di verità in `knowledge.py` (con provenienza, FASE 7) e di nuovo,
  in prosa, nel prompt `ANSWER_STYLE_GUIDE`. Poiché la classificazione del punteggio è già
  prodotta dal **guard deterministico** (sempre attivo, legge da `knowledge.py`), la
  ripetizione nel prompt era una seconda fonte di verità. Codice: rimosse da `config.py` le
  **4 righe** delle fasce numeriche (la riga «riporta le fasce < 9, >= 9 e < 16, >= 16» e le
  tre righe di classificazione `< 9` / `>= 9 e < 16` / `>= 16`), con un commento che
  documenta la decisione; **lasciate intatte** le due righe di *routing* per corso della
  stessa sezione (target FASE 9). Test: 2 guard di non-regressione in
  `tests/test_knowledge.py` (`test_style_guide_has_no_tolc_numeric_bands`,
  `test_style_guide_keeps_tolc_routing_rules`) → **175 test verdi** (173 + 2). **Risultati
  A/B misurati** (`llama3.1:8b`, `temperature=0`, 40 domande, più campioni indipendenti per
  separare il segnale dalla banda di rumore cross-process di FASE 7): le **6 domande gestite
  dal guard** (q01, q02, q03, q09, q26, q40) hanno risposta **byte-identica** con e senza le
  fasce (il guard short-circuita prima del prompt → comportamento numerico TOLC invariante
  per costruzione); l'unico delta è q28 (non-TOLC: posti per Economia Aziendale L-18, ramo
  LLM) che in una run no-bands passa a `abstain`, ma **oscilla** sotto lo stesso prompt
  (è `answer` in altre due run) → rumore, non effetto della modifica. behavior no-bands
  {0,875; 0,90; 0,90} vs con-bands {0,90; 0,90}: delta ≤1 domanda, **entro la banda di
  rumore di FASE 7**. Report: `baseline_20260620_152206` (A con fasce),
  `baseline_20260620_153742`/`_20260621_112434`/`_20260621_113108` (B/B2/B3 senza fasce).
  Doc: `changelog_tecnico.md` (voce dedicata con tabella A/B), `esperimenti_rag.md` (ESP-08).
  Relazioni e deliverable DOCX/PDF non rigenerati: per contratto (§12) l'integrazione
  narrativa avviene in FASE 16. Prossimo step: Ciclo 2 — FASE 9 (de-duplicare le regole di
  routing per corso nel prompt).
- **2026-06-20** — **Allineamento di coerenza dei deliverable alla FASE 7 (su
  richiesta).** Dopo la FASE 7 (173 test), che per contratto (§12) aveva aggiornato solo i
  `.md` vivi (`valutazione_rag.md` §1.1/§12/§14.4/§16, `changelog_tecnico.md`), relazioni,
  presentazione e deliverable erano rimasti fermi alla FASE 6 (160 test). Questo passo li
  riallinea **senza** anticipare l'integrazione narrativa completa (che resta la FASE 16).
  Interventi: (1) `relazione_unilaw_agent.md` e `relazione_completa_unilaw_agent.md` —
  conteggio test **160 → 173** (§10/§13.1, Appendice e nota di consolidamento della
  misura), aggiunta della FASE 7 con la quantificazione della variabilità (`--repeat N`,
  σ=0 *within-session* su 5 run, banda «≤±1 domanda» solo fra sessioni) e nuova riga nella
  tabella test della relazione completa; (2) `UniLaw_Agent_presentazione.pptx` — 160 →
  **173 test** e roadmap a «FASE 1–7 completate»; (3) **rigenerati i sei deliverable**
  `relazione_unilaw_agent.{docx,pdf}`, `relazione_completa_unilaw_agent.{docx,pdf}`,
  `roadmap_progetto.{docx,pdf}` (DOCX via pandoc; PDF via WeasyPrint con copertina
  centrata, indice con numeri di pagina e intestazione corrente). `readme.md` e
  `architettura_rag.md` non toccati: la FASE 7 non ha cambiato installazione, struttura né
  moduli. Resta un **allineamento di coerenza** (numeri e deliverable), non l'integrazione
  narrativa del Ciclo 2 nelle relazioni (FASE 16); copertina DOCX ancora a sinistra
  (centratura = FASE 15).
- **2026-06-20** — **Ciclo 2 — FASE 7 completata** (quantificare la variabilità del
  modello locale). Aggiornate le tabelle di stato (sez. 2 e 13). La fase dà un **numero**
  alla variabilità finora dichiarata solo come «±1 domanda», necessaria per leggere gli
  A/B del Blocco C. Codice (`eval/run_eval.py`): nuova opzione **`--repeat N`** che esegue
  il dataset N volte e produce un report di variabilità con **media e deviazione standard**
  (campionaria, ddof=1) per metrica e l'elenco delle domande che oscillano; funzioni pure
  `aggregate_repeats`/`aggregate_per_question` (testabili offline) e
  `write_variability_report`. Per robustezza ogni run salva **subito** il proprio
  `baseline_*.json` (un'interruzione non perde più il lavoro fatto) e
  **`--aggregate-reports`** ricostruisce il report di variabilità da report già salvati,
  **offline**. Test: nuovo `tests/test_eval_variability.py` (13 test offline) → **173 test
  verdi** (160 + 13). **Risultati misurati** (report
  `eval/reports/variability_20260620_131229.*`; `llama3.1:8b`, `temperature=0,0`, 5
  esecuzioni × 40 domande): tutte le metriche **identiche** su 5 run — behavior **0,90**,
  abstention/reason **0,923**, course/topic/retrieval/citation **1,00** — con **σ = 0,0**
  ovunque e **nessuna** domanda che oscilla (risposte byte-identiche). Verificata l'assenza
  di cache nel percorso eval (`setup_redis_cache` è solo nella UI; ogni run ha richiesto
  ~8–13 min di inferenza reale): la σ=0 è **vero determinismo greedy** a `temperature=0`,
  non un artefatto. Lettura onesta: la variabilità *within-session* (modello caldo, prompt
  identici, pipeline fissa) è nulla; la storica «±1 domanda» (q17) era un effetto
  **cross-session/cross-fase**. Banda di rumore: «≈0 entro la sessione; ≤±1 domanda tra
  sessioni diverse». Doc: `valutazione_rag.md` (§1.1 conteggio test, §12 e §14.4
  aggiornate, nuova §16 con tabelle), `changelog_tecnico.md` (voce dedicata). Relazioni e
  deliverable DOCX/PDF non rigenerati: per contratto (§12) l'integrazione narrativa avviene
  in FASE 16. Prossimo step: Ciclo 2 — FASE 8 (togliere dal prompt le fasce numeriche TOLC,
  primo refactor del Blocco C, da valutare in A/B con la banda di rumore appena misurata).
- **2026-06-19** — **Allineamento di coerenza dei deliverable alle FASE 5–6 (su
  richiesta).** Dopo le FASE 5 (149 test) e 6 (160 test), che per contratto (§12) avevano
  aggiornato solo i `.md` vivi (`changelog_tecnico.md`, `valutazione_rag.md`), i
  deliverable e le relazioni erano rimasti fermi alla FASE 4 (141 test). Questo passo li
  riallinea **senza** anticipare l'integrazione narrativa completa (che resta la FASE 16).
  Interventi: (1) `valutazione_rag.md` §1.1 — conteggio **149 → 160** con il «+11
  validazione held-out della soglia di astensione (Ciclo 2 — FASE 6)»; (2)
  `relazione_unilaw_agent.md` e `relazione_completa_unilaw_agent.md` — conteggio test
  **141 → 160** (incluse §10/§13.1 e i comandi in Appendice), aggiunta una nota di
  consolidamento della misura su FASE 5 (scoring dell'eval dai segnali del trace, verdetti
  stabili al *wording*) e FASE 6 (soglia 0,37 riprodotta dai dati ≈0,367 e validata 3/3
  sugli held-out), aggiornati i caveat «da ri-validare su held-out» ora risolti; (3)
  `UniLaw_Agent_presentazione.pptx` — 141 → **160 test** e roadmap a «FASE 1–6 completate»;
  (4) **rigenerati i sei deliverable** `relazione_unilaw_agent.{docx,pdf}`,
  `relazione_completa_unilaw_agent.{docx,pdf}`, `roadmap_progetto.{docx,pdf}` (DOCX via
  pandoc; PDF via WeasyPrint con copertina centrata, indice con numeri di pagina e
  intestazione corrente). `readme.md` e `architettura_rag.md` non toccati: FASE 5–6 non
  hanno cambiato installazione, struttura né moduli. Resta un **allineamento di coerenza**
  (numeri e deliverable), non l'integrazione narrativa del Ciclo 2 nelle relazioni, che
  resta la **FASE 16**. La copertina DOCX resta allineata a sinistra (centratura = FASE 15).
- **2026-06-19** — **Ciclo 2 — FASE 6 completata** (validazione held-out della soglia di
  astensione). Aggiornate le tabelle di stato (sez. 2 e 13). La soglia
  `ABSTENTION_OOD_MAX_STRENGTH` (0,37) governa la distinzione fra le due cause
  *quantitative* di astensione — `fuori_dominio` (retrieval strength bassa) vs
  `evidenza_insufficiente` (strength alta) — ed era stata fissata a mano e misurata sugli
  stessi casi (q17/q18/q19). Questa fase separa **calibrazione** e **validazione**.
  Codice: estratta in `abstention.py` la decisione binaria `classify_by_strength`
  (`classify_llm_abstention` la richiama → **refactor a comportamento invariato**) e
  aggiunte le funzioni pure `calibrate_ood_threshold` (soglia a massimo margine dai dati)
  e `threshold_accuracy`; **nessun cambio a `config.py`** (la calibrazione *riproduce*
  0,37). Nuovo harness `eval/abstention_threshold_validation.py` che misura la
  `retrieval_strength` dei negativi *threshold-relevant* con la sola pipeline di recupero
  (Chroma + BM25 + RRF, **nessun Ollama**), calibra sui soli storici e valida sugli
  held-out q34–q39 (riservati in FASE 4, mai usati per calibrare). Test: nuovo
  `tests/test_abstention_threshold.py` (11 test offline + consistenza sul report) →
  **160 test verdi** (149 + 11). **Risultati** (report
  `eval/reports/abstention_threshold_validation.*`; retrieval deterministico): la
  calibrazione a massimo margine sui soli storici dà **0,3667** ≈ la 0,37 di config (la
  soglia è quindi *riproducibile dai dati*, non arbitraria); sugli held-out q35/q36/q37 —
  **mai visti** — la soglia classifica **3 su 3** (accuratezza **1,00** sia con 0,37 sia
  con 0,3667), con margine netto fra fuori-dominio (≤0,333) e insufficiente (≥0,500). È la
  validazione fuori campione che mancava (cfr. §11.2 di `valutazione_rag.md`), che isola
  sulla sola decisione di soglia l'evidenza preliminare 6/6 della FASE 4. Limite onesto:
  insiemi piccoli (3+3) → evidenza coerente, non garanzia statistica; la
  `retrieval_strength` resta lessicale (la versione semantica + ri-taratura è la FASE 13,
  che dipende da questo split). Doc: `valutazione_rag.md` (§11.2 aggiornata + nuova §15
  con tabelle), `changelog_tecnico.md` (voce dedicata). Relazioni e deliverable DOCX/PDF
  non rigenerati: per contratto (§12) l'integrazione narrativa avviene in FASE 16.
  Prossimo step: Ciclo 2 — FASE 7 (quantificare la variabilità del modello locale,
  `--repeat N`).
- **2026-06-18** — **Ciclo 2 — FASE 5 completata** (scoring dell'eval dai segnali del
  trace). Aggiornate le tabelle di stato (sez. 2 e 13). Codice: `eval/run_eval.py` —
  `classify_behavior(answer, trace)` deduce l'esito (answer/abstain/clarify/
  unknown_course) in primo luogo dalla **causa di astensione strutturata** del `RagTrace`
  (`abstention_reason`, impostata dall'agente su ogni ramo) invece che dai soli marcatori
  testuali; i marcatori restano come **fallback** e l'LLM non raggiungibile — che non
  lascia segnale nel trace — è ancora riconosciuto dal testo. Aggiunti il campo
  `behavior_source` (trace/testo) per domanda e i conteggi aggregati nel report; refactor
  degli import (dipendenze pesanti ed effetti collaterali spostati in `main()`) per
  rendere il modulo importabile offline dai test. Test: nuovo `tests/test_eval_scoring.py`
  (8 test offline, trace sintetici, incluso il caso che motiva la fase — un'astensione che
  lo scorer testuale mancherebbe ma il trace coglie — e un controllo di consistenza sul
  report più recente) → **149 test verdi** (141 + 8). **Verdict-neutra per costruzione e
  verificata**: sulla baseline a 40 domande i verdetti coincidono **riga per riga** con il
  report FASE 4 (`baseline_20260618_132450`), 0 domande cambiate; nuovo report
  `eval/reports/baseline_20260618_185545.*` (`llama3.1:8b`): behavior **0,90**,
  course/topic **1,0**, retrieval/citation **1,0**, abstention/reason **0,923**, e **tutti
  i 40 verdetti dal trace** (`behavior_from_trace: 40`). Il valore è nella *stabilità* al
  variare del wording, non in un cambio di metriche (identiche). Doc: `valutazione_rag.md`
  (§1.1, §1.3 con la metodologia di scoring dal trace), `changelog_tecnico.md` (voce
  dedicata). Relazioni e deliverable DOCX/PDF non rigenerati: per contratto (§12)
  l'integrazione narrativa avviene in FASE 16. Prossimo step: Ciclo 2 — FASE 6
  (validazione held-out della soglia di astensione, sui negativi q34–q39 della FASE 4).
- **2026-06-18** — **Allineamento documentale alla FASE 4 (su richiesta).** Oltre al
  conteggio test (**132 → 141**), questa volta sono cambiate le **metriche-vetrina** per
  via dell'ampliamento del dataset (20 → 40 domande): le relazioni e la presentazione
  sono state portate alla **nuova baseline rappresentativa** — *behavior* **0,90**,
  astensione **0,923**, corso/argomento e retrieval/citazioni **1,00** — con una lettura
  onesta del calo da 0,95 (su 20 domande) a 0,90 (test più ampio e severo, non una
  regressione; famiglia di false astensioni q14/q21/q29). Interventi: `relazione_completa`
  §13.1–13.3, `relazione_unilaw_agent` (Abstract, §9 con la baseline a 40 domande, §12,
  Conclusioni), `UniLaw_Agent_presentazione.pptx` (slide Risultati e roadmap, FASE 1–4) e
  rigenerazione di tutti i deliverable (PDF impaginati + DOCX). I risultati storici su 20
  domande restano riportati come percorso del Ciclo 1. È un allineamento di coerenza con
  `valutazione_rag.md` (§14); l'integrazione narrativa completa resta la FASE 16.
- **2026-06-18** — **Ciclo 2 — FASE 4 completata** (ampliamento del dataset di
  valutazione). Aggiornate le tabelle di stato (sez. 2 e 13). Dati: dataset
  `eval/questions_baseline.jsonl` portato da **20 a 40 domande** (q21–q40), con
  copertura su quattro assi — più corsi/argomenti (L-16 prove finali/norme redazionali,
  L-19 piano studi, Informatica prova finale, **economia** mai testata prima, Erasmus,
  borsa), parafrasi/sinonimi, distrattori e **negativi held-out q34–q39** per la FASE 6;
  le 20 domande storiche restano invariate. Nomi file risolti dal disco reale e valori
  normativi verificati sui PDF (grounding). Test: nuovo `tests/test_eval_dataset.py`
  (9 test offline) che valida schema e **coerenza deterministica delle etichette** contro
  `infer_query_intent` → **141 test verdi** (132 + 9). Nuova baseline a 40 domande
  (`eval/reports/baseline_20260618_132450.*`, `llama3.1:8b`): behavior **0,90** (36/40),
  course/topic **1,0**, retrieval/citation **1,0** (27/27), abstention **0,923** (12/13),
  reason **0,923** (12/13). Esito istruttivo e onesto: i 4 errori di behavior sono del
  modello locale (q14 storico + **q21, q29 nuovi** = false astensioni con la fonte giusta
  recuperata → famiglia di errori, non caso isolato; q18 over-answering), mentre **tutti
  i 6 negativi held-out si astengono e sono classificati correttamente** (evidenza
  preliminare per la FASE 6) e l'intent resta 1,0 anche su corso nuovo e parafrasi.
  `retrieval_hit` resta 1,0: i distrattori non spiazzano il gold (per stressarlo servono
  documenti-trappola, estensione futura). Doc: `valutazione_rag.md` (§1.1–1.3 e nuova
  §14), `changelog_tecnico.md` (voce dedicata). Relazioni e deliverable DOCX/PDF non
  rigenerati: per contratto (§12) l'integrazione narrativa avviene in FASE 16. Prossimo
  step: Ciclo 2 — FASE 5 (scoring dell'eval più robusto, Blocco B).
- **2026-06-18** — **Allineamento di coerenza dei deliverable (su richiesta).** I `.md`
  (relazioni, `valutazione_rag.md`) erano già a **132** (FASE 3). Interventi: (1)
  **ripristino dei PDF delle relazioni** alla versione impaginata con copertina centrata
  e stile curato (`relazione_unilaw_agent.pdf`, `relazione_completa_unilaw_agent.pdf`),
  che erano stati rigenerati con pandoc semplice perdendo la copertina; (2)
  **presentazione `UniLaw_Agent_presentazione.pptx` aggiornata** da 130 a **132 test** e
  slide roadmap a «FASE 1–3 completate». Resta un **allineamento di coerenza**
  (numeri/deliverable), non l'integrazione narrativa del Ciclo 2 nelle relazioni (FASE 16).
- **2026-06-18** — **Ciclo 2 — FASE 3 completata** (blocco fonti onesto in astensione).
  Aggiornate le tabelle di stato (sez. 2 e 13). Codice: nuovo parametro
  `abstaining` in `citations.format_sources_block` — in astensione il fallback
  `sources[:3]` non è più etichettato «Fonti utilizzate» ma rietichettato onestamente
  «Documenti consultati (nessuno utilizzato per la risposta):», mentre le citazioni
  `[F#]` reali restano sempre mostrate come «Fonti citate»; in `agent.py:answer` l'esito
  di `is_abstention` è catturato una volta e inoltrato al blocco fonti; il delegante e i
  rami a template deterministico restano invariati (default `abstaining=False`). Test: 2
  nuovi in `tests/test_citations.py` (**132 test verdi**). **Neutralità eval per
  costruzione**: lo scorer estrae i documenti citati dai nomi file (non dal titolo del
  blocco) e calcola `citation_hit_rate` solo sulle answerable → tutte le metriche
  invariate; ispezione dal vivo dei negativi: su q19 compare il blocco onesto. Doc:
  `changelog_tecnico.md` (voce dedicata). Relazioni e deliverable DOCX/PDF non
  rigenerati: per contratto (§12) l'integrazione narrativa avviene in FASE 16. Prossimo
  step: Ciclo 2 — FASE 4 (ampliare il dataset di valutazione, Blocco B).
- **2026-06-18** — **Allineamento di coerenza della documentazione (su richiesta).**
  Portato il conteggio test corrente **123 → 130** nelle relazioni
  (`relazione_unilaw_agent.md`, `relazione_completa_unilaw_agent.md`) e in
  `valutazione_rag.md` (§1.1, con il +7 del Ciclo 2 — FASE 2); lasciate intatte le voci
  storiche di fase. **Rigenerati tutti i deliverable** (`relazione_unilaw_agent.{pdf,docx}`,
  `relazione_completa_unilaw_agent.{pdf,docx}`, `roadmap_progetto.{pdf,docx}`): i PDF con
  impaginazione curata. Come per la FASE 1, questo è un **allineamento di coerenza** (numeri
  e deliverable), non l'integrazione narrativa completa del Ciclo 2 nelle relazioni, che
  resta la **FASE 16**.
- **2026-06-18** — **Ciclo 2 — FASE 2 completata** (normalizzazione del formato
  citazioni `(F#)→[F#]`). Aggiornate le tabelle di stato (sez. 2 e 13). Codice: nuova
  funzione pura `citations.normalize_citations` (parentesi singole e di gruppo +
  riferimenti "nudi", limitata agli indici di fonti realmente recuperate), agganciata in
  `agent.py:answer` **prima** di `strip_invalid_citations`/grounding/blocco fonti; i
  template deterministici restano intatti. Test: 7 nuovi in `tests/test_citations.py`
  (130 test verdi). Documentazione: `changelog_tecnico.md` (voce dedicata). Relazioni e
  deliverable DOCX/PDF non rigenerati: per contratto (§12) l'integrazione narrativa nelle
  relazioni avviene in FASE 16. Prossimo step: Ciclo 2 — FASE 3 (blocco fonti onesto in
  astensione), sinergica con questa fase.
- **2026-06-17** — **Allineamento di coerenza della documentazione (su richiesta).**
  Portato il conteggio test corrente **115 → 123** nelle relazioni
  (`relazione_unilaw_agent.md`, `relazione_completa_unilaw_agent.md`) e in
  `valutazione_rag.md` (§1.1, con il +8 del Ciclo 2 — FASE 1); lasciate intatte le voci
  storiche di fase (es. "+5 → 115" alla FASE 8). **Rigenerati tutti i deliverable**
  (`relazione_unilaw_agent.{pdf,docx}`, `relazione_completa_unilaw_agent.{pdf,docx}`): i
  PDF con copertina centrata e impaginazione curata; i DOCX via pandoc con copertina
  ancora allineata a sinistra (la centratura resta la **FASE 15**). Questo è un
  **allineamento di coerenza**, non l'integrazione narrativa completa del Ciclo 2, che
  resta la **FASE 16**.
- **2026-06-17** — **Ciclo 2 — FASE 1 completata** (fix bug separatore migliaia in
  `tools.py`). Aggiornate le tabelle di stato (sez. 2 e 13). Codice: ramo "un punto + 3
  cifre = migliaia" in `_normalizza_numero_italiano`; test convertito in correttezza +
  test parametrico (123 test verdi). Documentazione: `changelog_tecnico.md` (voce
  dedicata), `valutazione_rag.md` (§5 segnata corretta). Relazioni e deliverable
  DOCX/PDF non rigenerati: per contratto (§12) l'integrazione nelle relazioni avviene in
  FASE 16 (con la copertina di FASE 15), per non lasciare i deliverable disallineati.
- **2026-06-17** — Esteso a roadmap di progetto completa: aggiunta la Parte I (Ciclo 1,
  FASE 0–9 + ESP-07, argomentate con problema/approccio/esito/proseguimento e tabella di
  evoluzione delle metriche) e la sintesi a colpo d'occhio; rinominato da
  `roadmap_ciclo2.md` a `roadmap_progetto.md`. La Parte II (Ciclo 2, FASE 1–16) resta il
  piano da svolgere.
- **2026-06-17** — Creazione (come `roadmap_ciclo2.md`): piano del Ciclo 2 con analisi,
  motivazioni, dipendenze e tabella di avanzamento.
