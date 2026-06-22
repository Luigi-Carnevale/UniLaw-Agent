# Valutazione del sistema RAG — UniLaw Agent

Questo documento descrive il dataset di valutazione, le metriche, i risultati
della baseline e la loro interpretazione. La baseline è la fotografia del
sistema **prima** di qualsiasi miglioramento del retrieval, e serve come termine
di confronto per le fasi successive.

> I risultati riportati sono stati **effettivamente eseguiti** il 2026-06-16 con
> il modello locale `llama3.1:8b`. Report grezzi:
> `eval/reports/baseline_20260616_224551.json` e `.md`.

## 1. Metodologia

### 1.1 Test automatici (`python -m pytest`)
230 test offline (62 in FASE 1, +4 confidence in FASE 2, +9 retrieval in FASE 3,
+7 reranker in FASE 4, +10 evidence/citazioni in FASE 5, +10 astensione in FASE 6,
+8 conoscenza normativa in FASE 7, +5 esportazione trace in FASE 8, +8 correttezza
del calcolo nel Ciclo 2 — FASE 1, +7 normalizzazione citazioni nel Ciclo 2 — FASE 2,
+2 blocco fonti onesto in astensione nel Ciclo 2 — FASE 3, +9 integrità del dataset
di valutazione nel Ciclo 2 — FASE 4, +8 scoring dell'eval dai segnali del trace nel
Ciclo 2 — FASE 5, +11 validazione held-out della soglia di astensione nel Ciclo 2 —
FASE 6, +13 quantificazione della variabilità del modello nel Ciclo 2 — FASE 7,
+2 guard di non-regressione sulle fasce TOLC tolte dal prompt nel Ciclo 2 — FASE 8,
+6 pulizia del post-processing dell'astensione nel Ciclo 2 — FASE 10,
+17 intent detection semantica opt-in nel Ciclo 2 — FASE 11,
+10 grounding semantico delle citazioni opt-in nel Ciclo 2 — FASE 12,
+11 `retrieval_strength` semantica e ricalibrazione OOD opt-in nel Ciclo 2 — FASE 13,
+11 mitigazione q14 (hint regola generale sulla tesi) nel Ciclo 2 — FASE 14; non richiedono
Ollama né l'indice né il download di modelli) coprono: calcolo sicuro,
classificazione del punteggio TOLC, riconoscimento dell'intento, funzioni di
metadata e firma del corpus, rami di astensione e **classificazione della causa**,
gestione e verifica delle citazioni, stima di affidabilità, retrieval ibrido
(tokenizzazione, BM25, RRF, fusione), reranker neurale (riordino, fallback),
evidence selection (selezione passaggi, grounding) e **integrità del dataset di
valutazione** (schema e coerenza deterministica delle etichette, Ciclo 2 — FASE 4).
Sono test di *characterization*: fissano il comportamento attuale così da rilevare
regressioni nei refactoring.

### 1.2 Dataset di valutazione (`eval/questions_baseline.jsonl`)
**40 domande etichettate** (20 storiche + 20 aggiunte nel Ciclo 2 — FASE 4),
distribuite sulle categorie richieste:

| Categoria | N. | di cui FASE 4 | Esempio |
|---|---|---|---|
| facili (`easy`) | 11 | +4 | "Quanti CFU sono assegnati alla prova finale di Scienze dell'amministrazione L-16?" |
| difficili (`hard`) | 8 | +5 | "Quanti posti sono disponibili per l'ammissione a Economia Aziendale L-18?" |
| ambigue (`ambiguous`) | 4 | +2 | "Come si compila il piano di studi?" |
| fuori dominio (`out_of_domain`) | 6 | +3 | "Quali sono i requisiti di accesso al corso di Ingegneria Civile?" |
| senza risposta nel corpus (`no_answer`) | 3 | +1 | "Quanti studenti sono iscritti al corso di Informatica L-31?" |
| con sinonimi (`synonym`) | 5 | +3 | "Con un Ris_Test di 18 al TOLC-I per Informatica posso iscrivermi senza OFA?" |
| mal formulate (`malformed`) | 3 | +2 | "prova finale l16 quanti cfu??" |

Per ogni domanda il dataset indica corso e argomento attesi, documenti attesi e
comportamento atteso (`answer` / `abstain` / `clarify` / `unknown_course`); per le
domande negative anche la causa di astensione attesa.

**Motivazione dell'ampliamento (FASE 4).** Con sole 20 domande su un corpus "saturo"
molte metriche erano al 100% e non discriminavano. Le 20 nuove domande estendono la
copertura su quattro assi:

- **più corsi/argomenti**: prove finali e norme redazionali di L-16, piano di studi
  di L-19, prova finale di Informatica L-31, ammissione a Economia (L-18, corso
  `economia` mai testato prima), durata e adempimenti Erasmus, scadenze della borsa —
  così da esercitare documenti del corpus mai toccati dal set originale;
- **parafrasi/sinonimi**: riformulazioni di domande note (TOLC, ammissione L-19,
  graduatorie borsa) per misurare la robustezza dell'intent alle variazioni lessicali;
- **distrattori**: domande la cui risposta sta in un documento simile ma diverso da
  altri (es. *Norme redazionali* vs *Linee guida prove finali* di L-16; ammissione
  L-16 citata anche nella guida di Economia), per stressare il retrieval dove il corpus
  contiene quasi-duplicati tra i tre corsi;
- **negativi held-out (q34–q39)**: un insieme di domande negative *non* usato per
  calibrare la soglia di astensione, riservato alla validazione fuori campione della
  **Ciclo 2 — FASE 6**; copre quattro cause (`fuori_dominio_corso`, `fuori_dominio`,
  `evidenza_insufficiente`, `ambigua`).

Le etichette `expected_course`/`expected_topic` sono verificate **automaticamente**
contro `infer_query_intent` da `tests/test_eval_dataset.py` (vedi §1.1): un'etichetta
incoerente fa fallire i test invece di falsare in silenzio le metriche. I valori
normativi delle nuove domande *answerable* sono stati verificati sui PDF del corpus
(onestà del grounding).

### 1.3 Metriche (`eval/run_eval.py`)
- **behavior_accuracy** — esito previsto corretto.
- **course_accuracy / topic_accuracy** — correttezza dell'intent.
- **retrieval_hit_rate** — almeno un documento atteso tra le fonti selezionate
  (solo domande con risposta attesa e documenti attesi: 27 su 40, era 13 su 20).
- **citation_hit_rate** — almeno un documento atteso tra quelli citati.
- **abstention_rate** — sulle 13 domande "negative" (fuori dominio / senza
  risposta / ambigue; erano 7), frazione in cui il sistema **non** ha fabbricato una
  risposta.

**Classificazione dell'esito dai segnali del trace (Ciclo 2 — FASE 5).** Per
calcolare `behavior_accuracy` lo scorer deve dedurre se il sistema ha *risposto*,
*si è astenuto*, ha *chiesto chiarimenti* o ha *rifiutato un corso fuori dominio*.
In origine questa classificazione si basava sui soli **marcatori testuali** della
risposta: una soluzione fragile, perché basta che il modello cambi formulazione
perché un'astensione reale venga letta come risposta. Dalla FASE 5 `classify_behavior`
usa invece i **segnali strutturati del `RagTrace`** — in primo luogo
`abstention_reason`, che l'agente imposta su ogni ramo di astensione (sia
deterministico — corso fuori dominio, ambigua, retrieval debole — sia generativo, la
cui `is_abstention` riconosce un insieme di formulazioni più ampio di quello dello
scorer testuale). I marcatori testuali restano come **fallback** e l'LLM non
raggiungibile, che non lascia un segnale nel trace, continua a essere riconosciuto dal
testo. Il report annota per ciascun esito se il verdetto viene dal *trace* o dal
*testo* (campo `behavior_source`), per trasparenza. La modifica è **verdict-neutra sui
casi noti** (sull'ultima baseline a 40 domande i due metodi concordano riga per riga),
ma rende lo scoring stabile al variare del wording del modello.

## 2. Risultati baseline (2026-06-16, `llama3.1:8b`)

| Metrica | Valore | Base di calcolo |
|---|---|---|
| behavior_accuracy | **0,90** | 18 / 20 |
| course_accuracy | **1,00** | 20 / 20 |
| topic_accuracy | **1,00** | 20 / 20 |
| retrieval_hit_rate | **1,00** | 13 / 13 |
| citation_hit_rate | **1,00** | 13 / 13 |
| abstention_rate | **0,857** | 6 / 7 |
| domande gestite da template deterministico | 7 | q01–q03, q06–q09 |
| domande gestite dal modello (LLM) | 6 | q04, q05, q14, q15, q16, q20 |
| domande LLM non disponibili | 0 | Ollama attivo |

Test automatici: **62 passati** in ~0,2 s.

## 3. Casi riusciti (esempi)

- **q01 — calcolo soglia TOLC (template deterministico).**
  *"Ho preso 11 al TOLC-I per Informatica L-31: posso immatricolarmi?"* →
  «Sì, puoi immatricolarti, ma con OFA … fascia >= 9 e < 16 …
  `[F1] regolamento-di-accesso-informatical-31-.pdf`». Fonte corretta e citata.

- **q09 — domanda mal formulata.** *"tolc informatica 11 ofa???"* → riconosciuta
  come accesso Informatica, punteggio 11 estratto, stessa risposta corretta di
  q01. Mostra robustezza alle formulazioni sciatte quando i segnali chiave sono
  presenti.

- **q10/q11 — fuori dominio (corso non nel corpus).** Medicina e Giurisprudenza →
  astensione esplicita con elenco dei corsi disponibili, senza usare documenti
  non pertinenti.

- **q17/q19 — astensione corretta dal modello.** "Orario delle lezioni" e
  "capitale della Francia" → «Non lo so in base ai documenti disponibili.»

## 4. Casi problematici (esempi) e interpretazione

- **q14 — falsa astensione su evidenza buona (il caso più istruttivo).**
  *"La tesi di Informatica L-31 è consultabile dopo la laurea?"* Il retrieval ha
  selezionato **il documento giusto** (`regolamento-tesi-2023.pdf`, pagg. 1–3),
  ma il modello ha risposto «Non lo so in base ai documenti disponibili.» →
  **non è un problema di retrieval, ma di generazione**: il modello non ha
  sfruttato un contesto pertinente. Motiva il lavoro su evidence selection e su
  un prompt meno vincolante (FASE 5).

- **q18 — over-answering / astensione imperfetta.**
  *"Quanto costa la mensa universitaria?"* Il modello ha dichiarato che il costo
  non è specificato, ma ha poi aggiunto informazioni tangenziali (date di accesso
  al servizio) tratte dal bando borsa, invece di astenersi in modo netto. Motiva
  un layer di astensione più rigoroso (FASE 6).

- **q04/q05 — i template deterministici L-19 non si sono attivati.** Le due
  domande di accesso a Scienze dell'Educazione L-19 sono state gestite dal
  modello, non dal template `accesso_scienze_educazione_l19`: il "gate" del
  template richiede che i chunk recuperati contengano frasi esatte
  ("prova di ammissione", "80 quesiti", …), condizione non soddisfatta per queste
  formulazioni. Conseguenza qualitativa: in q05 il modello ha risposto «Sì» alla
  domanda "c'è un TOLC?", formulazione imprecisa (esiste una *prova di
  ammissione*, non un TOLC). Conferma la fragilità segnalata in FASE 0: la rete
  di sicurezza deterministica dipende da un retrieval per frasi esatte.

## 5. Difetto del calcolo numerico (`tools.py`) — corretto nel Ciclo 2
In FASE 1 (Ciclo 1) il test di caratterizzazione documentava che "5% di 20.000€"
restituiva "1" invece di "1.000": un singolo punto usato come separatore di migliaia
veniva letto come decimale da `_normalizza_numero_italiano`.

**Stato: corretto in Ciclo 2 — FASE 1 (2026-06-17).** Un punto seguito da esattamente
3 cifre (senza virgola) è ora trattato come separatore di migliaia; «5% di 20.000€»
restituisce correttamente "1.000". Il test di caratterizzazione è stato convertito in
test di correttezza (`test_percentage_thousands_separator`) e affiancato da un test
parametrico (`test_normalizza_numero_italiano`) sui formati documentati. Dettagli in
`docs/changelog_tecnico.md` (Ciclo 2 — FASE 1).

## 6. Interpretazione complessiva

- **Intent e astensione "strutturata" sono solidi**: corso/argomento al 100% e
  astensione corretta su corsi ignoti e domande ambigue (rami senza LLM).
- **La retrieval-hit pari a 1,0 va letta con cautela**: misura l'adattamento del
  reranking ai 22 file attuali, non la robustezza verso corpus nuovi. È atteso
  che questo valore scenda introducendo documenti non previsti; le FASI 3–4
  servono proprio a rendere il retrieval robusto **a parità o miglioramento** di
  questa metrica.
- **I due fallimenti riguardano la generazione**, non il recupero: il modello a
  volte si astiene pur avendo buone fonti (q14) e a volte risponde oltre la
  domanda (q18). Sono i bersagli naturali delle FASI 5–6.

## 7. Riproducibilità

```bash
python -m pytest                  # 62 test offline
python eval/run_eval.py           # baseline completa (Ollama attivo, ~7 min su CPU)
python eval/run_eval.py --limit 9 # sottoinsieme deterministico/astensione (no Ollama)
```
I report datati vengono salvati in `eval/reports/`. Le esecuzioni successive
(post-miglioramenti) andranno confrontate con questa baseline in
`docs/esperimenti_rag.md`.

## 8. Confronto FASE 3 — retrieval ibrido (vettoriale + BM25, RRF)

Misurazioni del 2026-06-16 (modello `llama3.1:8b`). Dettaglio in
`docs/esperimenti_rag.md` (ESP-02); report
`eval/reports/baseline_20260616_235334.*` e `eval/reports/retrieval_ablation.json`.

### 8.1 Ablation del retrieval (senza LLM)
Confronto modalità `vettoriale` (= baseline) vs `hybrid` sulle 13 domande con
documenti attesi:

| Metrica | vettoriale | hybrid |
|---|---|---|
| retrieval-hit | 1,00 (13/13) | 1,00 (13/13) |
| documento corretto al rango 1 | 13/13 | 13/13 |
| rango gold migliorato / uguale / peggiorato | — | 0 / 13 / 0 |
| top-k identico alle due modalità | — | 9/13 |

Nessuna regressione: il documento corretto resta al rango 1 in entrambe le
modalità. In 4 casi (q02, q03, q09, q20) BM25 modifica i chunk di *supporto* alle
posizioni 2–5 (recall lessicale aggiuntiva), senza spostare il documento principale.

### 8.2 Eval completa con LLM (20 domande)

| Metrica | baseline (FASE 1) | hybrid (FASE 3) |
|---|---|---|
| behavior_accuracy | 0,90 | 0,85 |
| course / topic | 1,0 / 1,0 | 1,0 / 1,0 |
| retrieval-hit / citation-hit | 1,0 / 1,0 | 1,0 / 1,0 |
| abstention_rate | 0,857 | 0,714 |

**Unica differenza: q17** ("orario delle lezioni"). In hybrid l'arm BM25 recupera
`piano-di-studi-l31-informatica2025-2026.pdf` (contenuto sui semestri) e il modello
risponde con la struttura in semestri invece di astenersi. L'orario puntuale resta
assente dal corpus: sul criterio stretto è un'astensione in meno. Si tratta di un
**effetto di generazione/astensione** (stessa famiglia di q18), non di qualità del
retrieval (rango del documento atteso invariato, hit 1,0).

### 8.3 Interpretazione
Su un corpus già "saturo" (hit 1,0, documento corretto al rango 1) il top-line non
può salire: il contributo di BM25 è di **recall** (verificato dal test in cui BM25
fa emergere un documento che l'arm vettoriale non restituisce). La modalità
`use_bm25=False` riproduce esattamente la baseline. Il caso q17 conferma la
necessità di evidence selection (FASE 5) e astensione robusta (FASE 6). La
robustezza reale del retrieval ibrido andrà valutata su un corpus ampliato con
documenti "trappola".

## 9. Confronto FASE 4 — reranker neurale opzionale (cross-encoder)

Misurazioni del 2026-06-17. Reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
(opt-in, default OFF) applicato ai primi 15 candidati dopo euristica e filtro corso.
Report: `eval/reports/baseline_20260617_092819.*`, `retrieval_ablation_reranker.json`.

### 9.1 Costi misurati (CPU)
Caricamento ~77 s a freddo (~1–2 s a caldo, una tantum per processo); inferenza
~4 ms/coppia (~56 ms per 15 candidati, trascurabile a query); RAM di picco ~800 MB
(circa +500 MB rispetto al solo embedding); disco ~458 MB.

### 9.2 Ablation: hybrid vs hybrid + cross-encoder (senza LLM)

| Metrica | hybrid | hybrid + cross-encoder |
|---|---|---|
| retrieval-hit | 1,00 (13/13) | 1,00 (13/13) |
| rango gold migliorato / uguale / peggiorato | — | 0 / 7 / 6 |

Il cross-encoder **peggiora il rango** del documento corretto in 6/13 casi
(es. q01 da 1 a 5): riordina per pertinenza semantica generica e non conosce i
priori di dominio (boost dell'euristica sulle fonti autorevoli).

### 9.3 Eval completa con LLM

| Metrica | hybrid (FASE 3) | hybrid + cross-encoder (FASE 4) |
|---|---|---|
| behavior_accuracy | 0,85 | 0,90 |
| abstention_rate | 0,714 | 0,714 |
| retrieval / citation | 1,0 / 1,0 | 1,0 / 1,0 |

Unica differenza positiva: **q14** passa da astensione errata a risposta corretta
(il reranker riordina il contesto sulla consultabilità della tesi, risolvendo la
falsa astensione individuata in FASE 1). q17/q18 invariati. Il behavior aggregato
torna al livello della baseline euristica (0,90).

### 9.4 Interpretazione e decisione
Si tratta di un **trade-off, non di un guadagno netto**: il cross-encoder generico
aiuta un singolo caso borderline di generazione (q14) ma degrada il rango delle
fonti autorevoli e non migliora le metriche aggregate su questo corpus tarato.
Resta quindi **opzionale e disattivato di default**, con l'euristica come
ordinamento primario; la scelta è documentata e misurata, non dichiarata. Su corpus
più ampi o meno tarati il reranker potrebbe risultare utile: per questo è mantenuto,
configurabile (`UNILAW_RERANKER=1` o toggle in sidebar) e con fallback automatico.

## 10. Confronto FASE 5 — evidence selection + verifica citazioni

Misurazioni del 2026-06-17. Evidence selection (passaggi ≤700 caratteri, min 3 /
max 6 frasi pertinenti per fonte) e verifica citazioni (rimozione [F#] inventati +
grounding lessicale con politica "reduce"), entrambe attive di default e applicate
solo al ramo LLM. Report: `eval/reports/baseline_20260617_095838.*` (default),
`_100348` (`--no-evidence`), `_100804` (evidence+reranker).

### 10.1 Risultati ed evoluzione delle metriche

| Fase / configurazione | behavior | abstention | retrieval / citation |
|---|---|---|---|
| FASE 1 — euristico (vettoriale) | 0,90 | 0,857 | 1,0 / 1,0 |
| FASE 3 — hybrid (vett.+BM25) | 0,85 | 0,714 | 1,0 / 1,0 |
| FASE 4 — hybrid + reranker | 0,90 | 0,714 | 1,0 / 1,0 |
| FASE 5 `--no-evidence` (grounding solo) | 0,85 | 0,714 | 1,0 / 1,0 |
| **FASE 5 default (evidence + grounding)** | **0,95** | **1,00** | 1,0 / 1,0 |
| FASE 5 evidence + reranker | 0,95 | 0,857 | 1,0 / 1,0 |

### 10.2 Attribuzione e interpretazione
Il miglioramento (+0,10 behavior, +0,29 abstention rispetto all'hybrid) è
**interamente dell'evidence selection**: con `--no-evidence` i valori coincidono con
l'hybrid. Le evidenze focalizzate rimuovono il contenuto tangenziale che induceva
over-answering, e **q17 e q18 ora si astengono correttamente** (abstention 7/7). Il
grounding non altera la classificazione (politica "reduce": nota di trasparenza +
confidenza più bassa) ma toglie le citazioni inventate e segnala il supporto debole.
Il reranker neurale, anche con evidence, **non** aiuta (peggiora l'astensione):
confermato OFF di default.

### 10.3 Caso residuo q14 (verificato sulle fonti)
"La tesi è consultabile dopo la laurea?": `regolamento-tesi-2023.pdf` (p.2) contiene
esplicitamente le regole (consultabile con citazione / consultabile dopo embargo di
24 mesi / non consultabile) ed è correttamente recuperato, ma il modello locale 8B
si astiene comunque. Non è un errore di retrieval né di etichettatura, ma un limite
di **generazione** del modello locale: è il caso di studio "fonte presente ma non
sfruttata" per la FASE 6.

## 11. Astensione affidabile (FASE 6)

Il layer di astensione classifica la **causa** del "non lo so" in cinque categorie:
`fuori_dominio_corso`, `ambigua`, `retrieval_debole`, `fuori_dominio`,
`evidenza_insufficiente`. La distinzione fuori-dominio vs fonte-insufficiente usa la
*retrieval strength* (quota di token della domanda coperti dalla migliore fonte),
con soglia 0,37 calibrata sui casi reali. Nuova metrica `abstention_reason_accuracy`.

### 11.1 Risultati (eval completa, 2026-06-17)

| Metrica | Valore |
|---|---|
| behavior_accuracy | 0,95 |
| abstention_rate | 1,00 |
| **abstention_reason_accuracy** | **1,00 (7/7)** |

Cause classificate correttamente: q10/q11 → `fuori_dominio_corso`; q12/q13 →
`ambigua`; q17/q18 → `evidenza_insufficiente`; q19 → `fuori_dominio`. Nessuna
regressione sulle altre metriche. Report `eval/reports/baseline_20260617_104430.*`.

### 11.2 Interpretazione e limiti
Il sistema ora **dichiara perché si astiene**, distinguendo in particolare una vera
assenza di dominio (q19) da una "fonte presente ma insufficiente" (q14/q17/q18). La
calibrazione ha corretto un'assunzione: il corpus cita il servizio mensa, quindi q18
è insufficienza di evidenza, non fuori dominio. L'accuratezza 1,0 era però misurata
sullo stesso piccolo set usato per la calibrazione della soglia: la **validazione fuori
campione** è stata svolta nella **Ciclo 2 — FASE 6** (§15), su negativi held-out mai
usati per calibrare. La `retrieval_strength` resta lessicale (non semantica): la sua
versione semantica e la ri-taratura della soglia sono la Ciclo 2 — FASE 13.

## 12. Riproducibilità e variabilità del modello locale (FASE 7 — Ciclo 1)

FASE 7 (riduzione hard-coding) è un refactor a **comportamento invariato**: 110 test
verdi e tabelle generate dai dati byte-identiche a quelle codificate. La
riesecuzione dell'eval ha però dato behavior **0,90** e abstention **0,857** (vs
0,95/1,0 della FASE 6). Il diff riguarda **una sola** domanda, q17 (ramo LLM), che a
parità di pipeline oscilla tra astensione e risposta: il modello locale
`llama3.1:8b` non è perfettamente deterministico nemmeno a `temperature=0`.

**Implicazione metodologica.** Le metriche dipendenti dalla generazione (behavior e
abstention sui casi borderline) hanno una **variabilità di ±1 domanda** tra
esecuzioni; vanno lette come stime puntuali, non come valori esatti. I componenti
deterministici (classificatore TOLC, template, intent, retrieval, q01–q13) e i test
automatici sono invece **stabili e riproducibili**. Una valutazione più robusta
richiederebbe medie su più esecuzioni e un dataset più ampio: questa banda di rumore è
stata **quantificata** nel Ciclo 2 — FASE 7 (`--repeat N`, vedi §16), che la misura
all'interno di una singola sessione e ne precisa la natura.

## 13. RAG puro vs regole, e configurazione predefinita (ESP-07)

Per verificare che il sistema sappia davvero **leggere e comprendere** i documenti e
non dipenda da risposte codificate, la valutazione è stata ripetuta disattivando le
regole deterministiche. Dettaglio in `docs/esperimenti_rag.md` (ESP-07).

| Configurazione | behavior | citation-hit | retrieval-hit |
|---|---|---|---|
| RAG puro (nessuna regola) | 0,90 | 0,923 | 1,00 |
| **Predefinita (solo guard numerico TOLC)** | **0,95** | **1,00** | **1,00** |
| Tutti i template (legacy, `--prose-templates`) | 0,90–0,95 | 1,00 | 1,00 |

**Conclusione.** Togliendo del tutto le regole l'accuratezza aggregata non cala: il
RAG generativo risponde correttamente leggendo i PDF. I 5 template "di prosa" sono
quindi **disattivati di default**; resta attivo solo il **guard numerico TOLC-I**
(esattezza delle soglie + fonte canonica). Nella configurazione predefinita, su 13
domande con risposta solo 4 (TOLC) usano il guard, le altre 9 sono RAG puro. Report:
`eval/reports/baseline_20260617_155116.*` (default), `_153822.*` (RAG puro),
`_153222.*` (con prosa).

## 14. Dataset ampliato e nuova baseline di riferimento (Ciclo 2 — FASE 4)

Il dataset è passato da 20 a **40 domande** (§1.2). Questa è la **nuova baseline di
riferimento** con cui confrontare i refactor del Blocco C. Misurazione del 2026-06-18,
modello `llama3.1:8b`, configurazione predefinita (solo guard numerico TOLC). Report:
`eval/reports/baseline_20260618_132450.*`.

### 14.1 Risultati sul dataset a 40 domande

| Metrica | 20 domande (rif. FASE 6) | 40 domande (FASE 4) | Base di calcolo |
|---|---|---|---|
| behavior_accuracy | 0,95 | **0,90** | 36 / 40 |
| course_accuracy | 1,00 | **1,00** | 40 / 40 |
| topic_accuracy | 1,00 | **1,00** | 40 / 40 |
| retrieval_hit_rate | 1,00 | **1,00** | 27 / 27 |
| citation_hit_rate | 1,00 | **1,00** | 27 / 27 |
| abstention_rate | 1,00 | **0,923** | 12 / 13 |
| abstention_reason_accuracy | 1,00 (7/7) | **0,923** | 12 / 13 |

Domande gestite dal guard numerico TOLC-I: 6 (q01–q03, q09, q26, q40); le restanti 34
passano dal RAG generativo o dai rami di astensione deterministici.

### 14.2 Casi falliti (onestà dei risultati)

I 4 errori di `behavior` sono tutti del **modello locale**, non del retrieval né
dell'etichettatura (il documento corretto è sempre recuperato e citato, retrieval/
citation 1,0):

- **q14 (storico) e q21, q29 (nuovi) — false astensioni con la fonte corretta.** Su
  «durata massima della presentazione della prova finale di Informatica L-31» (q21,
  dato presente: *max 7 minuti*) e «durata massima della mobilità Erasmus per ciclo»
  (q29, dato presente: *12 mesi*, *24 per i corsi a ciclo unico*) il documento giusto
  è recuperato e citato, ma il modello 8B si astiene cercando un dettaglio più
  specifico. È **lo stesso limite di generazione di q14**, ora osservato su corso e
  argomento diversi: l'ampliamento ha mostrato che il fenomeno **non era un caso
  isolato** ma una famiglia di errori. È il bersaglio della mitigazione mirata
  (Ciclo 2 — FASE 14).
- **q18 (storico) — over-answering.** «Quanto costa la mensa universitaria?»: il modello
  ha fornito una risposta vaga invece di astenersi (oscilla tra run per il
  non-determinismo del modello locale, §12). Unico negativo che non si è astenuto e
  unica causa di astensione non rilevata (perché ha risposto).

### 14.3 Robustezza dell'intent e dei negativi held-out

- **Intent (course/topic) resta 1,0** anche su un corso mai testato prima (`economia`,
  q28), su parafrasi/sinonimi (q25, q26, q31) e su domande mal formulate (q32, q33):
  il riconoscimento a keyword copre bene le formulazioni introdotte (la sua fragilità
  semantica resta un tema delle FASE 11/13).
- **Negativi held-out (q34–q39): 6 su 6 corretti.** Tutti si sono astenuti e tutti
  sono stati classificati con la **causa giusta** — `fuori_dominio_corso` (q34),
  `fuori_dominio` (q35, q36), `evidenza_insufficiente` (q37), `ambigua` (q38, q39).
  È **evidenza preliminare** che la soglia di astensione (0,37) e il classificatore
  generalizzano oltre il set di calibrazione; la validazione formale fuori campione è
  la **Ciclo 2 — FASE 6**, che userà proprio questo set.

### 14.4 Interpretazione e limiti

L'ampliamento ha reso le metriche **più informative** senza introdurre falsi errori:
`behavior` scende da 0,95 a 0,90 esponendo una **famiglia** di false astensioni (q14/
q21/q29) prima rappresentata da un solo caso. Per contro **retrieval/citation restano
1,0**: i distrattori (es. *Norme redazionali* vs *Linee guida prove finali* di L-16)
vengono recuperati ma **non spiazzano** il documento gold dall'insieme selezionato. Ciò
conferma il caveat del §6: con questo corpus, già ben coperto, la `retrieval_hit` non
discrimina ancora; per stressarla davvero servirebbero **documenti-trappola** aggiunti
al corpus (non solo domande-trappola) — opzione lasciata a un'estensione futura per non
toccare l'indice in questa fase. Le metriche dipendenti dalla generazione mantengono la
variabilità di ±1 domanda (§12): vanno lette come stime puntuali, e la loro banda di
rumore è stata **quantificata** nella **Ciclo 2 — FASE 7** (`--repeat N`, §16): nulla
*entro la sessione*, ≤±1 domanda *tra sessioni diverse*.

**Aggiornamento (Ciclo 2 — FASE 14): q14 risolto, default 0,925.** La tabella §14.1 è la
baseline *storica* della FASE 4 (behavior 0,90, 36/40). La mitigazione mirata della FASE 14
(hint sul regolamento generale della tesi, **attivo di default**, gated sulla situazione di
retrieval) ribalta la falsa astensione di **q14** in risposta corretta: con blast radius
verificato (su 40 domande **solo q14** attiva l'hint, tutto il resto è identico per
costruzione), il comportamento predefinito sale a **0,925 (37/40)**, mentre
course/topic/retrieval/citation restano **1,00** e abstention/reason **0,923** (q14 è una
*answerable*, non incide sui negativi). Le altre false astensioni della stessa famiglia
(q16/q21/q29) **non** sono domande di consultabilità e restano fuori dallo scopo della fase:
sono il limite di generazione residuo del modello 8B.

## 15. Validazione held-out della soglia di astensione (Ciclo 2 — FASE 6)

La soglia `ABSTENTION_OOD_MAX_STRENGTH` (0,37) decide, quando il modello si astiene in
presenza di fonti, se la causa è `fuori_dominio` (le fonti non coprono i termini della
query, *retrieval strength* bassa) o `evidenza_insufficiente` (fonti pertinenti ma
risposta assente, strength alta). Fino alla FASE 6 era **fissata a mano** e misurata
sugli stessi casi: il limite dichiarato in §11.2. Questa fase separa **calibrazione** e
**validazione**.

**Metodo (riproducibile, senza LLM).** L'harness `eval/abstention_threshold_validation.py`
misura la `retrieval_strength` dei negativi *threshold-relevant* eseguendo la sola
pipeline di recupero (Chroma + BM25 + RRF; il valore non dipende dalla generazione),
**calibra** la soglia sui soli negativi storici (q17/q18/q19) con una regola esplicita
(`abstention.calibrate_ood_threshold`: soglia a massimo margine fra le due classi) e la
**valida** sui negativi held-out q34–q39, riservati in FASE 4 e mai usati per calibrare.
Le funzioni di calibrazione e accuratezza sono pure e testate offline
(`tests/test_abstention_threshold.py`). Report:
`eval/reports/abstention_threshold_validation.*`.

### 15.1 Strength misurate e soglia calibrata

| Insieme | id | causa attesa | retrieval strength |
|---|---|---|---|
| calibrazione | q19 | `fuori_dominio` | 0,333 |
| calibrazione | q17 | `evidenza_insufficiente` | 0,400 |
| calibrazione | q18 | `evidenza_insufficiente` | 0,667 |
| held-out | q35 | `fuori_dominio` | 0,333 |
| held-out | q36 | `fuori_dominio` | 0,250 |
| held-out | q37 | `evidenza_insufficiente` | 0,500 |

La calibrazione a **massimo margine** sui soli storici dà **0,3667** — cioè la 0,37
fissata a mano (differenza di solo arrotondamento): la soglia in uso è quindi
**riproducibile dai dati**, non arbitraria.

### 15.2 Accuratezza fuori campione

| Insieme | Soglia | Accuratezza |
|---|---|---|
| calibrazione (q17/q18/q19) | 0,3667 (calibrata) | 1,00 (3/3) |
| **held-out (q35/q36/q37)** | **0,37 (config)** | **1,00 (3/3)** |
| held-out (q35/q36/q37) | 0,3667 (calibrata) | 1,00 (3/3) |

Sugli held-out — **mai visti** in calibrazione — la soglia classifica correttamente
**3 casi su 3**: c'è un margine netto fra la strength più alta dei fuori-dominio held-out
(0,333) e la più bassa degli insufficienti (0,500), entro cui cade 0,37. È la
validazione formale che mancava in §11.2; conferma e isola, sulla sola decisione di
soglia, l'evidenza preliminare della FASE 4 (§14.3, dove i 6 held-out erano 6/6
nell'eval completa).

### 15.3 Limiti

Gli insiemi sono **piccoli** (3 + 3 casi che esercitano la soglia): l'accuratezza 1,0 è
**evidenza coerente**, non garanzia statistica — il corpus e i corsi sono pochi. Le altre
tre cause (`fuori_dominio_corso`, `ambigua`, `retrieval_debole`) non dipendono dalla
soglia ma da logica deterministica (rilevamento corso, ambiguità, assenza di fonti) e
restano fuori da questa validazione. La `retrieval_strength` qui è **lessicale**: la sua
versione semantica e la ri-taratura della soglia sullo stesso split sono la **Ciclo 2 —
FASE 13** (§17), che dipende proprio da questo split calibrazione/validazione.

## 16. Banda di rumore del modello locale quantificata (Ciclo 2 — FASE 7)

Fino al Ciclo 1 la variabilità del modello locale era dichiarata solo
**qualitativamente** («±1 domanda», §12). Senza un numero non si può stabilire se un
guadagno o un calo osservato in un confronto A/B (Blocco C) è reale o rientra nel rumore.
Questa fase aggiunge a `eval/run_eval.py` l'opzione **`--repeat N`**, che esegue l'intero
dataset N volte e riporta **media e deviazione standard** (campionaria, ddof=1) di ogni
metrica, più l'elenco delle **domande che cambiano verdetto** tra le esecuzioni.

**Metodo.** `python eval/run_eval.py --repeat 5` (modello `llama3.1:8b`,
`temperature=0,0`, configurazione predefinita, 40 domande). Ogni esecuzione salva subito
il proprio `baseline_*.json`; al termine viene scritto un report di variabilità
(`eval/reports/variability_20260620_131229.*`). Le funzioni di aggregazione
(`aggregate_repeats`, `aggregate_per_question`) sono pure e testate offline
(`tests/test_eval_variability.py`). **Nessuna cache** è attiva nel percorso di
valutazione: `setup_redis_cache()` è invocata solo dalla UI (`app_agent.py`), e la prova
decisiva è il tempo — ciascuna delle 5 esecuzioni ha richiesto ~8–13 minuti di inferenza
reale (una cache avrebbe reso quasi istantanee le run 2–5).

### 16.1 Risultati (5 esecuzioni × 40 domande)

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
**byte-identiche** nelle 5 esecuzioni.

### 16.2 Interpretazione e limiti

La variabilità *within-session* — stesso modello caldo, prompt identici, pipeline fissa —
è **nulla**: a `temperature=0` la decodifica è *greedy* e quindi deterministica per prompt
identici. Questo **affina** la nota storica «±1 domanda» (q17): quell'oscillazione non era
un fenomeno per-run ma un effetto **cross-session/cross-fase** — tra una fase e l'altra
cambiavano la pipeline (e dunque il prompt) oppure lo stato del runtime del modello
(riavvio del server, carico macchina, versione di Ollama, non-determinismo numerico nel
batching). La banda di rumore è quindi: **≈0 entro la sessione; ≤±1 domanda tra sessioni
diverse**.

**Valore metodologico.** Gli A/B del Blocco C, se eseguiti **nella stessa sessione a
pipeline fissa**, sono riproducibili: una differenza di metrica osservata è **segnale**,
non rumore di campionamento. **Limite onesto.** Le 5 esecuzioni sono consecutive e a
modello caldo: misurano il **limite inferiore** della variabilità, non la varianza
cross-session, dove restano possibili rare oscillazioni di ±1 domanda. La σ=0 va letta
come «riproducibilità a pipeline e runtime fissi», non come assenza assoluta di rumore.
Per casi di interruzione, `--aggregate-reports` ricostruisce il report di variabilità dai
`baseline_*.json` già salvati, **offline**.

## 17. Retrieval strength semantica per l'astensione (Ciclo 2 — FASE 13)

La validazione della §15 isolava la **decisione di soglia** ma la `retrieval_strength`
restava **lessicale** (overlap di token domanda↔fonte): penalizza le query parafrasate,
che pur essendo in dominio non condividono token con la fonte e rischiano la causa errata
`fuori_dominio`. Questa fase aggiunge una **variante semantica opzionale**
(`abstention.semantic_retrieval_strength`): la forza è la massima similarità di embedding
fra la domanda e il contenuto di ciascuna fonte (riusa il modello del retrieval, nessuna
nuova dipendenza), e la soglia OOD è **ricalibrata e validata sullo stesso split
calibrazione/validazione** della §15. **Opt-in, default OFF**
(`ABSTENTION_SEMANTIC_STRENGTH_ENABLED`, env `UNILAW_SEMANTIC_ABSTENTION`, flag
`--semantic-abstention`): con l'opzione spenta la classificazione resta **byte-identica**
a quella lessicale. Stesso harness della §15
(`eval/abstention_threshold_validation.py`, ora con sezione lessicale **e** semantica;
nessun LLM, retrieval deterministico).

### 17.1 Strength semantiche misurate e soglia calibrata

| Insieme | id | causa attesa | strength semantica |
|---|---|---|---|
| calibrazione | q19 | `fuori_dominio` | 0,460 |
| calibrazione | q17 | `evidenza_insufficiente` | 0,597 |
| calibrazione | q18 | `evidenza_insufficiente` | 0,674 |
| held-out | q35 | `fuori_dominio` | 0,324 |
| held-out | q36 | `fuori_dominio` | 0,404 |
| held-out | q37 | `evidenza_insufficiente` | 0,663 |

La calibrazione a **massimo margine** sui soli storici dà **0,5286**; in `config.py` la
soglia è fissata a **0,53** (arrotondamento), come la 0,37 lessicale rispetto alla sua
0,3667: anche la soglia semantica è quindi **riproducibile dai dati**, non arbitraria.

### 17.2 Accuratezza fuori campione

| Insieme | Soglia | Accuratezza |
|---|---|---|
| calibrazione (q17/q18/q19) | 0,5286 (calibrata) | 1,00 (3/3) |
| **held-out (q35/q36/q37)** | **0,53 (config)** | **1,00 (3/3)** |
| held-out (q35/q36/q37) | 0,5286 (calibrata) | 1,00 (3/3) |

Sugli held-out — **mai visti** in calibrazione — la forza semantica classifica
correttamente **3 casi su 3**, come la lessicale (§15.2). I due segnali **concordano** su
tutti e 6 i negativi threshold-relevant: la versione semantica non degrada la decisione e,
sui casi parafrasati, è attesa più robusta (i test unitari con embedder finto mostrano una
parafrasi senza token in comune correttamente riconosciuta `evidenza_insufficiente` dal
semantico ma `fuori_dominio` dal solo lessicale).

### 17.3 Limiti

La separazione regge ma è **più compressa** della lessicale: in valore assoluto la forza
semantica dei casi *fuori dominio* è **alta** (0,32–0,46), perché il MiniLM assegna
similarità moderata anche a query off-topic; il margine fra le classi (≤0,46 vs ≥0,60) è
quindi più stretto di quello lessicale (≤0,33 vs ≥0,50). Gli insiemi restano **piccoli**
(3 + 3): accuratezza 1,0 è **evidenza coerente**, non garanzia statistica. Per questi
motivi — e in coerenza con reranker (FASE 4), intent semantico (FASE 11) e grounding
semantico (FASE 12) — la variante è **implementata e validata fuori campione ma lasciata
OFF di default**, in attesa di un set più ampio. Sull'eval generativo a 40 domande la
fase è **neutra per costruzione** con l'opzione OFF (l'embedder non viene nemmeno
costruito); con l'opzione ON cambierebbe solo l'etichetta di causa sulle astensioni
*threshold-relevant*, su cui i due segnali concordano, quindi `abstention_reason_accuracy`
resterebbe invariata.
