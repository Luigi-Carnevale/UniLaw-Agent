<!--
  Relazione completa di progetto su UniLaw Agent: descrizione a 360 gradi del
  sistema (non un changelog delle modifiche). Convertibile in PDF/DOCX con pandoc
  (vedi Appendice). Simboli mantenuti ASCII-safe per una conversione pulita.
-->

<div align="center">

![Logo Università degli Studi di Salerno](../assets/logo_unisa.png){ width=180px }

# Università degli Studi di Salerno
## Dipartimento di Informatica

**Corso: Fondamenti di Intelligenza Artificiale**

---

# UniLaw Agent
### Relazione completa di progetto — un assistente RAG locale per i documenti universitari

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

1. Sommario esecutivo
2. Il problema e il contesto
3. Obiettivi del progetto
4. Requisiti e vincoli
5. Visione d'insieme dell'architettura
6. Il corpus documentale
7. La pipeline RAG, stadio per stadio
8. Comprensione della domanda: intento e memoria
9. Conoscenza normativa e regole deterministiche
10. Affidabilità: citazioni, astensione, confidenza
11. Interfaccia utente e osservabilità
12. Tecnologie utilizzate
13. Qualità e valutazione del sistema
14. Punti di forza
15. Limiti del sistema
16. Sviluppi futuri
17. Conclusioni
18. Appendice operativa
19. Glossario dei termini tecnici

<div style="page-break-after: always;"></div>

## 1. Sommario esecutivo

UniLaw Agent è un assistente conversazionale che permette di interrogare in
linguaggio naturale la documentazione universitaria ufficiale (regolamenti di
accesso, piani di studio, bandi, regolamenti della prova finale, guide) e di
ottenere risposte **fondate sui documenti** e **corredate dalle fonti**. Il sistema
è realizzato secondo il paradigma *Retrieval-Augmented Generation* (RAG) ed è
**interamente locale**: documenti, indice, modello di embedding e modello
linguistico risiedono ed eseguono sulla macchina dell'utente, senza dipendere da
servizi esterni.

La caratteristica distintiva del progetto non è la sola capacità di "rispondere",
ma l'attenzione alla **affidabilità**: il sistema cita le fonti che ha realmente
utilizzato, verifica che le citazioni corrispondano ai documenti recuperati, si
**astiene** in modo motivato quando le informazioni non sono presenti e chiede
**chiarimenti** quando la domanda è ambigua. Una rete di test automatici e una
valutazione sperimentale ripetibile accompagnano il sistema, rendendone la qualità
misurabile e non semplicemente dichiarata.

## 2. Il problema e il contesto

Chi studia in un ateneo deve spesso reperire un'informazione puntuale — una soglia
di punteggio per l'accesso, un requisito, una scadenza, le regole della prova
finale — all'interno di documenti normativi lunghi, scritti in linguaggio
burocratico e distribuiti su numerosi file PDF non sempre uniformi. La ricerca
manuale è lenta e soggetta a errori; d'altra parte, un assistente conversazionale
generico rischia di "inventare" dettagli plausibili ma non presenti nei documenti
(fenomeno noto come *allucinazione*), particolarmente dannoso in un dominio
normativo dove l'esattezza conta.

Un sistema RAG è adatto a questo scenario perché **vincola** la generazione del
modello ai contenuti effettivamente recuperati da una base documentale controllata:
riduce le risposte non fondate e rende possibile indicare con precisione la fonte
(file e pagina) di ciascuna affermazione. La scelta di un'architettura
**local-first** risponde a esigenze di privacy (nessun documento lascia la macchina)
e di riproducibilità (l'indice è persistente e il modello opera a temperatura nulla).

Il dominio applicativo è la documentazione di alcuni corsi di laurea dell'Università
degli Studi di Salerno: Informatica L-31, Scienze dell'Educazione L-19, Scienze
dell'Amministrazione e dell'Organizzazione L-16, oltre a documenti trasversali quali
il bando per la borsa di studio, il bando Erasmus e le guide alla tesi online.

## 3. Obiettivi del progetto

Il progetto persegue un sistema RAG che sia:

- **document-grounded**: ogni risposta deriva dai documenti recuperati e ne cita le
  fonti verificabili;
- **prudente**: si astiene o chiede chiarimenti invece di rispondere a vuoto;
- **robusto**: combina ricerca semantica e lessicale per non perdere termini esatti
  e codici;
- **trasparente**: ogni risposta è accompagnata da una traccia ispezionabile ed
  esportabile del ragionamento di recupero;
- **misurabile**: la qualità è valutata con test automatici e un dataset di domande;
- **estendibile**: l'aggiunta di nuovi PDF aggiorna automaticamente l'indice.

L'obiettivo non è una demo, ma un sistema *production-like*: progettato per
funzionare ragionevolmente anche su domande nuove, formulate male, incomplete o
fuori tema.

## 4. Requisiti e vincoli

**Requisiti funzionali.** Rispondere a domande in linguaggio naturale sui documenti
indicizzati; citare le fonti (file e, quando disponibile, pagina); riconoscere
corso e argomento della domanda; astenersi quando l'informazione non è presente;
chiedere chiarimento quando manca il corso; gestire un corpus aggiornabile con
ricostruzione dell'indice al variare dei PDF; gestire richieste di calcolo numerico
semplice.

**Requisiti non funzionali.** Esecuzione locale (privacy, nessun servizio cloud);
riproducibilità (temperatura nulla, indice persistente, firma del corpus);
tracciabilità del comportamento; robustezza ai PDF problematici (errori gestiti,
non bloccanti); affidabilità e citazione delle fonti come requisiti di qualità
prioritari.

**Vincoli.** Stack local-first basato su Ollama (modello linguistico), ChromaDB
(indice vettoriale) ed embedding multilingua locali; nessuna libreria pesante
introdotta senza motivazione; il sistema deve restare eseguibile su CPU.

## 5. Visione d'insieme dell'architettura

Il sistema è organizzato in moduli con responsabilità distinte, orchestrati da una
classe centrale (`UniLawResponder`) e presentati da un'interfaccia web Streamlit.

| Modulo | Responsabilità |
|---|---|
| `app_agent.py` | Interfaccia Streamlit, sessione, comandi, debug e stato |
| `agent.py` | Orchestrazione della pipeline RAG (`UniLawResponder`) |
| `intent.py` | Riconoscimento di corso/argomento e predicati di dominio |
| `retrieval.py` | Retrieval ibrido: query di espansione, ricerca vettoriale, BM25, fusione RRF |
| `reranking.py` | Reranking euristico e filtro metadata per corso |
| `neural_reranker.py` | Reranker neurale opzionale (cross-encoder multilingua) |
| `evidence.py` | Selezione dei passaggi più pertinenti (evidenze brevi e mirate) |
| `citations.py` | Estrazione e verifica delle citazioni |
| `abstention.py` | Classificazione della causa di astensione |
| `confidence.py` | Stima euristica dell'affidabilità |
| `knowledge.py` | Conoscenza normativa strutturata e tracciabile |
| `trace_export.py` | Esportazione della traccia in JSON/Markdown |
| `rag_types.py` | Modello dati (intento, fonte recuperata, traccia) |
| `database.py` | Ingestione PDF, chunking, embedding, ChromaDB, manifest |
| `config.py` | Costanti, prompt, configurazione e stile |
| `tools.py` | Calcolo numerico sicuro |

Lo stack tecnologico e il flusso di elaborazione sono descritti nelle sezioni 7 e 12.

## 6. Il corpus documentale

La base di conoscenza è costituita da un insieme di PDF ufficiali collocati nella
cartella `documenti/`. Al momento il corpus comprende ventidue documenti che, una
volta indicizzati, producono circa millequattrocento frammenti testuali (*chunk*).

A ogni documento vengono associati, in fase di ingestione, due metadati inferiti
dal nome del file:

- il **corso** (`course_tag`): Informatica, Scienze dell'Educazione, Scienze
  dell'Amministrazione, area economico-statistica, oppure "generale";
- il **tipo di documento** (`doc_type`): accesso, borsa, erasmus, piano di studi,
  tesi/prova finale, regolamento, guida, oppure "altro".

Questi metadati alimentano sia il filtro per corso sia il reranking, permettendo di
privilegiare le fonti pertinenti ed escludere quelle di altri corsi. La qualità
complessiva delle risposte dipende, in ultima analisi, dalla qualità e
dall'aggiornamento dei PDF indicizzati.

## 7. La pipeline RAG, stadio per stadio

Il flusso completo, dalla domanda alla risposta, attraversa gli stadi seguenti
(i rami marcati "senza modello" non richiedono il modello linguistico):

```
domanda utente
  -> calcolo numerico?            --sì-> risposta aritmetica            [senza modello]
  -> riconoscimento intento (corso, argomento, ambiguità, memoria)
       -> corso non riconosciuto  ----> astensione "fuori dominio"      [senza modello]
       -> domanda ambigua         ----> richiesta di chiarimento        [senza modello]
  -> retrieval ibrido (vettoriale + BM25, fusione RRF, deduplicazione)
  -> reranking euristico (+ reranker neurale opzionale) + filtro per corso
       -> nessuna evidenza        ----> astensione "retrieval debole"   [senza modello]
  -> evidence selection (passaggi brevi e mirati) + stima confidenza
  -> eventuale risposta deterministica controllata (casi normativi critici)
  -> costruzione del contesto e generazione vincolata (modello locale)
  -> verifica delle citazioni (rimozione riferimenti inventati + grounding)
  -> se astensione: classificazione della causa
  -> risposta + interpretazione + confidenza + fonti + traccia
```

### 7.1 Ingestione e parsing dei PDF
I documenti vengono letti dalla cartella `documenti/` ed estratti pagina per pagina.
Gli errori di lettura su singoli file non bloccano il processo: il documento
problematico viene saltato con un avviso, preservando l'indicizzazione degli altri.

### 7.2 Suddivisione in frammenti (chunking)
Il testo viene segmentato in frammenti di lunghezza contenuta (circa 900 caratteri,
con sovrapposizione di 150) tramite un divisore ricorsivo che cerca di rispettare i
confini naturali del testo. La sovrapposizione riduce il rischio di spezzare a metà
un'informazione rilevante tra due frammenti contigui.

### 7.3 Embedding e indicizzazione
Ogni frammento viene trasformato in un vettore numerico (*embedding*) tramite un
modello multilingua adatto all'italiano, ed è memorizzato in un indice ChromaDB
persistente. Il sistema mantiene un *manifest* con una **firma** del corpus (nome,
dimensione, data di modifica e hash SHA-256 di ciascun file): confrontando la firma
corrente con quella salvata, l'indice viene ricostruito automaticamente solo quando
i documenti cambiano, evitando ricostruzioni inutili.

### 7.4 Retrieval ibrido
Il recupero dei candidati combina due "viste" complementari sui frammenti:

- una **ricerca vettoriale** semantica, eseguita su più riformulazioni della
  domanda (con espansioni mirate per corso e argomento) e con criterio di
  *Maximal Marginal Relevance* per favorire la diversità dei risultati;
- una **ricerca lessicale BM25**, che premia la corrispondenza dei termini esatti e
  dei codici (sigle, numeri di articolo, denominazioni) che la sola semantica può
  non cogliere.

I due insiemi di risultati vengono fusi con la *Reciprocal Rank Fusion* (RRF), una
tecnica che combina gli ordinamenti senza dover normalizzare punteggi eterogenei:
ogni documento riceve un punteggio pari alla somma, sui due arm, di 1/(k + rango).
I risultati vengono inoltre deduplicati. La fusione produce un insieme di candidati
con maggiore copertura (*recall*), mentre l'ordinamento fine è affidato allo stadio
successivo.

### 7.5 Reranking e filtro per corso
I candidati vengono riordinati da un **reranker euristico** che applica priori di
dominio: premia le fonti del corso e del tipo documentale pertinenti, riconosce
parole chiave e nomi di file rilevanti e penalizza i documenti chiaramente fuori
tema. Un **filtro per metadati** evita inoltre che una domanda su un corso specifico
sia risolta con documenti di un altro corso. È disponibile, in opzione, un
**reranker neurale** basato su un cross-encoder multilingua, che riordina i primi
candidati confrontando direttamente domanda e passaggio; è disattivato per
impostazione predefinita e, qualora non disponibile, il sistema ricade
automaticamente sull'ordinamento euristico.

### 7.6 Selezione delle evidenze
Prima di interrogare il modello, ciascuna fonte selezionata viene ridotta ai
**passaggi più pertinenti** alla domanda: il frammento è suddiviso in frasi, le
frasi vengono ordinate per sovrapposizione lessicale con la domanda e se ne
trattiene un sottoinsieme (con un minimo garantito, per non perdere informazione, e
un tetto di lunghezza). In questo modo il modello riceve un contesto **più breve e
mirato**, con meno rumore e minore tendenza a divagare verso contenuti tangenziali.

### 7.7 Costruzione del contesto e generazione
Le evidenze selezionate vengono assemblate in un contesto in cui ogni fonte è
etichettata con un riferimento citabile ([F1], [F2], ...) comprensivo di file e
pagina. Il modello linguistico locale riceve il contesto, lo stile di risposta e la
domanda, con l'istruzione di rispondere **esclusivamente** sulla base del contesto e
di citare i riferimenti utilizzati. La temperatura nulla favorisce risposte
riproducibili.

### 7.8 Verifica delle citazioni
Dopo la generazione, il sistema rimuove eventuali riferimenti [F#] **inventati**
(che non corrispondono ad alcuna fonte realmente recuperata) e misura il
**supporto**: per ogni frase che cita una fonte, controlla la sovrapposizione
lessicale con il contenuto della fonte citata. Quando il supporto è debole, applica
una politica prudente di "riduzione" (abbassa la confidenza e aggiunge una nota di
trasparenza) anziché bloccare la risposta, evitando di sopprimere risposte
potenzialmente corrette su un modello locale.

### 7.9 Astensione e confidenza
Vedi sezione 10.

## 8. Comprensione della domanda: intento e memoria

Prima del recupero, il sistema interpreta la domanda riconoscendone il **corso** e
l'**argomento** (accesso/TOLC, borsa, Erasmus, prova finale/tesi, piano di studi).
Distingue inoltre i **corsi non presenti** nel corpus (per esempio Medicina o
Giurisprudenza), così da potersi astenere, e segnala le **domande ambigue**, quando
un argomento è citato senza indicare il corso.

La **memoria conversazionale** è volutamente minimale, "a slot": il sistema conserva
soltanto l'ultimo corso e l'ultimo argomento, non l'intera cronologia. Questo
consente di gestire domande ellittiche — ad esempio "E per la tesi?" dopo una
domanda su Informatica — riusando il corso precedente, senza però contaminare il
recupero con messaggi non pertinenti. È una scelta che privilegia la prevedibilità
del comportamento rispetto alla flessibilità di una memoria estesa.

## 9. Conoscenza normativa e regole deterministiche

In un dominio normativo alcune informazioni critiche (soglie, fasce, strutture di
prova) richiedono esattezza. Il sistema adotta perciò, accanto alla generazione, un
insieme misurato di **regole deterministiche**, classificate per natura:

- **Regole di calcolo e parsing** (calcolo aritmetico sicuro tramite analisi
  sintattica controllata; estrazione del punteggio dalla domanda): semplici,
  sicure, coperte da test.
- **Conoscenza normativa strutturata**: i valori normativi (per esempio le soglie di
  punteggio per l'accesso a Informatica L-31, o la struttura della prova di
  ammissione a Scienze dell'Educazione L-19) sono centralizzati in un unico modulo
  dati, ciascuno corredato dalla **provenienza** (citazione testuale verificata sul
  documento di origine). In questo modo i valori sono auditabili, non duplicati e
  non "inventati": il classificatore e le tabelle delle risposte li leggono da
  un'unica fonte di verità.
- **Guard numerico sulle soglie TOLC-I**: l'unico template attivo per impostazione
  predefinita. Applica le soglie di accesso a Informatica L-31 al punteggio
  indicato dallo studente, garantendo l'esattezza del verdetto (con/senza OFA) e la
  citazione della fonte canonica (il regolamento di accesso).
- **Controlli di affidabilità**: verifica delle citazioni e classificazione
  dell'astensione (sezione 10).

L'impostazione segue un principio di **fiducia graduata**: dove un errore numerico
sarebbe dannoso (le soglie di accesso), una regola deterministica fornisce garanzie;
in tutti gli altri casi è il recupero unito alla generazione vincolata a costruire la
risposta. Una verifica sperimentale (sezione 13) ha mostrato che, disattivando i
template "di prosa" che in passato riscrivevano intere risposte, il RAG generativo
mantiene la stessa accuratezza: tali template sono perciò **disattivati di default**
(riattivabili con `UNILAW_PROSE_TEMPLATES=1`) e il sistema risponde, per la grande
maggioranza delle domande, leggendo direttamente i documenti.

## 10. Affidabilità: citazioni, astensione, confidenza

L'affidabilità è il filo conduttore del progetto e si articola in tre meccanismi.

**Citazioni verificabili.** Ogni affermazione sostanziale rimanda a una fonte [F#]
con file e pagina; il sistema mantiene solo le citazioni corrispondenti a fonti
realmente recuperate e segnala quando il supporto testuale è debole.

**Astensione classificata.** Quando non può rispondere in modo fondato, il sistema
**si astiene dichiarando la causa**, distinguendo cinque situazioni:

| Causa | Significato |
|---|---|
| Corso fuori dominio | la domanda riguarda un corso non presente nel corpus |
| Domanda ambigua | manca il corso necessario a selezionare le fonti |
| Retrieval debole | nessun documento pertinente è stato recuperato |
| Fuori dominio | la domanda esula dall'ambito dei documenti indicizzati |
| Fonte presente ma insufficiente | esistono documenti pertinenti, ma non contengono in modo esplicito la risposta |

La distinzione tra "fuori dominio" e "fonte presente ma insufficiente" si basa su
una misura di pertinenza lessicale tra la domanda e le fonti recuperate: è ciò che
consente di dire onestamente "non è scritto nei documenti" invece di un generico
"non lo so".

**Confidenza.** A ogni risposta è associata una stima euristica di affidabilità
(alta/media/bassa) con motivazione, basata sulla coerenza tra fonti, corso e
argomento; la confidenza viene abbassata quando le citazioni risultano poco
supportate.

## 11. Interfaccia utente e osservabilità

L'interfaccia web (Streamlit) presenta una chat, un pannello di stato (knowledge
base, cache, numero di documenti, modello), un riepilogo della **pipeline** attiva
(retrieval ibrido, reranker, evidence selection, verifica citazioni, astensione) e
comandi operativi (ricostruzione dell'indice, caricamento di nuovi PDF, reset della
chat e della memoria).

Il cuore della trasparenza è la **traccia** (`RagTrace`): per ogni risposta il
sistema registra la domanda, corso e argomento riconosciuti, l'uso della memoria, la
modalità di retrieval e i punteggi di fusione, il reranker impiegato, la riduzione
operata dall'evidence selection, l'esito della verifica delle citazioni, la causa di
eventuale astensione, le query generate, le fonti selezionate e i documenti
scartati. La traccia è consultabile nell'interfaccia ed **esportabile in JSON e
Markdown**, così da poter essere allegata a una relazione o analizzata offline.

## 12. Tecnologie utilizzate

| Componente | Tecnologia | Ruolo |
|---|---|---|
| Interfaccia | Streamlit | Chat, stato, debug, comandi |
| Modello linguistico | Ollama, `llama3.1:8b` (temperatura 0) | Generazione vincolata al contesto |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (CPU) | Vettorializzazione di frammenti e query |
| Indice vettoriale | ChromaDB persistente | Ricerca semantica per similarità |
| Indice lessicale | BM25 Okapi (`rank_bm25`) | Recupero per termini esatti |
| Reranker opzionale | cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` | Riordino neurale dei candidati |
| Cache (opzionale) | Redis | Cache delle risposte del modello, se disponibile |
| Calcolo sicuro | Analisi sintattica con whitelist di operatori | Aritmetica senza esecuzione di codice arbitrario |

Tutte le componenti sono pensate per funzionare in locale e su CPU.

## 13. Qualità e valutazione del sistema

La qualità del sistema è verificata con due strumenti complementari.

### 13.1 Test automatici
Una suite di **230 test** (eseguibili offline con `python -m pytest`, senza modello
né indice) copre i componenti del sistema. **Esito: 230/230 superati.**

| Area coperta dai test | Esempi di verifica |
|---|---|
| Calcolo numerico sicuro | aritmetica, percentuali, rifiuto di input non ammessi |
| Classificazione punteggio TOLC | confini delle fasce (8/9/15.9/16), estrazione del valore |
| Riconoscimento intento | corso, argomento, ambiguità, corso ignoto, memoria |
| Metadati e firma del corpus | `course_tag`/`doc_type`, rilevamento modifiche |
| Retrieval ibrido | tokenizzazione, BM25, fusione RRF |
| Reranker neurale | riordino per punteggio, fallback |
| Evidence selection | selezione passaggi, minimo garantito |
| Verifica citazioni | rimozione riferimenti inventati, grounding |
| Astensione | classificazione della causa nei vari rami |
| Esportazione traccia | JSON valido, sezioni Markdown |
| Integrità del dataset di valutazione | schema e coerenza deterministica delle etichette |
| Scoring dell'eval dal trace | esito dedotto dai segnali strutturati della traccia |
| Soglia di astensione (held-out) | calibrazione dai dati e validazione fuori campione |
| Variabilità del modello (`--repeat N`) | aggregazione media/σ e per-domanda, report offline |
| Prompt senza fasce TOLC | guard di non-regressione: soglie solo in `knowledge.py`, routing conservato |
| Post-processing dell'astensione | l'astensione non viene riscritta né mascherata; confidenza ridotta in caso di incertezza |
| Intent detection semantica | gap-fill da embedding senza override delle keyword; default OFF, cablaggio |
| Grounding semantico delle citazioni | recupero per similarità di embedding additivo al lessicale; default OFF, fallback |
| `retrieval_strength` semantica | forza per embedding query↔fonte; soglia 0,53 calibrata e validata su held-out; default OFF |
| Mitigazione q14 (profilo di risposta) | predicato regolamento generale + hint gated; default ON, blast radius q14 |

### 13.2 Valutazione sperimentale
Un dataset di **40 domande etichettate** (20 storiche più 20 aggiunte nel Ciclo 2 —
FASE 4: facili, difficili, ambigue, fuori dominio, senza risposta, con sinonimi,
formulate male, con distrattori e negativi *held-out*) viene eseguito da uno script di
valutazione che misura, tramite la traccia di ogni risposta, le metriche seguenti.
Risultati sul dataset a 40 domande (modello locale `llama3.1:8b`); la colonna *Baseline
FASE 4* è la misura storica del 2026-06-18, la colonna *Finale* è la configurazione
predefinita a chiusura del Ciclo 2 (dopo la mitigazione di q14, FASE 14):

| Metrica | Baseline FASE 4 | Finale (FASE 14) | Base di calcolo |
|---|---|---|---|
| Correttezza del comportamento (rispondere/astenersi/chiarire) | 0,90 | **0,925** | 37 / 40 |
| Accuratezza riconoscimento corso | 1,00 | 1,00 | 40 / 40 |
| Accuratezza riconoscimento argomento | 1,00 | 1,00 | 40 / 40 |
| Retrieval hit-rate (documento atteso tra le fonti) | 1,00 | 1,00 | 27 / 27 |
| Correttezza delle citazioni | 1,00 | 1,00 | 27 / 27 |
| Astensione corretta sui casi negativi | 0,923 | 0,923 | 12 / 13 |
| Accuratezza della causa di astensione | 0,923 | 0,923 | 12 / 13 |

**Lettura onesta del dato.** Sul dataset originale di 20 domande la stessa
configurazione otteneva *behavior* 0,95 e astensione 1,00; l'ampliamento a 40 domande
(nuovi corsi come Economia, parafrasi, distrattori e sei negativi *held-out*) fornisce
una **stima più rappresentativa**. Il calo da 0,95 a 0,90 **non è una regressione del
sistema** — invariato — ma l'effetto di un test più ampio e severo: ha fatto emergere
una *famiglia* di false astensioni del modello locale (il caso storico q14 più i nuovi
q21 e q29, in cui la fonte corretta è recuperata e citata ma il modello 8B si astiene
cercando un dettaglio più specifico), oltre a un caso di *over-answering* (q18). I
quattro errori di comportamento sono tutti del modello generativo, non del recupero
(retrieval e citazioni restano 1,00) né dell'etichettatura. Di questi, **q14 è stato poi
risolto** dalla mitigazione mirata della FASE 14 (behavior 0,90 → 0,925); q16/q21/q29
restano il limite di generazione residuo. I **sei negativi
*held-out*** sono gestiti correttamente — tutti con astensione e causa giusta —
evidenza preliminare che la soglia di astensione generalizza oltre il set di
calibrazione. Questa generalizzazione è stata poi confermata formalmente nel
**Ciclo 2 — FASE 6**: isolando la sola decisione di soglia, la
`ABSTENTION_OOD_MAX_STRENGTH = 0,37` di configurazione è riprodotta dai dati storici
(≈0,367 a massimo margine) e, sui negativi *held-out* mai usati per calibrare,
classifica correttamente la causa in **3 casi su 3** (con margine netto tra
fuori-dominio e fonte insufficiente). Dettaglio in `docs/valutazione_rag.md` (§15).

Infine, nel **Ciclo 2 — FASE 7** la variabilità del modello locale — finora dichiarata
solo come «±1 domanda» — è stata **quantificata** con la nuova opzione `--repeat N`:
su 5 esecuzioni dell'intero dataset (40 domande, `temperature=0`) tutte le metriche
risultano identiche, con **deviazione standard 0** e nessuna domanda oscillante. La
variabilità *entro la sessione* è quindi nulla (determinismo *greedy* effettivo); la
banda storica «±1 domanda» è un effetto fra sessioni diverse. Dettaglio in
`docs/valutazione_rag.md` (§16).

Misurata la banda di rumore, il **Ciclo 2 — FASE 8** ha avviato il Blocco C dei refactor
di qualità rimuovendo una duplicazione: le soglie numeriche TOLC-I (9/16) erano scritte
sia in `knowledge.py` (unica fonte di verità, con provenienza) sia, in prosa, nel prompt
`ANSWER_STYLE_GUIDE`. Poiché la classificazione del punteggio è già prodotta dal guard
deterministico che legge da `knowledge.py`, le righe nel prompt erano una seconda fonte
di verità ed è stata rimossa. L'A/B conferma il comportamento invariato: le domande
gestite dal guard danno risposta byte-identica e gli altri casi restano entro la banda di
rumore della FASE 7. Il totale dei test automatici sale così a **175**.

Il **Ciclo 2 — FASE 9** ha invece prodotto un **risultato negativo**, riportato per
onestà metodologica: si è valutato di togliere dal prompt anche le regole di *routing* per
corso (ridondanti per il recupero, già svolto da reranker e filtro per metadata), ma l'A/B
nella stessa sessione ha misurato una regressione del comportamento (0,90 → 0,875). Quella
prosa, oltre al routing dei documenti, fornisce al modello un *framing* utile a rispondere
sui casi borderline; rimuoverla aumenta le false astensioni. Come per il reranker neurale
(Ciclo 1 — FASE 4), la scelta è stata di **mantenere** il routing nel prompt, con un test
di guardia contro la rimozione accidentale (conteggio test invariato a 175).

Il **Ciclo 2 — FASE 10** ha infine ripulito un caso speciale del post-processing: un ramo
attivo su argomento "accesso" e documenti di tipo accesso/regolamento/«altro»
**riscriveva** l'astensione con un testo generico, rendendola invisibile al rilevatore di
astensione e quindi scavalcando sia la classificazione della causa (FASE 6) sia il blocco
fonti onesto (FASE 3). Il ramo — di fatto dormiente sul dataset — è stato rimosso: il
post-processing ora si limita ad abbassare la confidenza in caso di incertezza, senza
riscrivere la risposta del modello. L'A/B conferma il comportamento invariato e il refactor
è coperto da 6 nuovi test unitari, portando il totale a **181**.

Il **Ciclo 2 — FASE 11** mitiga il limite dell'intent detection a sole parole chiave
(fragile verso le parafrasi). Un nuovo modulo (`semantic_intent.py`) classifica corso e
argomento per **similarità di embedding**, riusando l'embedder del vector store senza nuove
dipendenze, e **affianca** le keyword riempiendo solo le caselle lasciate vuote, senza mai
sovrascrivere un riconoscimento esistente. È **opt-in, disattivato di default** (soglie del
coseno ancora provvisorie). Poiché sul dataset attuale le keyword coprono già corso e
argomento su tutte le domande, l'attivazione è neutra per costruzione sull'eval; su 9
parafrasi senza i token-keyword il classificatore semantico ne recupera 7. Resta disponibile
in attesa di validare le soglie su un set più ampio; con i 17 test del modulo il totale sale
a **198**.

Il **Ciclo 2 — FASE 12** applica la stessa idea alla verifica delle citazioni: il grounding
introdotto in FASE 5 misurava il supporto delle frasi citanti per sola sovrapposizione
lessicale, bocciando le frasi corrette ma parafrasate. Ora un controllo opzionale per
**similarità di embedding** recupera quelle frasi quando sono semanticamente vicine a una
frase della fonte; la rete semantica si aggiunge al lessicale e con embedder assente il
risultato è invariato. La soglia 0,16 è calibrata dai dati (su una sonda separa 7/7
parafrasi ed estranee), ma la separazione sulle parafrasi normative è sottile: come per il
reranker e l'intent semantico, la funzione è **opt-in, default OFF**. Agendo solo sulla
confidenza (politica «reduce»), è neutra sulle metriche-vetrina; con i 10 test del modulo il
totale sale a **208**.

Il **Ciclo 2 — FASE 13** chiude il Blocco C portando la misura semantica nella distinzione
delle cause di astensione (FASE 6): la *retrieval strength* lessicale (soglia 0,37)
penalizzava le query parafrasate. Una forza per similarità di embedding query↔fonte affianca
quella lessicale; la soglia è stata ri-calibrata dai dati a **0,53** e validata **3/3
sull'held-out** della FASE 6, con i due segnali concordi su tutti i negativi. Come le altre
estensioni semantiche è **opt-in, default OFF** ed è neutra sull'eval per costruzione; con
gli 11 test del modulo il totale sale a **219**.

Il Blocco D si apre con la **mitigazione mirata** del caso q14 (Ciclo 2 — FASE 14), aperto
fin dal Ciclo 1: la falsa astensione sulla consultabilità della tesi, la cui regola vive in
un *regolamento generale* di Ateneo recuperato e completo nel contesto. Un predicato puro
riconosce il regolamento generale fra le fonti e, nel ramo consultabilità, aggiunge al
profilo di risposta un *hint* che autorizza l'uso della regola generale senza dettarne il
contenuto. È **attivo di default** e gated sulla situazione di retrieval; con blast radius
verificato (su 40 domande solo q14 attiva l'hint), q14 passa da astensione a risposta
corretta e il comportamento predefinito sale da 0,90 a **0,925 (37/40)**, a parità di tutte
le altre metriche. Con gli 11 test del modulo il totale sale a **230**. Le altre false
astensioni della stessa famiglia (q16/q21/q29), non di consultabilità, restano un limite di
generazione residuo del modello locale.

### 13.3 Il RAG legge davvero i documenti (RAG puro vs regole)
Per verificare che il sistema **non dipenda da risposte codificate**, la valutazione
è stata ripetuta disattivando completamente le regole deterministiche: ogni domanda
viene così risposta dal solo RAG generativo (retrieval, lettura del modello,
citazioni). Il confronto seguente è stato eseguito sul **set di 20 domande** (Ciclo 1):

| Configurazione | Comportamento | Citazioni | Retrieval |
|---|---|---|---|
| RAG puro (nessuna regola) | 0,90 | 0,923 | 1,00 |
| Predefinita (solo guard numerico TOLC) | 0,95 | 1,00 | 1,00 |

Togliendo del tutto le regole, l'accuratezza aggregata e il retrieval **non
calano**: le domande prima gestite da template (graduatorie e requisiti della borsa,
adempimenti Erasmus, ecc.) vengono risposte correttamente leggendo i documenti.
L'unico vantaggio residuo delle regole riguarda l'**esattezza numerica** delle
soglie di accesso e la citazione della fonte canonica — motivo per cui resta attivo
solo quel guard. Nella configurazione predefinita, su 13 domande con risposta solo 4
(le domande sul punteggio TOLC) passano dal guard numerico: le altre 9 sono **RAG
puro**.

**Nota di onestà sulla riproducibilità.** Le metriche dipendenti dalla generazione
oscillano di **circa una domanda** tra esecuzioni, perché il modello locale non è
perfettamente deterministico nemmeno a temperatura nulla: vanno lette come stime
puntuali. La saturazione del retrieval hit-rate riflette anche la dimensione
contenuta del corpus di test e non va confusa con una robustezza generale.

## 14. Punti di forza

- **Local-first**: privacy e riproducibilità, nessuna dipendenza da servizi esterni.
- **Risposte fondate e citate**: riferimenti a file e pagina, citazioni verificate.
- **Astensione affidabile e motivata**: il sistema sa dire, e spiegare, "non lo so".
- **Retrieval robusto**: semantico più lessicale, con fusione trasparente.
- **Conoscenza normativa auditabile**: valori critici con provenienza dalle fonti.
- **Trasparenza**: traccia completa, ispezionabile ed esportabile.
- **Misurabilità**: test automatici e valutazione ripetibile.
- **Aggiornabilità**: ricostruzione automatica dell'indice al variare del corpus.

## 15. Limiti del sistema

- La qualità dipende dalla qualità e dall'aggiornamento dei PDF; l'estrazione di
  tabelle dai PDF può essere imperfetta.
- Il modello linguistico locale (8 miliardi di parametri) ha capacità inferiori ai
  modelli di grande scala: può occasionalmente astenersi pur avendo fonti adeguate o
  introdurre lieve variabilità tra esecuzioni.
- Il reranking euristico e alcune regole sono tarati sul corpus attuale: andranno
  rivalutati su un corpus più ampio e su domande non viste.
- Il riconoscimento dell'intento si basa su liste di parole chiave, fragili verso
  formulazioni nuove o inattese.
- La verifica delle citazioni e la misura di pertinenza sono lessicali, non
  semantiche: non effettuano inferenza linguistica profonda.
- La valutazione, pur ripetibile e ampliata nel Ciclo 2 (da 20 a 40 domande), si basa
  ancora su un dataset contenuto.

## 16. Sviluppi futuri

- Riconoscimento dell'intento basato su rappresentazioni semantiche, in alternativa
  alle liste di parole chiave.
- Verifica delle citazioni con inferenza semantica (NLI) oltre al grounding
  lessicale.
- Reranker neurale calibrato sul dominio o fusione euristico/neurale, per non
  perdere i priori di dominio.
- Parsing tabellare dedicato e visualizzazione diretta delle pagine citate.
- Ampliamento del corpus e del dataset di valutazione, con domande non viste.
- Gestione delle versioni dei documenti ed esportazione delle conversazioni.

## 17. Conclusioni

UniLaw Agent dimostra come un assistente RAG locale possa essere non solo
funzionante, ma **affidabile e verificabile**: risponde citando le fonti, si astiene
in modo motivato quando le informazioni mancano, rende trasparente e tracciabile il
proprio percorso di recupero e fonda la conoscenza normativa critica su dati
auditabili. Il valore del progetto sta tanto nelle singole componenti quanto
nell'impianto complessivo, orientato a ridurre il rischio di risposte non fondate e
a rendere la qualità un fatto misurabile. I limiti residui — dipendenza dalla
qualità del corpus, capacità del modello locale, dimensione del set di valutazione —
delineano una direzione di lavoro chiara, senza intaccare l'utilizzabilità reale del
sistema nel suo dominio.

## 18. Appendice operativa

**Prerequisiti.** Python 3.10+, Ollama con il modello `llama3.1:8b`, le dipendenze
Python del progetto. Redis è opzionale (cache).

**Installazione e avvio.**
```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama serve
streamlit run app_agent.py
```

**Test e valutazione.**
```bash
python -m pytest                  # test automatici (offline)
python eval/run_eval.py           # valutazione completa (Ollama attivo)
python eval/retrieval_ablation.py # confronto retrieval (senza modello)
```

**Uso.** Inserire i PDF nella cartella `documenti/` (anche tramite l'uploader della
sidebar) e porre domande dalla chat. La traccia di ogni risposta è esportabile in
JSON/Markdown dal pannello di debug.

**Conversione di questo documento in PDF/DOCX.**
```bash
pandoc docs/relazione_completa_unilaw_agent.md -o relazione_completa.docx --toc --resource-path=docs
pandoc docs/relazione_completa_unilaw_agent.md -o relazione_completa.pdf  --toc --resource-path=docs --pdf-engine=xelatex
```

**Relazione di dettaglio tecnico-progettuale.** Per il percorso di sviluppo a fasi,
le scelte tecniche e i confronti sperimentali, si rimanda a
`docs/relazione_unilaw_agent.md`, `docs/architettura_rag.md`,
`docs/valutazione_rag.md` ed `docs/esperimenti_rag.md`.

## 19. Glossario dei termini tecnici

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
