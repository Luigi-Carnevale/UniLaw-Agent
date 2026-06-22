# Problemi noti e risoluzioni — UniLaw Agent

Registro dei problemi operativi riscontrati sul progetto e delle soluzioni
applicate. È un documento di troubleshooting: serve a riconoscere rapidamente un
sintomo già visto, capirne la causa radice e applicare la correzione o la
procedura di recupero corretta senza ridiagnosticare da zero.

Ogni voce riporta: sintomo, contesto, diagnosi svolta, causa radice, soluzione
applicata (con file/righe), verifica, procedura di recupero immediato e note di
prevenzione. Le date sono in formato `AAAA-MM-GG`.

## Indice dei problemi

| ID | Data | Problema | Stato |
|---|---|---|---|
| P-01 | 2026-06-22 | Rebuild della knowledge base da GUI → `sqlite3.OperationalError: attempt to write a readonly database` | Risolto |

---

## P-01 — Rebuild da GUI: «attempt to write a readonly database»

**Stato:** risolto (2026-06-22).
**Componenti coinvolti:** `database.py`, ChromaDB 0.4.24, Streamlit (processo
long-running).

### Sintomo

Avviando il rebuild della knowledge base dall'interfaccia grafica, dopo l'analisi
dei 22 PDF la procedura termina con errore e, da quel momento, **fare una domanda
non funziona più**. In log:

```
ERROR | database | Errore inizializzazione knowledge base
Traceback (most recent call last):
  File ".../database.py", line 253, in inizializza_conoscenza
    db = Chroma.from_documents(...)
  ...
  File ".../chromadb/db/mixins/embeddings_queue.py", line 172, in submit_embeddings
    results = cur.execute(sql, params).fetchall()
sqlite3.OperationalError: attempt to write a readonly database
```

(Gli `ERROR ... posthog | capture() takes 1 positional argument` e il warning
`torch.classes ... __path__._path` nello stesso log sono rumore innocuo:
telemetria ChromaDB disattivata e ispezione interna di PyTorch. Non sono la causa.)

### Diagnosi svolta

Il messaggio suggerisce un problema di permessi/disco, ma le verifiche lo hanno
**escluso**:

- Indice configurato in `~/Library/Application Support/UniLawAgent/chroma_db`
  (fuori da iCloud), directory e file `chroma.sqlite3` di proprietà dell'utente,
  permessi `-rw-r--r--`, nessun flag BSD immutabile (`uchg`).
- Disco con ~224 GB liberi.
- **Test decisivo:** un processo Python pulito che apre *lo stesso* file
  `chroma.sqlite3` ed esegue `CREATE/INSERT/COMMIT` riesce senza errori
  (`scrittura OK`). Quindi il file non è realmente readonly.
- `lsof -p <pid_streamlit>` ha mostrato il dato chiave: **due inode diversi sullo
  stesso path** `chroma.sqlite3` — uno da 12 MB (inode `7335247`, l'indice
  precedente funzionante, con i file HNSW `data_level0.bin`, `header.bin`, ecc.) e
  uno nuovo da ~8 KB (inode `7957702`). Il processo Streamlit girava **da prima**
  del rebuild (avviato alle 22:25, rebuild alle 22:34).

Due inode sullo stesso nome = il file è stato **cancellato e ricreato sotto i
piedi di un processo che lo teneva ancora aperto**.

### Causa radice

Il rebuild dalla GUI avviene **dentro il processo Streamlit già in esecuzione**,
che tiene la knowledge base aperta perché `inizializza_conoscenza` è cachata con
`@st.cache_resource`. Il flusso del pulsante "ricostruisci" (in `app_agent.py`)
fa `inizializza_conoscenza.clear()` + `force_rebuild=True`; al rerun,
`_delete_existing_index()` esegue `shutil.rmtree()` sulla cartella dell'indice.

Il punto critico: **ChromaDB 0.4 mantiene in cache, per tutta la durata del
processo, un "System" (con la relativa connessione SQLite) per ogni
`persist_directory`** (`SharedSystemClient`). Svuotare la cache di Streamlit non
chiude quella connessione. Quindi, dopo che `rmtree` ha cancellato i file:

1. la vecchia connessione SQLite resta in cache, ora **orfana** (il file su disco
   non esiste più);
2. il nuovo `Chroma.from_documents` **riusa quella connessione cachata** invece di
   aprirne una nuova sul file ricreato;
3. alla prima scrittura (`upsert` degli embeddings) SQLite rileva che il file sotto
   l'handle è stato rimosso/sostituito e risponde `attempt to write a readonly
   database`.

Da lì la KB in memoria è inutilizzabile e ogni domanda fallisce. Un **processo
appena avviato** non ha quella cache, ed è il motivo per cui il test isolato
scriveva senza problemi: non è mai stato un problema di permessi o di iCloud.

### Soluzione applicata

In `database.py`, dentro `_delete_existing_index()`, dopo la `rmtree` si **svuota
la cache di sistema di ChromaDB**, così il rebuild apre una connessione nuova sul
file appena ricreato:

```python
def _delete_existing_index() -> None:
    index_path = Path(CHROMA_PERSIST_DIRECTORY)

    if index_path.exists():
        shutil.rmtree(index_path)

    # ChromaDB (>=0.4) tiene in cache, per tutta la durata del processo, un "System"
    # (con la connessione SQLite) per ogni persist_directory. Dopo la rmtree un nuovo
    # client riuserebbe quella connessione orfana -> "attempt to write a readonly
    # database". Va svuotata perché il rebuild apra una connessione nuova.
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception as exc:  # difensivo: il percorso dell'API può variare tra versioni
        logger.warning("Impossibile svuotare la cache di sistema ChromaDB: %s", exc)
```

Effetto: da ora il rebuild dalla GUI **funziona senza dover riavviare** l'app. In
un processo appena avviato (cache vuota) la chiamata è un no-op. L'`except` rende
la correzione robusta a eventuali cambi di percorso dell'API tra versioni di
ChromaDB.

### Verifica

- `chromadb.__version__` → `0.4.24`; `from chromadb.api.client import
  SharedSystemClient; SharedSystemClient.clear_system_cache()` → import e chiamata
  OK.
- `database.py` supera il controllo di sintassi (`ast.parse`).

### Recupero immediato (una tantum)

Il processo che ha generato l'errore ha ancora in memoria lo stato rotto e il
**vecchio** codice: va riavviato una volta. L'indice su disco è rimasto a metà,
conviene ripulirlo:

```bash
kill <PID_streamlit>          # oppure Ctrl-C nel terminale dell'app
rm -rf "$HOME/Library/Application Support/UniLawAgent/chroma_db"
cd "/Users/luigicarnevale/Desktop/UniLaw-Agent-main"
.venv/bin/streamlit run app_agent.py
```

Al riavvio l'indice non viene trovato e viene ricostruito pulito (22 PDF →
chunk); dopodiché le domande funzionano. Per individuare il PID:
`ps aux | grep "streamlit run app_agent.py"`.

### Note e prevenzione

- **Indice fantasma in iCloud.** Dentro il progetto può restare un vecchio
  `./.chroma_db/chroma.sqlite3` (a 0 byte) da una configurazione precedente che
  salvava l'indice nella cartella di progetto sul Desktop sincronizzato da iCloud.
  Non è più usato (`config.py` salva in `~/Library/Application Support`,
  sovrascrivibile con `UNILAW_CHROMA_DIR`); si può rimuovere senza conseguenze
  (`rm -rf ./.chroma_db`). La scelta di tenere l'indice fuori da iCloud è proprio
  per evitare che la sincronizzazione blocchi il file SQLite durante le scritture.
- **Regola generale.** Cancellare/sostituire i file di un database SQLite mentre
  un processo li tiene aperti porta a errori `readonly`/corruzione. Con ChromaDB
  persistente, dopo aver rimosso la cartella dell'indice nello stesso processo,
  svuotare sempre `SharedSystemClient.clear_system_cache()` prima di ricostruire.
