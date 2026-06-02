import glob
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import streamlit as st
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_FOLDER,
    EMBEDDING_MODEL_NAME,
    INDEX_MANIFEST_FILE,
)


logger = logging.getLogger(__name__)


def _collect_pdf_files(folder_path: str) -> list[str]:
    return sorted(glob.glob(os.path.join(folder_path, "*.pdf")))


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def calcola_firma_documenti(folder_path: str = DOCUMENTS_FOLDER) -> dict:
    """
    Crea una firma stabile dei PDF per capire se l'indice va aggiornato.
    """
    if not os.path.exists(folder_path):
        return {"documents": []}

    documents = []

    for pdf_path in _collect_pdf_files(folder_path):
        stat = os.stat(pdf_path)

        documents.append(
            {
                "filename": os.path.basename(pdf_path),
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
                "sha256": _file_hash(pdf_path),
            }
        )

    return {"documents": documents}


def _read_manifest() -> dict | None:
    manifest_path = Path(CHROMA_PERSIST_DIRECTORY) / INDEX_MANIFEST_FILE

    if not manifest_path.exists():
        return None

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    except Exception as exc:
        logger.warning("Manifest non leggibile: %s", exc)
        return None


def _write_manifest(signature: dict) -> None:
    manifest_path = Path(CHROMA_PERSIST_DIRECTORY) / INDEX_MANIFEST_FILE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _infer_course_tag(filename: str) -> str:
    name = filename.lower()

    if "informatica" in name or "l31" in name or "l-31" in name:
        return "informatica"

    if "l19" in name or "l-19" in name or "scienze dell'educazione" in name:
        return "scienze_educazione"

    if "l16" in name or "l-16" in name or "amministrazione" in name:
        return "amministrazione"

    if "economiche" in name or "statistiche" in name or "economia" in name:
        return "economia"

    return "generale"


def _infer_doc_type(filename: str) -> str:
    name = filename.lower()

    if (
        "regolamento-di-accesso" in name
        or ("accesso" in name and "regolamento" in name)
        or "immatricolazione" in name
        or "ammissione" in name
    ):
        return "accesso"

    if "bando" in name and "borsa" in name:
        return "borsa"

    if "erasmus" in name:
        return "erasmus"

    if "piano-di-studi" in name or "piani-di-studio" in name or "piano di studi" in name:
        return "piano_studi"

    if "tesi" in name or "prova-finale" in name or "prova–finale" in name or "esame-finale" in name:
        return "tesi"

    if "regolamento" in name:
        return "regolamento"

    if "guida" in name:
        return "guida"

    return "altro"


def _enrich_metadata(doc, pdf_path: str):
    filename = os.path.basename(pdf_path)

    doc.metadata["source"] = pdf_path
    doc.metadata["filename"] = filename
    doc.metadata["course_tag"] = _infer_course_tag(filename)
    doc.metadata["doc_type"] = _infer_doc_type(filename)


def _load_and_split_documents(pdf_files: list[str]):
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for pdf_path in pdf_files:
        try:
            logger.info("Analizzo PDF: %s", os.path.basename(pdf_path))

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            for doc in documents:
                _enrich_metadata(doc, pdf_path)

            chunks = splitter.split_documents(documents)

            for chunk in chunks:
                if "filename" not in chunk.metadata:
                    _enrich_metadata(chunk, pdf_path)

            all_chunks.extend(chunks)

        except Exception as exc:
            logger.warning("Errore leggendo %s: %s", os.path.basename(pdf_path), exc)

    return all_chunks


def _build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )


def _build_chroma_settings():
    return Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )


def _should_rebuild(force_rebuild: bool, docs_signature: dict) -> bool:
    if force_rebuild:
        return True

    manifest = _read_manifest()

    if manifest != docs_signature:
        return True

    db_path = Path(CHROMA_PERSIST_DIRECTORY) / "chroma.sqlite3"

    return not db_path.exists()


def _delete_existing_index() -> None:
    index_path = Path(CHROMA_PERSIST_DIRECTORY)

    if index_path.exists():
        shutil.rmtree(index_path)


@st.cache_resource(show_spinner=False)
def inizializza_conoscenza(
    docs_signature: dict | None = None,
    force_rebuild: bool = False,
):
    """
    Inizializza o ricostruisce la knowledge base ChromaDB.
    """
    folder_path = DOCUMENTS_FOLDER

    if not os.path.exists(folder_path):
        return None, "⚠️ Cartella 'documenti' non trovata."

    pdf_files = _collect_pdf_files(folder_path)

    if not pdf_files:
        return None, "⚠️ Nessun PDF trovato nella cartella 'documenti'."

    docs_signature = docs_signature or calcola_firma_documenti(folder_path)
    rebuild = _should_rebuild(force_rebuild, docs_signature)

    if rebuild:
        _delete_existing_index()

    Path(CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)

    embeddings = _build_embeddings()
    chroma_settings = _build_chroma_settings()

    try:
        if rebuild:
            all_chunks = _load_and_split_documents(pdf_files)

            if not all_chunks:
                return None, "⚠️ Nessun contenuto valido estratto dai PDF."

            db = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
                client_settings=chroma_settings,
            )

            _write_manifest(docs_signature)

            return (
                db,
                f"✅ Knowledge base ricostruita. PDF letti: {len(pdf_files)}. Chunk creati: {len(all_chunks)}.",
            )

        db = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=chroma_settings,
        )

        existing = db.get()
        has_documents = bool(existing and existing.get("ids"))

        if not has_documents:
            inizializza_conoscenza.clear()
            return inizializza_conoscenza(
                docs_signature=docs_signature,
                force_rebuild=True,
            )

        return db, f"✅ Knowledge base caricata da disco. PDF disponibili: {len(pdf_files)}."

    except Exception as exc:
        logger.exception("Errore inizializzazione knowledge base")
        return None, f"⚠️ Errore inizializzazione knowledge base: {exc}"
