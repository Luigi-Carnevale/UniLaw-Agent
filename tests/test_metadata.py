"""Test delle funzioni di metadata e firma del corpus (`database.py`).

Verifica la classificazione per corso e tipo documento (dedotta dal nome file)
e il calcolo della firma del corpus usata per decidere il rebuild dell'indice.
I nomi file usati sono quelli reali presenti in `documenti/`.
"""

import pytest

from database import (
    _infer_course_tag,
    _infer_doc_type,
    _should_rebuild,
    calcola_firma_documenti,
)


@pytest.mark.parametrize(
    "filename,expected_tag",
    [
        ("regolamento-l31-informatica.pdf", "informatica"),
        ("piano-di-studi-l31-informatica2025-2026.pdf", "informatica"),
        ("regolamento-l19-scienze-dell'educazione.pdf", "scienze_educazione"),
        ("immatricolazione scienze dell'educazione-l-19.pdf", "scienze_educazione"),
        ("regolamento-l16-scienze-dell'amministrazione-e-dell'organizzazione.pdf", "amministrazione"),
        ("guida-studente-scienze-economiche-e-statistiche.pdf", "economia"),
        ("documento-generico-senza-corso.pdf", "generale"),
    ],
)
def test_infer_course_tag(filename, expected_tag):
    assert _infer_course_tag(filename) == expected_tag


@pytest.mark.parametrize(
    "filename,expected_type",
    [
        ("regolamento-di-accesso-informatical-31-.pdf", "accesso"),
        ("immatricolazione scienze dell'educazione-l-19.pdf", "accesso"),
        ("Bando Borsa di studio 25-26.pdf", "borsa"),
        ("bando-erasmus25-26.pdf.pdf", "erasmus"),
        ("piano-di-studi-l31-informatica2025-2026.pdf", "piano_studi"),
        ("regolamento-tesi-2023.pdf", "tesi"),
        ("Regolamento-della–prova–finale-informatica-l31.pdf", "tesi"),
        ("regolamento-l31-informatica.pdf", "regolamento"),
        ("guida-studente-scienze-economiche-e-statistiche.pdf", "guida"),
    ],
)
def test_infer_doc_type(filename, expected_type):
    assert _infer_doc_type(filename) == expected_type


def test_signature_empty_folder(tmp_path):
    sig = calcola_firma_documenti(str(tmp_path))
    assert sig == {"documents": []}


def test_signature_nonexistent_folder():
    sig = calcola_firma_documenti("/path/che/non/esiste/xyz")
    assert sig == {"documents": []}


def test_signature_detects_file(tmp_path):
    pdf = tmp_path / "esempio.pdf"
    pdf.write_bytes(b"%PDF-1.4 contenuto di prova")
    sig = calcola_firma_documenti(str(tmp_path))
    assert len(sig["documents"]) == 1
    entry = sig["documents"][0]
    assert entry["filename"] == "esempio.pdf"
    assert "sha256" in entry and len(entry["sha256"]) == 64
    assert entry["size"] > 0


def test_signature_changes_with_content(tmp_path):
    pdf = tmp_path / "esempio.pdf"
    pdf.write_bytes(b"contenuto A")
    sig_a = calcola_firma_documenti(str(tmp_path))
    pdf.write_bytes(b"contenuto B diverso")
    sig_b = calcola_firma_documenti(str(tmp_path))
    assert sig_a != sig_b


def test_force_rebuild_always_true():
    # Con force_rebuild=True la funzione deve sempre richiedere il rebuild.
    assert _should_rebuild(True, {"documents": []}) is True
