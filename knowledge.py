"""Conoscenza normativa strutturata e tracciabile (FASE 7).

Centralizza i valori normativi che erano **codificati e duplicati** nel codice e
nei prompt (soglie TOLC-I 9/16 per Informatica L-31; struttura della prova di
ammissione a Scienze dell'Educazione L-19). Diventano un'unica fonte di verità,
con **provenienza** (file e citazione verificata sulla fonte): così i dati sono
auditabili e non "inventati", e il classificatore e i template li leggono da qui.

I valori sono stati verificati sui PDF indicizzati (vedi `quote` in `SourceRef`).
Le funzioni che generano le tabelle riproducono esattamente l'output preesistente
(comportamento invariato), ma a partire dai dati anziché da stringhe fisse.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SourceRef:
    """Provenienza di un dato normativo: da quale documento proviene e cosa dice."""

    file: str
    page: Optional[int]
    quote: str


# ---------------------------------------------------------------------------
# Accesso a Informatica L-31 — soglie TOLC-I / Ris_Test
# ---------------------------------------------------------------------------
TOLC_INFORMATICA = {
    "ofa_min": 9,       # punteggio >= 9 e < 16  -> immatricolazione CON OFA
    "no_ofa_min": 16,   # punteggio >= 16        -> immatricolazione SENZA OFA
}

TOLC_INFORMATICA_SOURCE = SourceRef(
    file="regolamento-di-accesso-informatical-31-.pdf",
    page=0,
    quote=(
        "Possono immatricolarsi al Corso di Laurea in Informatica senza Obblighi "
        "Formativi Aggiuntivi (OFA) gli studenti che hanno partecipato al TOLC-I, "
        "ottenendo una votazione Ris_Test non inferiore a 16. Agli studenti che hanno "
        "ottenuto un Ris_Test inferiore a 16 e non inferiore a 9 [è consentita "
        "l'immatricolazione con OFA]."
    ),
)


def tolc_band_labels() -> dict:
    """Etichette delle fasce, derivate dalle soglie (es. '< 9', '>= 9 e < 16', '>= 16')."""
    ofa_min = TOLC_INFORMATICA["ofa_min"]
    no_ofa_min = TOLC_INFORMATICA["no_ofa_min"]
    return {
        "below": f"< {ofa_min}",
        "middle": f">= {ofa_min} e < {no_ofa_min}",
        "above": f">= {no_ofa_min}",
    }


def classify_tolc(score: float) -> dict:
    """Classifica il punteggio TOLC-I nelle tre fasce, leggendo le soglie dai dati."""
    labels = tolc_band_labels()

    if score < TOLC_INFORMATICA["ofa_min"]:
        return {
            "range": labels["below"],
            "short": (
                "No, l'immatricolazione diretta è sconsigliata o non prevista "
                "secondo le fonti disponibili."
            ),
            "consequence": "percorso di preparazione o recupero",
        }

    if score < TOLC_INFORMATICA["no_ofa_min"]:
        return {
            "range": labels["middle"],
            "short": "Sì, puoi immatricolarti, ma con OFA.",
            "consequence": "immatricolazione con OFA",
        }

    return {
        "range": labels["above"],
        "short": "Sì, puoi immatricolarti senza OFA.",
        "consequence": "immatricolazione senza OFA",
    }


def tolc_bands_table(col2_header: str, col3_header: str) -> str:
    """Tabella Markdown delle fasce TOLC, con intestazioni variabili. Output identico
    a quello prima codificato nei template (termina con doppio a-capo)."""
    labels = tolc_band_labels()
    return (
        f"| Punteggio Ris\\_Test | {col2_header} | {col3_header} |\n"
        "|---|---|---|\n"
        f"| {labels['below']} | Immatricolazione sconsigliata / non diretta | Percorso di preparazione o recupero |\n"
        f"| {labels['middle']} | Immatricolazione consentita | Con OFA |\n"
        f"| {labels['above']} | Immatricolazione consentita | Senza OFA |\n\n"
    )


# ---------------------------------------------------------------------------
# Prova di ammissione a Scienze dell'Educazione L-19
# ---------------------------------------------------------------------------
L19_ADMISSION = {
    "total_questions": 80,
    "duration": "2 ore e 30 minuti",
    "areas": [
        ("Cultura generale", 30),
        ("Conoscenze di base di lingua inglese", 10),
        ("Abilità logiche e analitiche", 20),
        ("Comprensione del testo", 20),
    ],
}

L19_ADMISSION_SOURCE = SourceRef(
    file="immatricolazione scienze dell'educazione-l-19.pdf",
    page=0,
    quote=(
        "La prova di ammissione al corso, della durata di 2 ore e 30 minuti, consiste "
        "nella soluzione di n. 80 quesiti a risposta multipla, su argomenti di: "
        "Cultura generale (30 quesiti); Conoscenze di base di lingua inglese (10 quesiti); "
        "Abilità logiche ed analitiche (20 quesiti); Comprensione del testo (20 quesiti)."
    ),
)


def l19_test_table_markdown() -> str:
    """Tabella Markdown della prova L-19, generata dai dati. Output identico a prima."""
    lines = ["| Area della prova | Quesiti |", "|---|---:|"]
    total = 0
    for area, count in L19_ADMISSION["areas"]:
        lines.append(f"| {area} | {count} |")
        total += count
    lines.append(f"| **Totale** | **{total}** |")
    return "\n".join(lines) + "\n"
