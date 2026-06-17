"""Regole deterministiche sul punteggio TOLC-I (accesso a Informatica L-31).

`extract_tolc_score` è un parsing numerico sicuro (regola "accettabile").
`classify_tolc_score` resta l'API usata da template e test, ma in FASE 7 delega a
`knowledge.classify_tolc`: le soglie 9/16 non sono più codificate qui, bensì nel
layer di conoscenza normativa strutturata e tracciabile (`knowledge.py`).
"""

import re
from typing import Optional

from knowledge import classify_tolc


def extract_tolc_score(question: str) -> Optional[float]:
    q = question.lower()

    q = re.sub(r"\bl\s*[-]?\s*31\b", " ", q)
    q = re.sub(r"\bl31\b", " ", q)

    patterns = [
        r"\bpreso\s+(\d+(?:[,.]\d+)?)\b",
        r"\bpunteggio\s+(?:di\s+)?(\d+(?:[,.]\d+)?)\b",
        r"\bris[_\s-]?test\s+(?:di\s+)?(\d+(?:[,.]\d+)?)\b",
        r"\bho\s+ottenuto\s+(\d+(?:[,.]\d+)?)\b",
        r"\bcon\s+(\d+(?:[,.]\d+)?)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)

        if match:
            value = match.group(1).replace(",", ".")

            try:
                return float(value)

            except ValueError:
                return None

    candidates = re.findall(r"\b\d+(?:[,.]\d+)?\b", q)
    numeric = []

    for c in candidates:
        try:
            value = float(c.replace(",", "."))

        except ValueError:
            continue

        if 0 <= value <= 50:
            numeric.append(value)

    if len(numeric) == 1:
        return numeric[0]

    return None


def classify_tolc_score(score: float) -> dict[str, str]:
    # Soglie e fasce derivano ora dal layer di conoscenza strutturata (FASE 7).
    return classify_tolc(score)
