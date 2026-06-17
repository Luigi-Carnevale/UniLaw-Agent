import ast
import operator
import re
from typing import Optional


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """
    Valuta in modo sicuro solo espressioni aritmetiche semplici.
    Evita l'uso diretto di eval().
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.Num):
        return node.n

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        operand = _safe_eval(node.operand)
        return _ALLOWED_OPERATORS[type(node.op)](operand)

    raise ValueError("Espressione non consentita.")


def _format_number(value: float | int) -> str:
    """
    Formatta i numeri in stile italiano.
    """
    if isinstance(value, float) and value.is_integer():
        value = int(value)

    return f"{value:,}".replace(",", "X").replace(".", ",").replace("X", ".")


def _normalizza_numero_italiano(value: str) -> float:
    """
    Converte stringhe numeriche in formato italiano (e misto) in float.

    Regole di disambiguazione del punto:
    - virgola e punto insieme  -> punto=migliaia, virgola=decimale (20.000,50 -> 20000.5)
    - solo virgola             -> separatore decimale (5,5 -> 5.5)
    - più punti                -> tutti separatori di migliaia (1.234.567 -> 1234567)
    - un solo punto seguito da esattamente 3 cifre -> separatore di migliaia (20.000 -> 20000)
    - altrimenti il punto è decimale (20.5 -> 20.5; 20000.50 -> 20000.5)
    """
    cleaned = value.strip().replace("€", "").replace(" ", "")

    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    elif cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    elif re.fullmatch(r"-?\d+\.\d{3}", cleaned):
        # Un singolo punto seguito da esattamente 3 cifre: separatore di migliaia
        # (es. "20.000"); senza questo ramo float() lo leggerebbe come decimale (20.0).
        cleaned = cleaned.replace(".", "")

    return float(cleaned)


def _estrai_calcolo_percentuale(text: str) -> Optional[str]:
    """
    Gestisce frasi del tipo:
    - Calcola il 5% di 20.000€
    - Quanto è il 20% di 1350?
    - 15% su 200
    """
    pattern = re.compile(
        r"(?P<percentuale>\d+(?:[\.,]\d+)?)\s*%\s*(?:di|su|\*)\s*(?P<base>\d+(?:[\.,]\d{3})*(?:[\.,]\d+)?)\s*€?",
        flags=re.IGNORECASE,
    )

    match = pattern.search(text)
    if not match:
        return None

    percentuale = _normalizza_numero_italiano(match.group("percentuale"))
    base = _normalizza_numero_italiano(match.group("base"))

    risultato = base * percentuale / 100

    return _format_number(risultato)


def calcola_espressione_sicura(espressione: str) -> str:
    """
    Calcola un'espressione aritmetica semplice.
    Sono ammessi solo numeri, parentesi e operatori aritmetici.
    """
    espressione = espressione.strip()

    pattern_sicuro = re.compile(r"^[0-9+\-*/%.() \t]+$")

    if not pattern_sicuro.fullmatch(espressione):
        return "Errore: l'espressione contiene caratteri non ammessi."

    try:
        parsed = ast.parse(espressione, mode="eval")
        risultato = _safe_eval(parsed.body)
        return _format_number(risultato)

    except Exception as exc:
        return f"Errore di calcolo: {exc}"


def prova_calcolo_sicuro(user_text: str) -> Optional[str]:
    """
    Attiva il calcolo solo in casi chiaramente matematici.
    Supporta:
    - espressioni aritmetiche pure: 2+2, 10/2, (5+3)*2
    - percentuali in linguaggio naturale: Calcola il 5% di 20.000€
    """
    text = user_text.strip()

    percentuale = _estrai_calcolo_percentuale(text)
    if percentuale is not None:
        return percentuale

    if re.fullmatch(r"[0-9+\-*/%.() \t]+", text):
        return calcola_espressione_sicura(text)

    return None