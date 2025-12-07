# Strumenti "extra" che l'agente può usare. 
import re
from langchain.tools import tool

@tool
def calcolatrice_tasse(espressione: str):
    """Utile SOLO quando devi fare calcoli matematici precisi (somme, percentuali, tasse).
    Input: una espressione matematica scritta come stringa (es: '20000 * (5/100)')."""
    
    espressione = espressione.strip()
    # Regex Whitelist per sicurezza
    pattern_sicuro = re.compile(r'^[0-9+\-*/%.()\s]+$')
    
    if not pattern_sicuro.match(espressione):
        return "Errore: L'espressione contiene caratteri non ammessi per sicurezza."

    try:
        # Esecuzione sandboxata
        risultato = eval(espressione, {"__builtins__": None}, {})
        return str(risultato)
    except Exception as e:
        return f"Errore di calcolo: {e}"