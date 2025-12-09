# Importa il modulo 're' (Regular Expressions). 
# Serve per definire regole di controllo sul testo (es. "accetta solo numeri").
import re

# Importa il decoratore 'tool' dalla libreria LangChain.
# Questo serve a trasformare una normale funzione Python in uno "strumento" che l'Intelligenza Artificiale può decidere di usare.
from langchain.tools import tool

# @tool è un "decoratore". Dice a LangChain: "Ehi, questa funzione qui sotto è un attrezzo che l'Agente può usare".
@tool
def calcolatrice_tasse(espressione: str):
    """
    Questa è la DOCSTRING (Documentazione). È importantissima perché l'AI la legge per capire QUANDO usare questo tool.
    Qui diciamo all'Agente: "Usa questo strumento SOLO per calcoli matematici precisi come tasse o percentuali".
    Input: Si aspetta una stringa di testo contenente un calcolo, esempio: '20000 * (5/100)'.
    """
    
    # .strip() rimuove eventuali spazi vuoti all'inizio e alla fine della stringa per pulirla.
    espressione = espressione.strip()

    # --- LIVELLO DI SICUREZZA 1: La Whitelist (Lista Bianca) ---
    # Definiamo una "Regular Expression" (Regex) che agisce come un filtro di sicurezza.
    # r'^[0-9+\-*/%.()\s]+$' significa: "Accetta SOLO numeri, operatori (+-*/%), punti, parentesi e spazi".
    # Se c'è una lettera o un comando strano, il pattern non corrisponderà.
    pattern_sicuro = re.compile(r'^[0-9+\-*/%.()\s]+$')
    
    # Controlliamo se l'input dell'utente rispetta il pattern sicuro.
    # Se 'match' restituisce False (cioè None), significa che c'è un carattere proibito.
    if not pattern_sicuro.match(espressione):
        # Blocchiamo tutto e restituiamo un errore. Questo impedisce all'AI di eseguire codice Python pericoloso.
        return "Errore: L'espressione contiene caratteri non ammessi per sicurezza."

    # Proviamo ad eseguire il calcolo. Usiamo 'try' per gestire eventuali errori matematici (es. divisione per zero).
    try:
        # --- LIVELLO DI SICUREZZA 2: Esecuzione Sandboxata ---
        # La funzione 'eval' esegue una stringa come se fosse codice Python. È potente ma pericolosa.
        # Il secondo argomento {"__builtins__": None} è la "Sandbox": stiamo dicendo a Python
        # di eseguire il calcolo SENZA avere accesso alle funzioni interne del sistema (come cancellare file o aprire internet).
        risultato = eval(espressione, {"__builtins__": None}, {})
        
        # Se il calcolo riesce, convertiamo il numero risultante in stringa (str) per poterlo restituire all'AI.
        return str(risultato)
        
    except Exception as e:
        # Se succede qualcosa di storto (es. sintassi errata, 10 diviso 0), catturiamo l'errore.
        # Restituiamo un messaggio leggibile che spiega cosa è andato storto.
        return f"Errore di calcolo: {e}"