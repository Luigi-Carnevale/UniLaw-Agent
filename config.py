# Costanti, impostazioni visive e prompt. Qui puoi cambiare una regola del prompt o l'interfaccia grafica. 
import os
import logging
from langchain.prompts import PromptTemplate

# --- CONFIGURAZIONE AMBIENTE ---
def setup_environment():
    # Nascondo GPU e Telemetria
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    # Zittiamo i log di pypdf
    logging.getLogger("pypdf").setLevel(logging.ERROR)

# --- CSS E STILE ---
CSS_STYLES = """
<style>
    [data-testid="stStatusWidget"] { visibility: hidden; }
    .stChatMessage { border-radius: 15px; }
    .main-header { font-size: 3rem; color: #b71c1c; text-align: center; font-weight: 800; font-family: 'Helvetica Neue', sans-serif; margin-top: -20px; }
    .sub-header { font-size: 1.3rem; color: white; text-align: center; margin-bottom: 30px; font-style: italic; }
</style>
"""

# --- PROMPT TEMPLATES ---
qa_template = """
Sei UniLaw Agent, un assistente universitario che risponde SOLO in base ai documenti forniti.

REGOLE:
- Usa esclusivamente le informazioni nel CONTENUTO riportato sotto.
- Quando rispondi, NON aggiungere interpretazioni personali, rimani aderente al testo.
- Se le informazioni non sono presenti o non sono sufficienti per rispondere con sicurezza,
  rispondi esattamente: "Non lo so in base ai documenti disponibili."
- Non inventare regole, date, importi o procedure.
- Se la domanda è generica, limita la risposta a ciò che è chiaramente scritto nei documenti.

CONTENUTO DEI DOCUMENTI:
{context}

DOMANDA DELL’UTENTE:
{question}

Risposta (chiara, strutturata in punti, e aderente al testo dei documenti):
"""

QA_PROMPT = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"],
)

SYSTEM_MESSAGE = """
Sei UniLaw Agent, assistente dell'Università. 

REGOLE:
1. Usa SEMPRE lo strumento KnowledgeBase_Universitaria per rispondere a domande
   su regolamenti, piani di studio, tesi, borse di studio, esami, immatricolazioni, Erasmus, ecc. 
2. Rispondi SOLO in base alle informazioni presenti nei documenti. 
3. Se i documenti non contengono una risposta chiara, scrivi esattamente:
   "Non lo so in base ai documenti disponibili." 
4. Non inventare mai regole, cifre, date o procedure. 
5. Quando il tool ti restituisce la risposta con le fonti (file e pagina),
   non modificarla e non aggiungere riferimenti che non esistono nei PDF. 
"""