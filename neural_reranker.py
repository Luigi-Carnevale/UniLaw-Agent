"""Reranker neurale opzionale basato su cross-encoder multilingua (FASE 4).

Caratteristiche di progetto:
- **Opt-in**: disattivato di default (`config.RERANKER_ENABLED`).
- **Fallback automatico**: se `sentence-transformers` o il modello non sono
  disponibili (non scaricato, errore di caricamento), `rerank` restituisce i
  documenti nell'ordine ricevuto (= reranking euristico), senza sollevare errori.
- **Lazy loading**: il modello viene caricato solo al primo uso effettivo
  (nessun costo se il reranker resta disattivato).
- **Testabile offline**: il punteggiatore (`scorer`) è iniettabile, così i test
  non richiedono né il download del modello né `sentence-transformers`.

Il cross-encoder riceve coppie (domanda, testo del chunk) e produce un punteggio
di pertinenza; viene applicato solo ai primi `top_n` candidati già ordinati
dall'euristica e filtrati per corso, così da raffinare la selezione finale senza
perdere i priori di dominio.
"""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def rerank_by_scores(docs: list, scores: list) -> list:
    """Riordina i documenti per punteggio decrescente (ordinamento stabile).

    Funzione pura: testabile senza modello.
    """
    indexed = list(enumerate(docs))
    indexed.sort(key=lambda pair: scores[pair[0]], reverse=True)
    return [doc for _, doc in indexed]


class CrossEncoderReranker:
    """Wrapper attorno a un cross-encoder, con caricamento pigro e fallback."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
        scorer: Optional[Callable[[str, list], Optional[list]]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._scorer = scorer  # iniettabile nei test
        self._model = None
        self._load_failed = False

    def _ensure_model(self) -> None:
        if self._scorer is not None or self._model is not None or self._load_failed:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            logger.info("Reranker neurale caricato: %s", self.model_name)
        except Exception as exc:  # modello assente o errore di import/caricamento
            logger.warning(
                "Reranker neurale non disponibile (%s): uso il reranking euristico.",
                exc,
            )
            self._load_failed = True

    def available(self) -> bool:
        """True se è possibile produrre punteggi (scorer iniettato o modello caricabile)."""
        self._ensure_model()
        return self._scorer is not None or self._model is not None

    def score(self, question: str, docs: list) -> Optional[list]:
        """Punteggi di pertinenza per i documenti; None se il modello non è disponibile."""
        if self._scorer is not None:
            return self._scorer(question, docs)

        self._ensure_model()
        if self._model is None:
            return None

        pairs = [(question, doc.page_content) for doc in docs]
        return [float(s) for s in self._model.predict(pairs)]

    def rerank(self, question: str, docs: list, top_n: int) -> list:
        """Riordina i primi `top_n` documenti col cross-encoder; il resto invariato.

        In caso di fallback (modello non disponibile) restituisce `docs` invariati.
        """
        if not docs:
            return docs

        head = docs[:top_n]
        tail = docs[top_n:]

        scores = self.score(question, head)
        if scores is None:
            return docs  # fallback: ordine euristico preservato

        return rerank_by_scores(head, scores) + tail
