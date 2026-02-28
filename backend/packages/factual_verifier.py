from sentence_transformers import CrossEncoder as _CrossEncoder
import numpy as _np

class HuggingFaceCrossEncoder:
    def __init__(self, model_name: str, model_kwargs: dict = None):
        device = (model_kwargs or {}).get('device', 'cpu')
        self._encoder = _CrossEncoder(model_name, device=device)
    def score(self, sentence_pairs):
        return _np.array(self._encoder.predict(sentence_pairs))

class FactualVerifier:
    def __init__(self, model_name: str = "cross-encoder/stsb-roberta-base"):
        self.reranker = HuggingFaceCrossEncoder(model_name=model_name)

    def score(self, context: str, answer: str) -> float:
        """Returns a factuality score in [0..1] for a (context, answer) pair."""
        # Wandle das map-Objekt zuerst in eine Liste um, dann greife auf das erste Element zu
        scores = self.reranker.score([[context, answer]])
        return float(list(scores)[0])
