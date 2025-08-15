from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class FactualVerifier:
    def __init__(self, model_name: str = "cross-encoder/stsb-roberta-base"):
        self.reranker = HuggingFaceCrossEncoder(model_name=model_name)

    def score(self, context: str, answer: str) -> float:
        """Returns a factuality score in [0..1] for a (context, answer) pair."""
        # Wandle das map-Objekt zuerst in eine Liste um, dann greife auf das erste Element zu
        scores = self.reranker.score([[context, answer]])
        return float(list(scores)[0])
