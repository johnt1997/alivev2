from pyserini.search.lucene import LuceneSearcher

class BM25Retriever:
    def __init__(self, index_dir: str):
        self.searcher = LuceneSearcher(index_dir)

    def get_scores(self, query: str, k: int) -> dict[str, float]:
        hits = self.searcher.search(query, k)
        out = {}
        for h in hits:
            doc = self.searcher.doc(h.docid)        # stored fields
            key = doc.get('id') if doc is not None else str(h.docid)
            out[key] = float(h.score)
        return out
