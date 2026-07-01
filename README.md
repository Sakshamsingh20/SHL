"""
Hybrid retrieval over the SHL catalog.

Why hybrid: BM25 nails exact-token matches (product names like "OPQ32r", tech terms
like "AWS", "Docker") which matter a lot here since users reference specific catalog
items by name during Refine/Compare turns. TF-IDF + cosine similarity over word
1-2 grams gives a softer, more "semantic-ish" signal for paraphrased intent ("stakeholder
management" matching "communication" / "influencing" tests) without needing a downloaded
transformer model -- the catalog is short and keyword-dense enough that this is a
deliberate, justified trade-off over a heavier embedding model (see approach doc).
We fuse both rankings with reciprocal rank fusion (RRF), which avoids score-scaling
issues between the two methods.
"""
import os
import pickle
import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.catalog import Assessment, load_catalog

INDEX_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "index_cache.pkl")


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9+#.]+", text.lower())


class CatalogIndex:
    def __init__(self, assessments: list):
        self.assessments = assessments
        self.by_url = {a.url: a for a in assessments}
        self.by_name_lower = {a.name.lower(): a for a in assessments}

        texts = [a.searchable_text() for a in assessments]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def save(self, path: str = INDEX_CACHE):
        with open(path, "wb") as f:
            pickle.dump({"assessments": self.assessments}, f)

    def search(self, query: str, top_k: int = 10) -> list:
        """Returns list of (Assessment, score) sorted by fused relevance, best first."""
        if not query.strip():
            return []

        q_vec = self._vectorizer.transform([query])
        dense_scores = cosine_similarity(q_vec, self._tfidf_matrix)[0]
        dense_order = np.argsort(-dense_scores)

        q_tokens = _tokenize(query)
        bm25_scores = self._bm25.get_scores(q_tokens)
        bm25_order = np.argsort(-bm25_scores)

        # Reciprocal Rank Fusion
        k_rrf = 60
        fused = {}
        for rank, idx in enumerate(dense_order):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k_rrf + rank)
        for rank, idx in enumerate(bm25_order):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k_rrf + rank)

        ranked = sorted(fused.items(), key=lambda x: -x[1])
        return [(self.assessments[idx], score) for idx, score in ranked[:top_k]]

    def find_by_name(self, name: str) -> Optional[Assessment]:
        """Exact / near-exact name lookup, used for grounding Compare-turn claims."""
        name_l = name.lower().strip()
        if name_l in self.by_name_lower:
            return self.by_name_lower[name_l]
        # loose contains-match fallback
        for a in self.assessments:
            if name_l in a.name.lower() or a.name.lower() in name_l:
                return a
        return None


_index_singleton: Optional[CatalogIndex] = None


def get_index() -> CatalogIndex:
    global _index_singleton
    if _index_singleton is None:
        here = os.path.dirname(__file__)
        catalog_path = os.path.join(here, "..", "data", "shl_product_catalog.json")
        assessments = load_catalog(catalog_path)
        _index_singleton = CatalogIndex(assessments)
    return _index_singleton


if __name__ == "__main__":
    idx = get_index()
    for q in [
        "Java developer with stakeholder management",
        "OPQ32r",
        "entry level contact center spoken english",
        "personality test for sales leaders",
    ]:
        print(f"\nQuery: {q}")
        for a, score in idx.search(q, top_k=5):
            print(f"  {score:.4f}  {a.name}  ({a.test_type})")
