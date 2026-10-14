from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class BaseEmbedder(ABC):
    name: str

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> "BaseEmbedder":
        raise NotImplementedError

    @abstractmethod
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfSvdEmbedder(BaseEmbedder):
    """Offline embedding fallback using TF-IDF followed by latent semantic projection."""

    def __init__(self, dimensions: int = 256, random_state: int = 42) -> None:
        self.name = "tfidf-svd"
        self.dimensions = dimensions
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.995,
            sublinear_tf=True,
            max_features=50000,
        )
        self.reducer: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfSvdEmbedder":
        matrix = self.vectorizer.fit_transform(list(texts))
        upper = min(matrix.shape[0] - 1, matrix.shape[1] - 1, self.dimensions)
        if upper >= 2:
            self.reducer = TruncatedSVD(n_components=upper, random_state=self.random_state)
            self.reducer.fit(matrix)
        self._fitted = True
        return self

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TF-IDF embedder must be fitted before encoding.")
        sparse = self.vectorizer.transform(list(texts))
        if self.reducer is not None:
            dense = self.reducer.transform(sparse)
        else:
            dense = sparse.toarray()
        return normalize(np.asarray(dense, dtype=np.float32), norm="l2", copy=False)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.name = f"sentence-transformers:{model_name}"
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        output = self.model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(output, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


def create_embedder(backend: str, model_name: str, random_state: int = 42) -> BaseEmbedder:
    backend = backend.lower().strip()
    if backend in {"tfidf", "tfidf-svd", "offline"}:
        return TfidfSvdEmbedder(random_state=random_state)
    if backend in {"sentence_transformers", "sentence-transformers", "semantic"}:
        return SentenceTransformerEmbedder(model_name)
    if backend == "auto":
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception:
            return TfidfSvdEmbedder(random_state=random_state)
    raise ValueError(f"Unknown embedding backend: {backend}")

