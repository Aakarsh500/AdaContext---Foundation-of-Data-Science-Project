from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from typing import Sequence

import numpy as np

from .text import lexical_tokens


class BM25Index:
    """Small, dependency-free Okapi BM25 implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.doc_lengths = np.array([], dtype=np.float32)
        self.avg_doc_length = 0.0
        self.idf: dict[str, float] = {}

    def fit(self, documents: Sequence[str]) -> "BM25Index":
        self.term_frequencies = [Counter(lexical_tokens(document)) for document in documents]
        self.doc_lengths = np.asarray(
            [sum(frequencies.values()) for frequencies in self.term_frequencies], dtype=np.float32
        )
        self.avg_doc_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0
        document_frequency: defaultdict[str, int] = defaultdict(int)
        for frequencies in self.term_frequencies:
            for term in frequencies:
                document_frequency[term] += 1
        n_documents = len(self.term_frequencies)
        self.idf = {
            term: log(1.0 + (n_documents - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return self

    def scores(self, query: str) -> np.ndarray:
        if not self.term_frequencies:
            return np.array([], dtype=np.float32)
        query_terms = lexical_tokens(query)
        scores = np.zeros(len(self.term_frequencies), dtype=np.float32)
        average = max(self.avg_doc_length, 1.0)
        for index, frequencies in enumerate(self.term_frequencies):
            length_norm = 1.0 - self.b + self.b * float(self.doc_lengths[index]) / average
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * length_norm
                score += self.idf.get(term, 0.0) * numerator / denominator
            scores[index] = score
        return scores

