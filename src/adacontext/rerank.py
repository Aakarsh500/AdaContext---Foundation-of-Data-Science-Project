from __future__ import annotations

from typing import Sequence

import numpy as np

from .schemas import Candidate


def minmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum - minimum < 1e-9:
        return np.ones_like(array) if maximum > 0 else np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


class LightweightReranker:
    name = "weighted-feature-reranker"

    def __init__(self, config: dict) -> None:
        retrieval = config["retrieval"]
        self.weights = {
            "bm25": float(retrieval["bm25_weight"]),
            "dense": float(retrieval["dense_weight"]),
            "fusion": float(retrieval["fusion_weight"]),
            "coverage": float(retrieval["lexical_coverage_weight"]),
        }

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        for candidate in candidates:
            candidate.rerank_score = (
                self.weights["bm25"] * candidate.bm25_score
                + self.weights["dense"] * candidate.dense_score
                + self.weights["fusion"] * candidate.fusion_score
                + self.weights["coverage"] * candidate.coverage_score
            )
        candidates.sort(key=lambda item: item.rerank_score, reverse=True)
        return candidates


class CrossEncoderReranker:
    def __init__(self, model_name: str, fallback: LightweightReranker) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.name = f"cross-encoder:{model_name}"
        self.model = CrossEncoder(model_name)
        self.fallback = fallback

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        base = self.fallback.rerank(query, candidates)
        base_scores = minmax([candidate.rerank_score for candidate in base])
        pairs = [(query, candidate.chunk.search_text) for candidate in base]
        cross_scores = minmax(self.model.predict(pairs, show_progress_bar=False))
        for candidate, base_score, cross_score in zip(base, base_scores, cross_scores):
            candidate.rerank_score = float(0.25 * base_score + 0.75 * cross_score)
        base.sort(key=lambda item: item.rerank_score, reverse=True)
        return base


def create_reranker(config: dict):
    fallback = LightweightReranker(config)
    if not bool(config["retrieval"].get("use_cross_encoder", False)):
        return fallback
    try:
        return CrossEncoderReranker(config["retrieval"]["cross_encoder_model"], fallback)
    except Exception:
        return fallback

