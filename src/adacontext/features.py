from __future__ import annotations

import math
import re

import numpy as np

from .schemas import Candidate
from .text import lexical_coverage, lexical_tokens


FEATURE_NAMES = [
    "question_word_count",
    "entity_count",
    "numeric_count",
    "comparison_flag",
    "multihop_flag",
    "broad_flag",
    "calculation_flag",
    "top_score",
    "score_gap",
    "score_entropy",
    "bm25_dense_agreement",
    "top_term_coverage",
    "section_dispersion",
    "candidate_count",
    "largest_drop_position",
]


COMPARISON_TERMS = {"compare", "difference", "versus", "vs", "both", "respectively", "change"}
MULTIHOP_TERMS = {"why", "relationship", "cause", "effect", "evidence", "explain", "how"}
BROAD_TERMS = {"summarize", "summary", "discuss", "limitations", "overall", "across", "main"}
CALCULATION_TERMS = {"calculate", "total", "sum", "difference", "percentage", "percent", "ratio"}


def _normalized_entropy(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    array = np.asarray(scores, dtype=np.float64)
    shifted = (array - array.max()) / 0.15
    probabilities = np.exp(shifted)
    probabilities /= max(probabilities.sum(), 1e-12)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return entropy / math.log(len(scores))


def extract_query_features(question: str, candidates: list[Candidate]) -> dict[str, float]:
    words = lexical_tokens(question)
    terms = set(words)
    capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9-]+\b", question)
    entity_count = max(0, len(capitalized) - int(bool(capitalized and question.startswith(capitalized[0]))))
    scores = [float(candidate.final_score) for candidate in candidates]
    top_score = scores[0] if scores else 0.0
    score_gap = top_score - scores[1] if len(scores) > 1 else top_score

    dense_top = {candidate.chunk.id for candidate in candidates if (candidate.dense_rank or 9999) <= 5}
    bm25_top = {candidate.chunk.id for candidate in candidates if (candidate.bm25_rank or 9999) <= 5}
    union = dense_top | bm25_top
    agreement = len(dense_top & bm25_top) / len(union) if union else 0.0
    top_five = candidates[:5]
    sections = {(item.chunk.document_id, item.chunk.section) for item in top_five}
    section_dispersion = len(sections) / len(top_five) if top_five else 0.0

    drops = [scores[index] - scores[index + 1] for index in range(len(scores) - 1)]
    largest_drop_position = float(int(np.argmax(drops)) + 1) if drops else 1.0
    top_coverage = lexical_coverage(question, candidates[0].chunk.search_text) if candidates else 0.0

    return {
        "question_word_count": float(len(words)),
        "entity_count": float(entity_count),
        "numeric_count": float(len(re.findall(r"\b\d+(?:\.\d+)?\b", question))),
        "comparison_flag": float(bool(terms & COMPARISON_TERMS)),
        "multihop_flag": float(bool(terms & MULTIHOP_TERMS)),
        "broad_flag": float(bool(terms & BROAD_TERMS)),
        "calculation_flag": float(bool(terms & CALCULATION_TERMS)),
        "top_score": float(top_score),
        "score_gap": float(score_gap),
        "score_entropy": float(_normalized_entropy(scores)),
        "bm25_dense_agreement": float(agreement),
        "top_term_coverage": float(top_coverage),
        "section_dispersion": float(section_dispersion),
        "candidate_count": float(len(candidates)),
        "largest_drop_position": largest_drop_position,
    }


def feature_vector(features: dict[str, float], names: list[str] | None = None) -> np.ndarray:
    ordered = names or FEATURE_NAMES
    return np.asarray([[float(features.get(name, 0.0)) for name in ordered]], dtype=np.float64)

