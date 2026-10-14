from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .rerank import minmax
from .schemas import Candidate


def diversity_aware_select(
    candidates: list[Candidate],
    budget: int,
    embedding_lookup: Callable[[str], np.ndarray],
    relevance_weight: float = 0.72,
    neighbour_bonus: float = 0.06,
    difficulty: str = "moderate",
) -> list[Candidate]:
    if not candidates or budget <= 0:
        return []
    budget = min(budget, len(candidates))
    relevance = minmax([candidate.final_score for candidate in candidates])
    top_raw = max(float(candidates[0].final_score), 1e-9)
    ratio_threshold = {"simple": 0.52, "moderate": 0.30, "complex": 0.16}.get(
        difficulty, 0.30
    )
    minimum = {"simple": 1, "moderate": min(2, budget), "complex": min(3, budget)}.get(
        difficulty, 1
    )

    selected_indices: list[int] = []
    while len(selected_indices) < budget:
        best_index: int | None = None
        best_utility = -float("inf")
        for index, candidate in enumerate(candidates):
            if index in selected_indices:
                continue
            if selected_indices:
                vector = embedding_lookup(candidate.chunk.id)
                similarities = [
                    float(np.dot(vector, embedding_lookup(candidates[chosen].chunk.id)))
                    for chosen in selected_indices
                ]
                novelty = 1.0 - max(similarities)
                new_section = float(
                    all(
                        candidate.chunk.section != candidates[chosen].chunk.section
                        or candidate.chunk.document_id != candidates[chosen].chunk.document_id
                        for chosen in selected_indices
                    )
                )
                adjacent = float(
                    any(
                        candidate.chunk.document_id == candidates[chosen].chunk.document_id
                        and abs(candidate.chunk.chunk_index - candidates[chosen].chunk.chunk_index) == 1
                        for chosen in selected_indices
                    )
                )
            else:
                novelty = 1.0
                new_section = 1.0
                adjacent = 0.0
            token_penalty = min(candidate.chunk.token_count / 1000.0, 0.25)
            utility = (
                relevance_weight * float(relevance[index])
                + (1.0 - relevance_weight) * novelty
                + 0.04 * new_section
                + neighbour_bonus * adjacent
                - 0.05 * token_penalty
            )
            if utility > best_utility:
                best_index = index
                best_utility = utility

        if best_index is None:
            break
        raw_ratio = float(candidates[best_index].final_score) / top_raw
        if len(selected_indices) >= minimum and raw_ratio < ratio_threshold:
            break
        candidates[best_index].selection_score = float(best_utility)
        selected_indices.append(best_index)

    selected = [candidates[index] for index in selected_indices]
    return sorted(
        selected,
        key=lambda candidate: (
            candidate.chunk.document_title,
            candidate.chunk.section_index,
            candidate.chunk.chunk_index,
        ),
    )


def elbow_select(candidates: list[Candidate], maximum: int = 10) -> list[Candidate]:
    if not candidates:
        return []
    limited = candidates[: min(maximum, len(candidates))]
    if len(limited) == 1:
        return limited
    scores = np.asarray([candidate.final_score for candidate in limited], dtype=float)
    drops = scores[:-1] - scores[1:]
    k = int(np.argmax(drops)) + 1
    k = max(1, min(k, maximum))
    return sorted(
        limited[:k],
        key=lambda candidate: (
            candidate.chunk.document_title,
            candidate.chunk.section_index,
            candidate.chunk.chunk_index,
        ),
    )
