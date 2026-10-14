from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_NAMES, feature_vector
from .schemas import BudgetDecision


class BudgetController:
    def __init__(self, config: dict) -> None:
        adaptive = config["adaptive"]
        self.label_to_budget = {
            "simple": int(adaptive["simple_budget"]),
            "moderate": int(adaptive["moderate_budget"]),
            "complex": int(adaptive["complex_budget"]),
        }
        self.maximum_budget = int(adaptive["maximum_budget"])
        self.model_path = Path(adaptive["budget_model_path"])
        self.model = None
        self.feature_names = FEATURE_NAMES
        self.model_metadata: dict = {}
        if self.model_path.exists():
            try:
                artifact = joblib.load(self.model_path)
                self.model = artifact["model"]
                self.feature_names = list(artifact.get("feature_names", FEATURE_NAMES))
                self.label_to_budget.update(artifact.get("label_to_budget", {}))
                self.model_metadata = artifact.get("metadata", {})
            except Exception:
                self.model = None

    def _heuristic(self, features: dict[str, float]) -> BudgetDecision:
        points = 0.0
        points += 1.5 * features["comparison_flag"]
        points += 1.0 * features["multihop_flag"]
        points += 1.5 * features["broad_flag"]
        points += 1.0 * features["calculation_flag"]
        points += 1.0 if features["question_word_count"] >= 16 else 0.0
        points += 1.0 if features["score_entropy"] >= 0.82 else 0.0
        points += 1.0 if features["score_gap"] <= 0.08 else 0.0
        points += 1.0 if features["bm25_dense_agreement"] <= 0.25 else 0.0
        points += 0.5 if features["section_dispersion"] >= 0.8 else 0.0
        points -= 1.0 if features["top_term_coverage"] >= 0.8 and features["score_gap"] >= 0.15 else 0.0

        if points <= 1.0:
            label = "simple"
        elif points <= 4.0:
            label = "moderate"
        else:
            label = "complex"
        confidence = float(min(0.95, 0.55 + abs(points - 2.5) * 0.08))
        reason = (
            f"Rule score {points:.1f}; entropy={features['score_entropy']:.2f}, "
            f"top-gap={features['score_gap']:.2f}, rank-agreement={features['bm25_dense_agreement']:.2f}."
        )
        return BudgetDecision(
            label=label,
            budget=min(self.maximum_budget, self.label_to_budget[label]),
            confidence=confidence,
            source="interpretable heuristic",
            reason=reason,
        )

    def decide(self, features: dict[str, float]) -> BudgetDecision:
        if self.model is None:
            return self._heuristic(features)
        vector = feature_vector(features, self.feature_names)
        label = str(self.model.predict(vector)[0])
        if label not in self.label_to_budget:
            return self._heuristic(features)
        confidence = 0.5
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vector)[0]
            confidence = float(np.max(probabilities))
        reason = (
            f"Learned controller predicted '{label}' from retrieval uncertainty, query form, "
            "rank agreement, coverage, and evidence dispersion features."
        )
        return BudgetDecision(
            label=label,
            budget=min(self.maximum_budget, int(self.label_to_budget[label])),
            confidence=confidence,
            source="trained budget model",
            reason=reason,
        )

    def expand(self, decision: BudgetDecision) -> BudgetDecision:
        labels = ["simple", "moderate", "complex"]
        current_index = labels.index(decision.label) if decision.label in labels else 1
        next_label = labels[min(current_index + 1, len(labels) - 1)]
        return BudgetDecision(
            label=next_label,
            budget=min(self.maximum_budget, self.label_to_budget[next_label]),
            confidence=decision.confidence,
            source=decision.source,
            reason=f"{decision.reason} Context expanded because initial retrieval confidence was low.",
            expanded=True,
        )

