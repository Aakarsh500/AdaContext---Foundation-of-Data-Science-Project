from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from .budget import BudgetController
from .chunking import chunker_from_config
from .config import ensure_output_directories
from .features import extract_query_features
from .generators import AnswerGenerator, create_generator
from .retrieval import HybridRetriever
from .schemas import BudgetDecision, Candidate, Document, PipelineResult
from .selection import diversity_aware_select, elbow_select


STRATEGIES = {
    "full_context",
    "bm25_fixed",
    "dense_fixed",
    "hybrid_fixed",
    "elbow",
    "adaptive",
}


class AdaptiveContextPipeline:
    def __init__(self, config: dict, generator: AnswerGenerator | None = None) -> None:
        self.config = config
        ensure_output_directories(config)
        self.chunker = chunker_from_config(config)
        self.generator = generator or create_generator(config)
        self.controller = BudgetController(config)
        self.documents: list[Document] = []
        self.chunks = []
        self.retriever: HybridRetriever | None = None

    def index_documents(self, documents: list[Document]) -> "AdaptiveContextPipeline":
        if not documents:
            raise ValueError("At least one document is required.")
        self.documents = documents
        self.chunks = self.chunker.chunk_documents(documents)
        if not self.chunks:
            raise ValueError("The supplied documents did not produce any text chunks.")
        self.retriever = HybridRetriever(self.chunks, self.config)
        return self

    @property
    def embedding_backend(self) -> str:
        return self.retriever.embedding_backend if self.retriever else "not-indexed"

    def _require_index(self) -> HybridRetriever:
        if self.retriever is None:
            raise RuntimeError("Call index_documents() before asking a question.")
        return self.retriever

    def retrieve_candidates(
        self,
        question: str,
        document_ids: Iterable[str] | None = None,
    ) -> list[Candidate]:
        return self._require_index().retrieve(question, document_ids=document_ids)

    def _all_context_candidates(self, document_ids: Iterable[str] | None) -> list[Candidate]:
        retriever = self._require_index()
        allowed = retriever.allowed_indices(document_ids)
        return [Candidate(chunk=self.chunks[int(index)], rerank_score=1.0) for index in allowed]

    def _fixed_decision(self, label: str, budget: int, reason: str) -> BudgetDecision:
        return BudgetDecision(
            label=label,
            budget=budget,
            confidence=1.0,
            source="evaluation baseline",
            reason=reason,
        )

    @staticmethod
    def _source_order(candidates: list[Candidate]) -> list[Candidate]:
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.chunk.document_title,
                candidate.chunk.section_index,
                candidate.chunk.chunk_index,
            ),
        )

    def _choose_context(
        self,
        strategy: str,
        candidates: list[Candidate],
        features: dict[str, float],
        document_ids: Iterable[str] | None,
        fixed_k: int,
    ) -> tuple[list[Candidate], BudgetDecision]:
        retriever = self._require_index()
        if strategy == "full_context":
            selected = self._all_context_candidates(document_ids)
            return selected, self._fixed_decision(
                "full", len(selected), "All document chunks were supplied."
            )
        if strategy == "bm25_fixed":
            selected = sorted(candidates, key=lambda item: item.bm25_score, reverse=True)[:fixed_k]
            return self._source_order(selected), self._fixed_decision(
                "fixed", fixed_k, f"Fixed top-{fixed_k} BM25 baseline."
            )
        if strategy == "dense_fixed":
            selected = sorted(candidates, key=lambda item: item.dense_score, reverse=True)[:fixed_k]
            return self._source_order(selected), self._fixed_decision(
                "fixed", fixed_k, f"Fixed top-{fixed_k} dense-retrieval baseline."
            )
        if strategy == "hybrid_fixed":
            return self._source_order(candidates[:fixed_k]), self._fixed_decision(
                "fixed", fixed_k, f"Fixed top-{fixed_k} hybrid baseline."
            )
        if strategy == "elbow":
            maximum = int(self.config["adaptive"]["maximum_budget"])
            selected = elbow_select(candidates, maximum=maximum)
            return selected, self._fixed_decision(
                "elbow", len(selected), "Stopped at the largest adjacent reranker-score drop."
            )

        decision = self.controller.decide(features)
        adaptive = self.config["adaptive"]
        low_confidence = (
            features["top_score"] < float(adaptive["low_confidence_threshold"])
            or (features["score_gap"] < 0.015 and features["score_entropy"] > 0.90)
        )
        if bool(adaptive.get("enable_confidence_expansion", True)) and low_confidence:
            decision = self.controller.expand(decision)
        selected = diversity_aware_select(
            candidates,
            decision.budget,
            retriever.embedding_for_chunk,
            relevance_weight=float(adaptive["mmr_relevance_weight"]),
            neighbour_bonus=float(adaptive["neighbour_bonus"]),
            difficulty=decision.label,
        )
        return selected, decision

    def _cap_for_generator(self, selected: list[Candidate]) -> tuple[list[Candidate], bool]:
        maximum = int(self.config["generation"].get("max_context_tokens", 0))
        if self.generator.provider == "extractive" or maximum <= 0:
            return selected, False
        capped: list[Candidate] = []
        used = 0
        for candidate in selected:
            if capped and used + candidate.chunk.token_count > maximum:
                break
            capped.append(candidate)
            used += candidate.chunk.token_count
        return capped, len(capped) < len(selected)

    def ask(
        self,
        question: str,
        strategy: str = "adaptive",
        document_ids: Iterable[str] | None = None,
        fixed_k: int | None = None,
        log_query: bool = True,
    ) -> PipelineResult:
        strategy = strategy.lower().strip()
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {sorted(STRATEGIES)}")
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        retriever = self._require_index()
        document_ids = list(document_ids) if document_ids is not None else None
        fixed_k = int(fixed_k or self.config["evaluation"]["fixed_k"])
        total_started = time.perf_counter()

        retrieval_started = time.perf_counter()
        candidates = retriever.retrieve(question, document_ids=document_ids)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        features = extract_query_features(question, candidates)

        selection_started = time.perf_counter()
        selected, decision = self._choose_context(
            strategy, candidates, features, document_ids, fixed_k
        )
        selected, truncated = self._cap_for_generator(selected)
        selection_ms = (time.perf_counter() - selection_started) * 1000

        answer = self.generator.generate(question, selected)
        selected_tokens = sum(candidate.chunk.token_count for candidate in selected)
        allowed = retriever.allowed_indices(document_ids)
        full_tokens = sum(self.chunks[int(index)].token_count for index in allowed)
        reduction = 1.0 - selected_tokens / full_tokens if full_tokens else 0.0
        result = PipelineResult(
            question=question,
            strategy=strategy,
            answer=answer,
            selected=selected,
            candidates=candidates,
            budget=decision,
            features=features,
            retrieval_ms=retrieval_ms,
            selection_ms=selection_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
            selected_tokens=selected_tokens,
            full_document_tokens=full_tokens,
            token_reduction=reduction,
            embedding_backend=retriever.embedding_backend,
            context_truncated=truncated,
        )
        if log_query:
            self._log_result(result)
        return result

    def _log_result(self, result: PipelineResult) -> None:
        log_path = Path(self.config["project"]["logs_dir"]) / "query_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = result.to_dict(include_text=False)
        record["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
