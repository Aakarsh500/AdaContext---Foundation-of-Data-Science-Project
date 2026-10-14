from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Section:
    name: str
    text: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    id: str
    title: str
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            f"{section.name}\n{section.text}" if section.name else section.text
            for section in self.sections
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    document_title: str
    section: str
    text: str
    chunk_index: int
    section_index: int
    token_count: int
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        return f"{self.document_title}\n{self.section}\n{self.text}".strip()

    @property
    def citation(self) -> str:
        page = f", page {self.page}" if self.page is not None else ""
        return f"{self.document_title} — {self.section}{page} [{self.id}]"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QuestionRecord:
    id: str
    document_id: str
    question: str
    answers: list[str]
    evidence: list[str] = field(default_factory=list)
    difficulty: str | None = None
    split: str = "test"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Candidate:
    chunk: Chunk
    bm25_score: float = 0.0
    dense_score: float = 0.0
    fusion_score: float = 0.0
    coverage_score: float = 0.0
    rerank_score: float = 0.0
    bm25_rank: int | None = None
    dense_rank: int | None = None
    selection_score: float = 0.0

    @property
    def final_score(self) -> float:
        return self.rerank_score

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        output: dict[str, Any] = {
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "document_title": self.chunk.document_title,
            "section": self.chunk.section,
            "chunk_index": self.chunk.chunk_index,
            "token_count": self.chunk.token_count,
            "bm25_score": float(self.bm25_score),
            "dense_score": float(self.dense_score),
            "fusion_score": float(self.fusion_score),
            "coverage_score": float(self.coverage_score),
            "rerank_score": float(self.rerank_score),
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "selection_score": float(self.selection_score),
        }
        if include_text:
            output["text"] = self.chunk.text
        return output


@dataclass(slots=True)
class BudgetDecision:
    label: str
    budget: int
    confidence: float
    source: str
    reason: str
    expanded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerResponse:
    text: str
    provider: str
    model: str
    citations: list[str]
    input_tokens: int
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    question: str
    strategy: str
    answer: AnswerResponse
    selected: list[Candidate]
    candidates: list[Candidate]
    budget: BudgetDecision
    features: dict[str, float]
    retrieval_ms: float
    selection_ms: float
    total_ms: float
    selected_tokens: int
    full_document_tokens: int
    token_reduction: float
    embedding_backend: str
    context_truncated: bool = False

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        return {
            "question": self.question,
            "strategy": self.strategy,
            "answer": self.answer.to_dict(),
            "selected": [item.to_dict(include_text=include_text) for item in self.selected],
            "candidates": [item.to_dict(include_text=include_text) for item in self.candidates],
            "budget": self.budget.to_dict(),
            "features": {key: float(value) for key, value in self.features.items()},
            "retrieval_ms": float(self.retrieval_ms),
            "selection_ms": float(self.selection_ms),
            "total_ms": float(self.total_ms),
            "selected_tokens": int(self.selected_tokens),
            "full_document_tokens": int(self.full_document_tokens),
            "token_reduction": float(self.token_reduction),
            "embedding_backend": self.embedding_backend,
            "context_truncated": self.context_truncated,
        }

