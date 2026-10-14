from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .bm25 import BM25Index
from .embeddings import BaseEmbedder, TfidfSvdEmbedder, create_embedder
from .rerank import create_reranker, minmax
from .schemas import Candidate, Chunk
from .text import lexical_coverage


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], config: dict) -> None:
        if not chunks:
            raise ValueError("Cannot build a retrieval index without chunks.")
        self.chunks = chunks
        self.config = config
        self.chunk_by_id = {chunk.id: chunk for chunk in chunks}
        self.index_by_chunk_id = {chunk.id: index for index, chunk in enumerate(chunks)}
        self.document_indices: dict[str, list[int]] = {}
        for index, chunk in enumerate(chunks):
            self.document_indices.setdefault(chunk.document_id, []).append(index)

        texts = [chunk.search_text for chunk in chunks]
        retrieval = config["retrieval"]
        backend = retrieval.get("embedding_backend", "auto")
        model_name = retrieval["embedding_model"]
        seed = int(config["project"].get("random_seed", 42))
        self.embedder: BaseEmbedder = create_embedder(backend, model_name, seed)
        try:
            self.embedder.fit(texts)
            self.embeddings = self.embedder.encode_documents(texts)
        except Exception:
            if isinstance(self.embedder, TfidfSvdEmbedder):
                raise
            self.embedder = TfidfSvdEmbedder(random_state=seed).fit(texts)
            self.embeddings = self.embedder.encode_documents(texts)
        self.embedding_backend = self.embedder.name
        self.bm25 = BM25Index().fit(texts)
        self.reranker = create_reranker(config)

    def allowed_indices(self, document_ids: Iterable[str] | None = None) -> np.ndarray:
        if document_ids is None:
            return np.arange(len(self.chunks), dtype=int)
        indices: list[int] = []
        for document_id in document_ids:
            indices.extend(self.document_indices.get(document_id, []))
        return np.asarray(sorted(set(indices)), dtype=int)

    def embedding_for_chunk(self, chunk_id: str) -> np.ndarray:
        return self.embeddings[self.index_by_chunk_id[chunk_id]]

    def retrieve(
        self,
        question: str,
        document_ids: Iterable[str] | None = None,
        candidate_pool_size: int | None = None,
    ) -> list[Candidate]:
        allowed = self.allowed_indices(document_ids)
        if allowed.size == 0:
            return []
        pool_size = int(candidate_pool_size or self.config["retrieval"]["candidate_pool_size"])
        query_embedding = self.embedder.encode_queries([question])[0]
        dense_raw = np.asarray(self.embeddings[allowed] @ query_embedding, dtype=np.float32)
        bm25_raw = self.bm25.scores(question)[allowed]

        dense_order = np.argsort(-dense_raw, kind="stable")
        bm25_order = np.argsort(-bm25_raw, kind="stable")
        dense_rank = np.empty(len(allowed), dtype=int)
        bm25_rank = np.empty(len(allowed), dtype=int)
        dense_rank[dense_order] = np.arange(1, len(allowed) + 1)
        bm25_rank[bm25_order] = np.arange(1, len(allowed) + 1)

        rrf_constant = int(self.config["retrieval"].get("rrf_constant", 60))
        rrf_raw = 1.0 / (rrf_constant + dense_rank) + 1.0 / (rrf_constant + bm25_rank)
        dense_normalized = minmax(dense_raw)
        bm25_normalized = minmax(bm25_raw)
        rrf_normalized = minmax(rrf_raw)

        preliminary = (
            0.45 * dense_normalized + 0.35 * bm25_normalized + 0.20 * rrf_normalized
        )
        selected_local = np.argsort(-preliminary, kind="stable")[: min(pool_size, len(allowed))]
        candidates: list[Candidate] = []
        for local_index in selected_local:
            global_index = int(allowed[local_index])
            chunk = self.chunks[global_index]
            candidates.append(
                Candidate(
                    chunk=chunk,
                    bm25_score=float(bm25_normalized[local_index]),
                    dense_score=float(dense_normalized[local_index]),
                    fusion_score=float(rrf_normalized[local_index]),
                    coverage_score=float(lexical_coverage(question, chunk.search_text)),
                    bm25_rank=int(bm25_rank[local_index]),
                    dense_rank=int(dense_rank[local_index]),
                )
            )

        candidates = self.reranker.rerank(question, candidates)
        rerank_limit = int(self.config["retrieval"]["rerank_pool_size"])
        return candidates[: min(rerank_limit, len(candidates))]

