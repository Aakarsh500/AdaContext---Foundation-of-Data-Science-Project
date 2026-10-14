from __future__ import annotations

import hashlib

from .schemas import Chunk, Document
from .text import approximate_token_count, clean_text, split_sentences


class StructureAwareChunker:
    def __init__(
        self,
        target_tokens: int = 220,
        overlap_tokens: int = 40,
        min_chunk_tokens: int = 45,
        max_chunk_tokens: int = 330,
    ) -> None:
        if target_tokens <= 0 or max_chunk_tokens < target_tokens:
            raise ValueError("Chunk token limits are invalid.")
        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("Overlap must be non-negative and smaller than target size.")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens

    def _split_long_sentence(self, sentence: str) -> list[str]:
        words = sentence.split()
        if approximate_token_count(sentence) <= self.max_chunk_tokens:
            return [sentence]
        step = max(1, self.target_tokens - self.overlap_tokens)
        windows = []
        for start in range(0, len(words), step):
            window = " ".join(words[start : start + self.target_tokens]).strip()
            if window:
                windows.append(window)
            if start + self.target_tokens >= len(words):
                break
        return windows

    def _carry_overlap(self, sentences: list[str]) -> list[str]:
        if not sentences or self.overlap_tokens == 0:
            return []
        carried: list[str] = []
        count = 0
        for sentence in reversed(sentences):
            sentence_tokens = approximate_token_count(sentence)
            if carried and count + sentence_tokens > self.overlap_tokens:
                break
            carried.append(sentence)
            count += sentence_tokens
            if count >= self.overlap_tokens:
                break
        return list(reversed(carried))

    def _chunk_section(self, text: str) -> list[str]:
        sentences: list[str] = []
        for sentence in split_sentences(text):
            sentences.extend(self._split_long_sentence(sentence))
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = approximate_token_count(sentence)
            should_flush = (
                current
                and current_tokens >= self.min_chunk_tokens
                and current_tokens + sentence_tokens > self.target_tokens
            )
            if should_flush:
                chunks.append(clean_text(" ".join(current)))
                current = self._carry_overlap(current)
                current_tokens = approximate_token_count(" ".join(current))
            current.append(sentence)
            current_tokens += sentence_tokens
            if current_tokens >= self.max_chunk_tokens:
                chunks.append(clean_text(" ".join(current)))
                current = self._carry_overlap(current)
                current_tokens = approximate_token_count(" ".join(current))

        if current:
            tail = clean_text(" ".join(current))
            if chunks and approximate_token_count(tail) < self.min_chunk_tokens:
                merged = clean_text(f"{chunks[-1]} {tail}")
                if approximate_token_count(merged) <= self.max_chunk_tokens:
                    chunks[-1] = merged
                else:
                    chunks.append(tail)
            else:
                chunks.append(tail)
        return chunks

    def chunk_document(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0
        for section_index, section in enumerate(document.sections):
            for section_chunk_index, text in enumerate(self._chunk_section(section.text)):
                raw_id = f"{document.id}:{section_index}:{section_chunk_index}:{text[:80]}"
                digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:10]
                chunks.append(
                    Chunk(
                        id=f"{document.id}-c{chunk_index:04d}-{digest}",
                        document_id=document.id,
                        document_title=document.title,
                        section=section.name or "Document",
                        text=text,
                        chunk_index=chunk_index,
                        section_index=section_index,
                        token_count=approximate_token_count(text),
                        page=section.page,
                        metadata={
                            **section.metadata,
                            "section_chunk_index": section_chunk_index,
                        },
                    )
                )
                chunk_index += 1
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk_document(document)]


def chunker_from_config(config: dict) -> StructureAwareChunker:
    values = config["chunking"]
    return StructureAwareChunker(
        target_tokens=int(values["target_tokens"]),
        overlap_tokens=int(values["overlap_tokens"]),
        min_chunk_tokens=int(values["min_chunk_tokens"]),
        max_chunk_tokens=int(values["max_chunk_tokens"]),
    )
