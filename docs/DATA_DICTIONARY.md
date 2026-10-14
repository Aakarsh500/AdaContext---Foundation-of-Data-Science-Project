# Data dictionary

## Document

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable document identifier |
| `title` | string | Human-readable title |
| `sections` | list | Ordered sections or PDF pages |
| `metadata` | object | Dataset, year, domain, or source properties |

## Section

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Section heading or page label |
| `text` | string | Cleaned source text |
| `page` | integer/null | One-based PDF page when available |
| `metadata` | object | Additional source properties |

## Chunk

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Deterministic identifier derived from source and text |
| `document_id` | string | Parent document |
| `section` | string | Parent section |
| `chunk_index` | integer | Order within the document |
| `section_index` | integer | Order of the section |
| `text` | string | Evidence text sent to retrieval/generation |
| `token_count` | integer | Library-independent approximate token count |
| `page` | integer/null | Source page when available |

## Question record

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Question identifier |
| `document_id` | string | Document containing the answer |
| `question` | string | User or benchmark question |
| `answers` | list[string] | Acceptable gold answers |
| `evidence` | list[string] | Human-annotated supporting spans |
| `difficulty` | string/null | Simple, moderate, complex, or derived |
| `split` | string | Train, validation, or test |
| `metadata` | object | Dataset and answer-type information |

## Query log

Every line of `logs/query_log.jsonl` contains:

- Strategy and question.
- Final answer and citations.
- Candidate and selected chunk identifiers.
- BM25, dense, fusion, coverage, reranker, and selection scores.
- Predicted budget, confidence, source, and reason.
- Query feature vector.
- Selected and full-document token counts.
- Retrieval, selection, answer, and total latency.
- Embedding backend and timestamp.

