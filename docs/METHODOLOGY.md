# Methodology

## 1. Research objective

For a question \(q\), document chunks \(C\), and selected context \(S \subseteq C\), AdaContext studies the constrained objective:

\[
S^* = \arg\max_S \left(Q(q,S) - \lambda \frac{T(S)}{T(C)} - \mu L(S)\right)
\]

where \(Q\) is answer quality, \(T\) is context-token cost, and \(L\) is latency. Because answer quality is unknown before generation, the online system uses retrieval evidence and an offline-trained budget policy as proxies.

## 2. Preprocessing

Text is Unicode-normalized, dehyphenated across PDF line breaks, whitespace-normalized, and divided along section and sentence boundaries. Chunks target 220 approximate tokens with a 40-token overlap. Titles, section headings, pages, and source identifiers remain attached to every chunk.

## 3. Candidate retrieval

BM25 captures exact terminology and numbers. A Sentence Transformer captures semantic similarity; TF-IDF/SVD supplies a deterministic offline alternative. Dense and sparse rankings are combined using reciprocal rank fusion:

\[
RRF(c) = \frac{1}{k+r_d(c)} + \frac{1}{k+r_b(c)}
\]

where \(r_d\) and \(r_b\) are dense and BM25 ranks and \(k=60\). A lightweight feature reranker combines normalized BM25, dense, RRF, and lexical-coverage scores. An optional cross-encoder replaces most of this score with direct question–chunk relevance.

## 4. Adaptive features

The controller uses only information available at inference time:

- Question word, entity, and number counts.
- Comparison, multihop, broad, and calculation indicators.
- Top relevance score and top-two gap.
- Normalized entropy of candidate scores.
- Jaccard agreement between the top five BM25 and dense results.
- Query-term coverage in the first chunk.
- Dispersion of top candidates across sections.
- Candidate count and largest score-drop position.

## 5. Budget labels and model

Candidate budgets are 2, 5, and 8 chunks. For datasets without difficulty labels, gold evidence is mapped to chunks and the smallest rank depth that recovers it determines the class. Missing or widely dispersed evidence is labelled complex.

Multinomial logistic regression and a class-balanced random forest are compared using stratified cross-validated balanced accuracy. The better model is fitted to all permitted training records and serialized with its feature order and metadata. If no model is present, an interpretable scoring rule provides a fully functional fallback.

## 6. Diversity-aware selection

For candidate \(c\) and already selected set \(S\):

\[
U(c|S) = \alpha R(c) + (1-\alpha)(1-\max_{s\in S}\cos(c,s))
       + \gamma I_{new\ section} + \eta I_{adjacent} - \delta TokenCost(c)
\]

Candidates are added greedily until the predicted budget or a difficulty-specific marginal relevance threshold is reached. Low top-score confidence, or a small score gap combined with high entropy, expands the budget by one class.

## 7. Generation

The default deterministic generator extracts non-redundant sentences by TF-IDF question similarity and lexical coverage. It ensures the whole project runs without network access. Ollama and generic OpenAI-compatible chat endpoints use a grounded prompt that forbids unsupported knowledge and requires chunk citations.

## 8. Baselines

- Full document.
- BM25 fixed top-k.
- Dense fixed top-k.
- Hybrid fixed top-k.
- Largest-score-drop elbow.
- Proposed adaptive policy.

The hybrid fixed-k comparison isolates whether improvements arise from retrieval quality or from adaptation.

## 9. Metrics

Retrieval is evaluated by gold-evidence recall and context precision. Answers use exact match and maximum token-level F1 over accepted references. Efficiency uses selected chunks, selected tokens, token reduction, and component/total latency. Strategy means receive paired bootstrap 95% confidence intervals. The primary visual is a token-cost versus answer-quality Pareto chart.

## 10. Controlled stress test

Gold evidence is inserted at the start, middle, or end of synthetic contexts containing 10, 25, or 50 distractor sections. Full-context and adaptive methods receive the same questions. This separates evidence position and context length from question content.

