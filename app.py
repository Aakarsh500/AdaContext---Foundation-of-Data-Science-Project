from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adacontext.config import load_config  # noqa: E402
from adacontext.datasets import load_qasper, load_sample_dataset  # noqa: E402
from adacontext.eda import run_eda  # noqa: E402
from adacontext.evaluation import evaluate_strategies  # noqa: E402
from adacontext.generators import create_generator  # noqa: E402
from adacontext.ingest import document_from_bytes  # noqa: E402
from adacontext.pipeline import AdaptiveContextPipeline, STRATEGIES  # noqa: E402


st.set_page_config(
    page_title="AdaContext",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main-title {font-size: 2.5rem; font-weight: 750; letter-spacing: -0.04em; margin-bottom: 0;}
      .subtitle {color: #526b64; margin-top: .2rem; margin-bottom: 1.4rem;}
      .evidence-card {border-left: 4px solid #2E7D6B; padding: .8rem 1rem; margin: .6rem 0;
                      background: #f3f7f5; border-radius: .35rem;}
      .small-note {color: #60736e; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def build_runtime_config(
    embedding_label: str,
    use_cross_encoder: bool,
    provider: str,
    ollama_url: str,
    ollama_model: str,
    compatible_url: str,
    compatible_model: str,
) -> dict:
    backend = "tfidf" if embedding_label.startswith("Offline") else "sentence_transformers"
    return load_config(
        overrides={
            "retrieval": {
                "embedding_backend": backend,
                "use_cross_encoder": use_cross_encoder,
            },
            "generation": {
                "provider": provider,
                "ollama_base_url": ollama_url,
                "ollama_model": ollama_model,
                "compatible_base_url": compatible_url,
                "compatible_model": compatible_model,
            },
        }
    )


def load_source(source: str, uploads, qasper_split: str, qasper_limit: int):
    if source == "Bundled sample benchmark":
        return load_sample_dataset(ROOT)
    if source == "Official QASPER dataset":
        return load_qasper(
            ROOT / "data" / "raw" / "qasper",
            splits=[qasper_split],
            max_documents_per_split=qasper_limit,
        )
    documents = []
    for uploaded in uploads or []:
        documents.append(document_from_bytes(uploaded.getvalue(), uploaded.name))
    return documents, []


def pipeline_signature(source: str, uploads, backend: str, reranker: bool, provider: str, extra: str):
    upload_signature = tuple((item.name, item.size) for item in (uploads or []))
    return (source, upload_signature, backend, reranker, provider, extra)


def render_result(result) -> None:
    st.subheader("Answer")
    st.markdown(result.answer.text)
    if result.answer.citations:
        with st.expander("Source citations", expanded=True):
            for citation in result.answer.citations:
                st.markdown(f"- {citation}")

    metric_columns = st.columns(5)
    metric_columns[0].metric("Difficulty", result.budget.label.title())
    metric_columns[1].metric("Chunks", len(result.selected), delta=f"budget {result.budget.budget}")
    metric_columns[2].metric("Context tokens", f"{result.selected_tokens:,}")
    metric_columns[3].metric("Token saving", f"{result.token_reduction:.1%}")
    metric_columns[4].metric("Total time", f"{result.total_ms:.0f} ms")

    evidence_tab, ranking_tab, decision_tab = st.tabs(
        ["Selected evidence", "Retrieval ranking", "Adaptive decision"]
    )
    with evidence_tab:
        for candidate in result.selected:
            st.markdown(
                f"<div class='evidence-card'><b>{candidate.chunk.document_title}</b><br>"
                f"{candidate.chunk.section} · {candidate.chunk.id} · "
                f"{candidate.chunk.token_count} tokens<br><br>{candidate.chunk.text}</div>",
                unsafe_allow_html=True,
            )
    with ranking_tab:
        selected_ids = {candidate.chunk.id for candidate in result.selected}
        ranking = pd.DataFrame(
            [
                {
                    "selected": candidate.chunk.id in selected_ids,
                    "chunk": candidate.chunk.id,
                    "section": candidate.chunk.section,
                    "hybrid/rerank": round(candidate.rerank_score, 4),
                    "dense": round(candidate.dense_score, 4),
                    "BM25": round(candidate.bm25_score, 4),
                    "coverage": round(candidate.coverage_score, 4),
                    "tokens": candidate.chunk.token_count,
                }
                for candidate in result.candidates
            ]
        )
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        if not ranking.empty:
            st.bar_chart(ranking.set_index("chunk")[["hybrid/rerank", "dense", "BM25"]])
    with decision_tab:
        st.info(result.budget.reason)
        st.write(
            f"Controller: **{result.budget.source}** · confidence: "
            f"**{result.budget.confidence:.1%}** · expanded: **{result.budget.expanded}**"
        )
        feature_frame = pd.DataFrame(
            [{"feature": key, "value": value} for key, value in result.features.items()]
        )
        st.dataframe(feature_frame, use_container_width=True, hide_index=True)
        st.caption(f"Embedding backend: {result.embedding_backend}")
        if result.context_truncated:
            st.warning("Context was capped to the configured generator context limit.")


st.markdown("<p class='main-title'>AdaContext</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Budget-aware hybrid evidence selection for long-document question answering</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("System setup")
    qasper_available = all(
        (ROOT / "data" / "raw" / "qasper" / f"{split}.parquet").exists()
        for split in ("train", "validation", "test")
    )
    source_options = ["Bundled sample benchmark", "Upload documents"]
    if qasper_available:
        source_options.append("Official QASPER dataset")
    source = st.selectbox("Document source", source_options)
    uploads = None
    qasper_split = "test"
    qasper_limit = 20
    if source == "Upload documents":
        uploads = st.file_uploader(
            "PDF, TXT, Markdown, or HTML",
            type=["pdf", "txt", "md", "markdown", "html", "htm"],
            accept_multiple_files=True,
        )
    elif source == "Official QASPER dataset":
        qasper_split = st.selectbox("QASPER split", ["test", "validation", "train"])
        qasper_limit = st.slider("Documents to index", 1, 100, 20)
    elif not qasper_available:
        st.caption("For QASPER: `python scripts/download_qasper.py`")

    embedding_label = st.selectbox(
        "Embedding backend",
        ["Offline TF-IDF/SVD (instant)", "Sentence Transformers (semantic, downloads once)"],
    )
    use_cross_encoder = st.toggle(
        "Cross-encoder reranking",
        value=False,
        help="Higher precision, but downloads another model and is slower on CPU.",
    )
    provider_display = st.selectbox(
        "Answer generator",
        ["Extractive (no setup)", "Ollama (local LLM)", "OpenAI-compatible endpoint"],
    )
    provider = {
        "Extractive (no setup)": "extractive",
        "Ollama (local LLM)": "ollama",
        "OpenAI-compatible endpoint": "openai-compatible",
    }[provider_display]
    ollama_url = "http://localhost:11434"
    ollama_model = "llama3.2:3b"
    compatible_url = "http://localhost:1234/v1"
    compatible_model = "local-model"
    if provider == "ollama":
        ollama_url = st.text_input("Ollama URL", os.getenv("OLLAMA_BASE_URL", ollama_url))
        ollama_model = st.text_input("Ollama model", os.getenv("OLLAMA_MODEL", ollama_model))
    elif provider == "openai-compatible":
        compatible_url = st.text_input("Endpoint", os.getenv("LLM_BASE_URL", compatible_url))
        compatible_model = st.text_input("Model", os.getenv("LLM_MODEL", compatible_model))
        api_key = st.text_input("API key", os.getenv("LLM_API_KEY", ""), type="password")
        if api_key:
            os.environ["LLM_API_KEY"] = api_key

    signature = pipeline_signature(
        source,
        uploads,
        embedding_label,
        use_cross_encoder,
        provider,
        f"{qasper_split}:{qasper_limit}:{ollama_model}:{compatible_model}",
    )
    rebuild = st.button("Build / refresh index", type="primary", use_container_width=True)


if rebuild or st.session_state.get("pipeline_signature") != signature:
    try:
        documents, questions = load_source(source, uploads, qasper_split, qasper_limit)
        if documents:
            config = build_runtime_config(
                embedding_label,
                use_cross_encoder,
                provider,
                ollama_url,
                ollama_model,
                compatible_url,
                compatible_model,
            )
            with st.spinner("Parsing, chunking, embedding, and indexing documents..."):
                pipeline = AdaptiveContextPipeline(
                    config,
                    generator=create_generator(config, provider),
                ).index_documents(documents)
            st.session_state.pipeline = pipeline
            st.session_state.documents = documents
            st.session_state.questions = questions
            st.session_state.pipeline_signature = signature
            st.session_state.pop("last_result", None)
        else:
            st.info("Upload at least one document, then build the index.")
    except Exception as error:
        st.error(f"Could not build the index: {error}")


pipeline = st.session_state.get("pipeline")
documents = st.session_state.get("documents", [])
questions = st.session_state.get("questions", [])

if pipeline is None:
    st.stop()

st.success(
    f"Indexed {len(documents)} document(s) into {len(pipeline.chunks)} chunks using "
    f"{pipeline.embedding_backend}."
)

ask_tab, compare_tab, experiment_tab, methodology_tab = st.tabs(
    ["Ask", "Compare strategies", "Experiments", "Methodology"]
)

with ask_tab:
    document_options = {"All indexed documents": None}
    document_options.update({document.title: document.id for document in documents})
    selected_title = st.selectbox("Search scope", list(document_options))
    selected_document_id = document_options[selected_title]

    sample_question = None
    if questions:
        matching = [
            question
            for question in questions
            if selected_document_id is None or question.document_id == selected_document_id
        ]
        question_options = ["Write my own question"] + [question.question for question in matching]
        chosen_question = st.selectbox("Benchmark question", question_options)
        if chosen_question != "Write my own question":
            sample_question = next(q for q in matching if q.question == chosen_question)
            selected_document_id = sample_question.document_id

    default_question = sample_question.question if sample_question else ""
    question_text = st.text_area(
        "Question",
        value=default_question,
        placeholder="Ask a question whose answer should be in the indexed documents...",
        height=90,
    )
    strategy_labels = {
        "Adaptive learned/heuristic budget": "adaptive",
        "Hybrid fixed top-k": "hybrid_fixed",
        "Dense fixed top-k": "dense_fixed",
        "BM25 fixed top-k": "bm25_fixed",
        "Largest score-drop elbow": "elbow",
        "Full document": "full_context",
    }
    strategy_label = st.selectbox("Context-selection strategy", list(strategy_labels))
    fixed_k = st.slider(
        "Fixed-k baseline size",
        1,
        10,
        int(pipeline.config["evaluation"]["fixed_k"]),
        disabled="fixed" not in strategy_labels[strategy_label],
    )
    if st.button("Answer question", type="primary", disabled=not question_text.strip()):
        try:
            with st.spinner("Retrieving evidence and generating an answer..."):
                result = pipeline.ask(
                    question_text,
                    strategy=strategy_labels[strategy_label],
                    document_ids=[selected_document_id] if selected_document_id else None,
                    fixed_k=fixed_k,
                )
            st.session_state.last_result = result
        except Exception as error:
            st.error(f"Question failed: {error}")
    if st.session_state.get("last_result") is not None:
        render_result(st.session_state.last_result)

with compare_tab:
    st.write("Run the same question through every context-selection baseline.")
    comparison_question = st.text_input(
        "Question to compare",
        value=st.session_state.last_result.question if st.session_state.get("last_result") else "",
    )
    if st.button("Run strategy comparison", disabled=not comparison_question.strip()):
        rows = []
        progress = st.progress(0.0)
        for index, strategy in enumerate(sorted(STRATEGIES)):
            result = pipeline.ask(comparison_question, strategy=strategy, log_query=False)
            rows.append(
                {
                    "strategy": strategy,
                    "chunks": len(result.selected),
                    "tokens": result.selected_tokens,
                    "saving": result.token_reduction,
                    "time_ms": result.total_ms,
                    "answer": result.answer.text,
                }
            )
            progress.progress((index + 1) / len(STRATEGIES))
        comparison = pd.DataFrame(rows)
        st.session_state.comparison = comparison
    if st.session_state.get("comparison") is not None:
        comparison = st.session_state.comparison
        st.dataframe(
            comparison.style.format({"saving": "{:.1%}", "time_ms": "{:.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(comparison.set_index("strategy")[["tokens"]])

with experiment_tab:
    st.subheader("Reproducible benchmark")
    if not questions:
        st.info("Benchmark evaluation is available for the bundled sample and QASPER sources.")
    else:
        available_splits = sorted({question.split for question in questions})
        evaluation_split = st.selectbox("Evaluation split", available_splits)
        evaluation_questions = [q for q in questions if q.split == evaluation_split]
        maximum = max(1, len(evaluation_questions))
        limit = st.slider("Questions", 1, maximum, min(maximum, 12))
        selected_strategies = st.multiselect(
            "Strategies",
            sorted(STRATEGIES),
            default=["full_context", "hybrid_fixed", "elbow", "adaptive"],
        )
        if st.button("Run evaluation", disabled=not selected_strategies):
            output = ROOT / "reports" / "generated" / "streamlit_evaluation"
            with st.spinner("Running paired evaluation..."):
                _, summary = evaluate_strategies(
                    pipeline,
                    evaluation_questions[:limit],
                    selected_strategies,
                    output,
                )
            st.session_state.evaluation_summary = summary
        if st.session_state.get("evaluation_summary") is not None:
            summary = st.session_state.evaluation_summary
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.scatter_chart(summary, x="median_selected_tokens", y="token_f1", color="strategy")
            st.download_button(
                "Download summary CSV",
                summary.to_csv(index=False),
                file_name="adacontext_evaluation_summary.csv",
                mime="text/csv",
            )

        st.subheader("Exploratory data analysis")
        if st.button("Generate EDA report"):
            output = ROOT / "reports" / "generated" / "streamlit_eda"
            with st.spinner("Computing chunk, evidence, and score-distribution analyses..."):
                summary = run_eda(pipeline, questions, output)
            st.json(summary)
            st.success(f"Interactive charts saved in {output}")

with methodology_tab:
    st.subheader("Research pipeline")
    st.markdown(
        """
1. Parse documents and preserve section/page metadata.
2. Build BM25 and dense-vector indexes over overlapping, structure-aware chunks.
3. Fuse lexical and semantic ranks, then optionally use a cross-encoder.
4. Extract query-complexity and retrieval-uncertainty features.
5. Predict a simple, moderate, or complex context budget.
6. Select relevant, non-redundant evidence using a token-aware MMR objective.
7. Expand the budget when retrieval confidence is low.
8. Answer with evidence citations and log cost, latency, and selected chunks.
9. Compare against full-context, fixed-k, and elbow baselines with paired evaluation.
        """
    )
    st.code(
        "utility = answer_quality - λ·token_ratio - μ·latency\n"
        "chunk_utility = relevance + novelty + coverage - token_cost",
        language="text",
    )
    log_path = ROOT / "logs" / "query_log.jsonl"
    if log_path.exists():
        st.download_button(
            "Download query log",
            log_path.read_bytes(),
            file_name="query_log.jsonl",
            mime="application/jsonl",
        )
