# AdaContext

**Budget-aware hybrid evidence selection for long-document question answering**

AdaContext is a complete Foundation of Data Science project that studies whether a question-specific context budget can preserve answer quality while reducing tokens and latency. It is more than a basic RAG demo: it includes hybrid retrieval, reranking, feature engineering, a learned budget controller, diversity-aware evidence selection, baselines, ablations, statistical evaluation, EDA, a controlled lost-in-the-middle test, monitoring, and a Streamlit application.

The project runs on CPU. Its bundled benchmark and extractive answerer work without an API key or model download.

## What is included

- PDF, text, Markdown, HTML, sample JSON, and QASPER ingestion.
- Structure-aware chunking with overlap and source metadata.
- Dependency-free BM25 lexical retrieval.
- Sentence Transformer dense retrieval with automatic offline TF-IDF/SVD fallback.
- Reciprocal-rank fusion and optional cross-encoder reranking.
- Query/retrieval feature extraction.
- Trainable logistic-regression or random-forest budget controller.
- Simple, moderate, and complex context budgets.
- Token-aware, diversity-aware chunk selection with confidence expansion.
- Extractive, Ollama, and OpenAI-compatible answer backends.
- Full-context, BM25, dense, hybrid, elbow, and adaptive baselines.
- Evidence recall, context precision, EM, token F1, token use, and latency metrics.
- Bootstrap confidence intervals and a quality/cost Pareto chart.
- EDA tables and interactive Plotly charts.
- A controlled beginning/middle/end evidence-position stress test.
- Query-level JSONL monitoring logs.
- Automated tests and submission-ready project documentation.

## Fastest start on Windows

From PowerShell:

```powershell
cd C:\Users\Aakarsh\Desktop\AdaContext
.\setup.ps1
.\run_app.ps1
```

If PowerShell blocks local scripts, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

The initial screen uses the bundled 24-question benchmark and the offline TF-IDF/SVD backend. Click **Answer question** to run it immediately.

## Run without creating a virtual environment

The main dependencies are already usable if installed globally:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m adacontext.cli demo --dataset sample --embedding-backend tfidf
python -m streamlit run app.py
```

## Reproduce the bundled experiments

Run the complete local workflow:

```powershell
.\run_all.ps1
```

This command:

1. Trains the budget controller.
2. Generates EDA artifacts.
3. Evaluates all six context strategies.
4. Runs the lost-in-the-middle stress test.
5. Executes the automated test suite.

Generated outputs appear under `reports/generated/`.

## Run the research experiment on QASPER

Download the official dataset:

```powershell
python scripts\download_qasper.py
```

Start with a laptop-friendly subset:

```powershell
python scripts\train_budget_model.py --dataset qasper --split train --max-documents 200 --embedding-backend tfidf
python scripts\run_eda.py --dataset qasper --split validation --max-documents 50 --embedding-backend tfidf
python scripts\evaluate.py --dataset qasper --split test --max-documents 50 --limit-questions 200 --embedding-backend tfidf
```

For semantic embeddings, replace `tfidf` with `sentence_transformers`. The configured model is downloaded once and then cached locally. QASPER is distributed under CC BY 4.0; retain its attribution if redistributing the data.

## Use a local LLM

The extractive generator is intentionally zero-setup and deterministic. For a generative demo, start an Ollama server with any installed chat model, select **Ollama** in the app, and enter its model name. From the CLI:

```powershell
$env:OLLAMA_MODEL = "your-installed-model"
python -m adacontext.cli demo --dataset sample --provider ollama --embedding-backend tfidf
```

LM Studio, vLLM, Ollama's compatible endpoint, and other OpenAI-compatible servers can be selected with:

```powershell
$env:LLM_BASE_URL = "http://localhost:1234/v1"
$env:LLM_MODEL = "your-model"
$env:LLM_API_KEY = "not-needed-for-most-local-servers"
python -m adacontext.cli demo --dataset sample --provider openai-compatible --embedding-backend tfidf
```

Do not commit real API keys. `.env` is ignored.

## Main commands

```text
adacontext demo              Ask one question and print the complete trace
adacontext download-qasper   Download official QASPER Parquet files
adacontext train-budget      Train and save the context-budget classifier
adacontext evaluate          Evaluate retrieval and selection baselines
adacontext eda               Generate EDA tables and interactive charts
adacontext stress-test       Run evidence-position stress tests
```

Use `adacontext <command> --help` for all options.

## Architecture

```text
Documents -> structure-aware chunks -> BM25 index -----\
                                   -> dense index ------> rank fusion -> reranker
Question -> query/retrieval features -------------------> budget controller
reranked chunks + budget -> diversity/token selector -> grounded answer
low confidence ----------------------------------------> larger budget
answer + trace -> evaluation tables, charts, and JSONL logs
```

At inference time, the controller estimates a budget class from question form, relevance-score entropy, score gaps, BM25/dense agreement, term coverage, and section dispersion. The selector then maximizes relevance and novelty while penalizing unnecessary tokens.

## Project structure

```text
AdaContext/
├── app.py                         Streamlit demonstration and dashboard
├── config/default.yaml            Reproducible experiment configuration
├── data/sample/                   Bundled documents and 24 gold QA records
├── data/raw/qasper/               Downloaded QASPER Parquet files
├── docs/                          Report, methods, and data dictionary
├── notebooks/                     Reproducible EDA and experiment notebooks
├── reports/generated/             Tables and interactive plots
├── scripts/                       Convenient experiment entry points
├── src/adacontext/                Main Python package
└── tests/                         Offline automated tests
```

## Important output files

- `artifacts/budget_controller.joblib`: trained controller.
- `artifacts/budget_controller.metrics.json`: training and cross-validation diagnostics.
- `logs/query_log.jsonl`: selected chunks, scores, budgets, tokens, and latency.
- `reports/generated/eda/`: EDA CSV and HTML files.
- `reports/generated/evaluation/evaluation_details.csv`: question-level results.
- `reports/generated/evaluation/evaluation_summary.csv`: strategy comparison.
- `reports/generated/evaluation/quality_cost_pareto.html`: accuracy/cost frontier.
- `reports/generated/stress_test/`: lost-in-the-middle analysis.

## Methodological safeguards

- QASPER splits are document-based, preventing questions from the same paper from leaking across splits.
- Controller features never include the correct answer.
- Gold evidence is used to create training labels and evaluate retrieval, not at inference time.
- Thresholds and model choice should be finalized on training/validation data before reporting test results.
- Full question-level outputs are retained so aggregate scores can be audited.
- The report separates demonstration results from held-out research results.

## References

- Dasigi et al., [A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://arxiv.org/abs/2105.03011), 2021.
- Jeong et al., [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403), 2024.
- Bai et al., [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](https://arxiv.org/abs/2412.15204), 2024.
- [Sentence Transformers semantic-search documentation](https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html).

