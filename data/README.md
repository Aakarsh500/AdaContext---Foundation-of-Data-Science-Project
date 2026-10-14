# Data

## Bundled sample

`sample/documents.json` contains four fictional technical reports written for this project. `sample/questions.jsonl` contains 24 questions with answers, evidence spans, difficulty labels, and train/validation/test tags. Because the reports are fictional and internally generated, the sample can be distributed with the MIT-licensed code.

The sample is a software and workflow check, not a substitute for the final QASPER experiment.

## QASPER

Run `python scripts/download_qasper.py` to place the official auto-converted Parquet splits under `raw/qasper/`. The downloader reads the current Parquet manifest from the Hugging Face dataset repository instead of hard-coding temporary storage URLs.

QASPER contains 5,049 questions over 1,585 NLP papers with supporting-evidence annotations. It is licensed CC BY 4.0. See the [dataset card](https://huggingface.co/datasets/allenai/qasper) for attribution, limitations, and the complete schema.

Downloaded dataset files are intentionally excluded from source control.

