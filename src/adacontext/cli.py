from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PROJECT_ROOT, load_config
from .datasets import download_qasper, load_qasper, load_sample_dataset
from .eda import run_eda
from .evaluation import evaluate_strategies
from .generators import create_generator
from .pipeline import AdaptiveContextPipeline, STRATEGIES
from .stress import run_lost_in_middle_stress_test
from .training import train_budget_controller


def _runtime_config(args: argparse.Namespace) -> dict:
    overrides: dict = {"retrieval": {}, "generation": {}}
    if getattr(args, "embedding_backend", None):
        overrides["retrieval"]["embedding_backend"] = args.embedding_backend
    if getattr(args, "cross_encoder", False):
        overrides["retrieval"]["use_cross_encoder"] = True
    if getattr(args, "provider", None):
        overrides["generation"]["provider"] = args.provider
    overrides = {key: value for key, value in overrides.items() if value}
    return load_config(getattr(args, "config", None), overrides=overrides)


def _load_dataset(args: argparse.Namespace, config: dict):
    if args.dataset == "sample":
        documents, questions = load_sample_dataset(config["_project_root"])
    else:
        qasper_dir = Path(config["_project_root"]) / "data" / "raw" / "qasper"
        documents, questions = load_qasper(
            qasper_dir,
            splits=[args.split] if getattr(args, "split", None) else None,
            max_documents_per_split=getattr(args, "max_documents", None),
        )
    split = getattr(args, "split", None)
    if split:
        questions = [question for question in questions if question.split == split]
    limit = getattr(args, "limit_questions", None)
    if limit:
        questions = questions[:limit]
    relevant_ids = {question.document_id for question in questions}
    if relevant_ids:
        documents = [document for document in documents if document.id in relevant_ids]
    return documents, questions


def _add_common_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("sample", "qasper"), default="sample")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument(
        "--embedding-backend",
        choices=("auto", "sentence_transformers", "tfidf"),
        default="tfidf",
    )
    parser.add_argument("--cross-encoder", action="store_true")
    parser.add_argument("--config", type=Path)


def command_demo(args: argparse.Namespace) -> int:
    config = _runtime_config(args)
    documents, questions = _load_dataset(args, config)
    generator = create_generator(config, args.provider)
    pipeline = AdaptiveContextPipeline(config, generator=generator).index_documents(documents)
    question_text = args.question or (questions[0].question if questions else "What is this about?")
    document_id = args.document_id or (questions[0].document_id if questions else None)
    result = pipeline.ask(
        question_text,
        strategy=args.strategy,
        document_ids=[document_id] if document_id else None,
    )
    print(json.dumps(result.to_dict(include_text=True), indent=2, ensure_ascii=False))
    return 0


def command_download_qasper(args: argparse.Namespace) -> int:
    output = args.output or PROJECT_ROOT / "data" / "raw" / "qasper"
    paths = download_qasper(output, force=args.force)
    for path in paths:
        print(path.resolve())
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = _runtime_config(args)
    documents, questions = _load_dataset(args, config)
    pipeline = AdaptiveContextPipeline(config).index_documents(documents)
    output = args.output or Path(config["adaptive"]["budget_model_path"])
    training_csv = Path(config["project"]["reports_dir"]) / "budget_training_rows.csv"
    report = train_budget_controller(pipeline, questions, output, training_csv)
    print(json.dumps(report, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config = _runtime_config(args)
    documents, questions = _load_dataset(args, config)
    if not questions:
        raise ValueError("No questions matched the requested dataset split.")
    pipeline = AdaptiveContextPipeline(
        config, generator=create_generator(config, args.provider)
    ).index_documents(documents)
    output = args.output or Path(config["project"]["reports_dir"]) / "evaluation"
    _, summary = evaluate_strategies(pipeline, questions, args.strategies, output)
    print(summary.to_string(index=False))
    print(f"\nSaved evaluation artifacts to {Path(output).resolve()}")
    return 0


def command_eda(args: argparse.Namespace) -> int:
    config = _runtime_config(args)
    documents, questions = _load_dataset(args, config)
    pipeline = AdaptiveContextPipeline(config).index_documents(documents)
    output = args.output or Path(config["project"]["reports_dir"]) / "eda"
    summary = run_eda(pipeline, questions, output)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved EDA artifacts to {Path(output).resolve()}")
    return 0


def command_stress(args: argparse.Namespace) -> int:
    config = _runtime_config(args)
    documents, questions = _load_dataset(args, config)
    output = args.output or Path(config["project"]["reports_dir"]) / "stress_test"
    summary = run_lost_in_middle_stress_test(
        config,
        documents,
        questions,
        output,
        maximum_questions=args.maximum_questions,
    )
    print(summary.to_string(index=False))
    print(f"\nSaved stress-test artifacts to {Path(output).resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adacontext",
        description="Adaptive budget-aware context selection for long-document QA",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Ask one question and print the full trace")
    _add_common_dataset_arguments(demo)
    demo.add_argument("--question")
    demo.add_argument("--document-id")
    demo.add_argument("--strategy", choices=sorted(STRATEGIES), default="adaptive")
    demo.add_argument(
        "--provider", choices=("extractive", "ollama", "openai-compatible"), default="extractive"
    )
    demo.set_defaults(func=command_demo)

    downloader = subparsers.add_parser("download-qasper", help="Download official QASPER Parquet")
    downloader.add_argument("--output", type=Path)
    downloader.add_argument("--force", action="store_true")
    downloader.set_defaults(func=command_download_qasper)

    trainer = subparsers.add_parser("train-budget", help="Train the context-budget controller")
    _add_common_dataset_arguments(trainer)
    trainer.set_defaults(split="train")
    trainer.add_argument("--output", type=Path)
    trainer.set_defaults(func=command_train)

    evaluator = subparsers.add_parser("evaluate", help="Run all retrieval/selection baselines")
    _add_common_dataset_arguments(evaluator)
    evaluator.add_argument(
        "--provider", choices=("extractive", "ollama", "openai-compatible"), default="extractive"
    )
    evaluator.add_argument("--strategies", nargs="+", choices=sorted(STRATEGIES))
    evaluator.add_argument("--output", type=Path)
    evaluator.set_defaults(func=command_evaluate)

    eda = subparsers.add_parser("eda", help="Generate EDA tables and interactive charts")
    _add_common_dataset_arguments(eda)
    eda.set_defaults(split=None)
    eda.add_argument("--output", type=Path)
    eda.set_defaults(func=command_eda)

    stress = subparsers.add_parser("stress-test", help="Run controlled lost-in-the-middle tests")
    _add_common_dataset_arguments(stress)
    stress.set_defaults(split=None)
    stress.add_argument("--maximum-questions", type=int, default=6)
    stress.add_argument("--output", type=Path)
    stress.set_defaults(func=command_stress)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

