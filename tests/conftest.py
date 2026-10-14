from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adacontext.config import load_config  # noqa: E402
from adacontext.datasets import load_sample_dataset  # noqa: E402
from adacontext.pipeline import AdaptiveContextPipeline  # noqa: E402


@pytest.fixture(scope="session")
def sample_data():
    return load_sample_dataset(ROOT)


@pytest.fixture(scope="session")
def offline_config(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("adacontext")
    return load_config(
        overrides={
            "retrieval": {
                "embedding_backend": "tfidf",
                "use_cross_encoder": False,
            },
            "adaptive": {
                "budget_model_path": str(temporary / "missing-model.joblib"),
            },
            "project": {
                "logs_dir": str(temporary / "logs"),
                "reports_dir": str(temporary / "reports"),
                "artifacts_dir": str(temporary / "artifacts"),
            },
        }
    )


@pytest.fixture(scope="session")
def pipeline(sample_data, offline_config):
    documents, _ = sample_data
    return AdaptiveContextPipeline(offline_config).index_documents(documents)

