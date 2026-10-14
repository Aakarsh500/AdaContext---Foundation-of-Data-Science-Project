"""AdaContext: adaptive evidence selection for long-document QA."""

from .config import load_config
from .pipeline import AdaptiveContextPipeline
from .schemas import Document, QuestionRecord, Section

__all__ = [
    "AdaptiveContextPipeline",
    "Document",
    "QuestionRecord",
    "Section",
    "load_config",
]

__version__ = "1.0.0"

