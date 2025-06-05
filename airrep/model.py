from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable

from sentence_transformers import SentenceTransformer

@dataclass
class AirRep:
    """Simple wrapper for an attribution-friendly embedding model."""

    model_name: str = "Alibaba-NLP/gte-small-en-v1.5"

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        """Encode a list of texts into embeddings."""
        return self.model.encode(list(texts), show_progress_bar=False)
