from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .model import AirRep


def encode_texts(model: AirRep, texts: Iterable[str]) -> np.ndarray:
    return np.array(model.encode(list(texts)))


def influence_scores(train_embeds: np.ndarray, test_embeds: np.ndarray) -> np.ndarray:
    """Compute simple dot-product influence scores."""
    return test_embeds @ train_embeds.T
