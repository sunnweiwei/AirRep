from __future__ import annotations

from typing import Iterable, List

import numpy as np
from scipy.stats import spearmanr

from .model import AirRep
from .training import SubsetLoss


def evaluate_lds(model: AirRep, pairs: Iterable[SubsetLoss]) -> float:
    """Compute LDS score between predicted and true losses."""
    true = []
    pred = []
    for pair in pairs:
        emb = np.mean(model.encode(pair.subset), axis=0)
        # distance from origin as simple proxy
        pred.append(np.linalg.norm(emb))
        true.append(pair.loss)
    corr, _ = spearmanr(true, pred)
    return float(corr)
