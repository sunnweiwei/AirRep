from __future__ import annotations

from typing import Iterable, List, Union

import numpy as np
from scipy.stats import spearmanr

from .model import AirRep
from .training import SubsetLoss, AirRepModel


def evaluate_lds(model: Union[AirRepModel, AirRep], pairs: Iterable[SubsetLoss]) -> float:
    """Compute LDS score between predicted and true losses."""
    true = []
    pred = []
    for pair in pairs:
        if isinstance(model, AirRepModel):
            pred.append(model.predict_loss(pair.subset))
        else:
            emb = np.mean(model.encode(pair.subset), axis=0)
            pred.append(np.linalg.norm(emb))
        true.append(pair.loss)
    corr, _ = spearmanr(true, pred)
    return float(corr)


class LDSEvaluator:
    """Helper for evaluating LDS correlation."""

    def __init__(self, pairs: Iterable[SubsetLoss]) -> None:
        self.pairs = list(pairs)

    def eval(self, model: Union[AirRepModel, AirRep]) -> float:
        return evaluate_lds(model, self.pairs)
