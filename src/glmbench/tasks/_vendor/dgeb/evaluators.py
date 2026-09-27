"""DGEB evaluators — vendored from TattaBio/DGEB ``dgeb/evaluators.py``.

This is the metric/probe AUTHORITY for the DGEB tasks. The two evaluator classes
below — ``logRegClassificationEvaluator`` (EC DNA classification) and ``EDSEvaluator``
(Evolutionary Distance Similarity, DGEB's phylogeny tasks) — have their ``__init__``
and ``__call__`` copied **char-for-char** from upstream. The benchmarks ARE these
reductions (the sklearn ``LogisticRegression`` probe + ``f1_score(average="macro")`` for
EC; the paired cosine/manhattan/euclidean distances + ``scipy.stats.pearsonr`` →
``top_corr`` for EDS); we vendor and *call* them rather than re-derive.

The ONLY change from upstream is in the ``Evaluator`` base: upstream's ``__init__`` also
calls ``torch.manual_seed`` / ``torch.cuda.manual_seed_all``, which would pull ``torch``
into the torch-free core. Those two lines are removed here. This is
**inert** for both evaluators: the classification probe's randomness is governed entirely
by ``LogisticRegression(random_state=self.seed)``, and ``EDSEvaluator`` is fully
deterministic (paired distances + Pearson over fixed inputs — no RNG at all). The
``random``/``numpy`` seeding is kept verbatim.

We did NOT vendor upstream's other evaluators (clustering / retrieval / pair / bigene),
which import ``torch`` + ``pytrec_eval`` at module scope and would violate the
torch-free-core red line; this extracted subset is the minimal torch-free vendor that
lets us still *vendor-and-call* the exact probe/metric statements char-for-char.

Copy, never edit. ``SHA256SUMS`` pins this file; a unit test catches any silent drift.
Provenance + Apache-2.0 license: see ``NOTICE``.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
)
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

logger = logging.getLogger(__name__)


class Evaluator(ABC):
    """Base class for all evaluators.

    Extend this class and implement __call__ for custom evaluators.

    NOTE (gLMBench vendoring): upstream also seeds torch here
    (``torch.manual_seed`` / ``torch.cuda.manual_seed_all``); those two lines are
    removed to keep the core torch-free — inert for the classification probe, whose
    randomness is fixed by ``LogisticRegression(random_state=seed)``.
    """

    def __init__(self, seed: int = 42, **kwargs: object) -> None:
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    @abstractmethod
    def __call__(self) -> dict:
        """Run the evaluator and return a scores dict."""


class logRegClassificationEvaluator(Evaluator):  # noqa: N801 - upstream class name, kept verbatim
    def __init__(
        self,
        embeds_train,
        y_train,
        embeds_test,
        y_test,
        max_iter=1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embeds_train = embeds_train
        self.y_train = y_train
        self.embeds_test = embeds_test
        self.y_test = y_test
        self.max_iter = max_iter

    def __call__(self):
        scores = {}
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=-1,
            max_iter=self.max_iter,
            verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
        )
        logger.info(f"Encoding {len(self.embeds_train)} training embeds...")
        X_train = np.asarray(self.embeds_train)
        logger.info(f"Encoding {len(self.embeds_test)} test embeds...")
        X_test = np.asarray(self.embeds_test)
        logger.info("Fitting logistic regression classifier...")
        clf.fit(X_train, self.y_train)
        logger.info("Evaluating...")
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="macro")
        scores["accuracy"] = accuracy
        scores["f1"] = f1
        if len(np.unique(self.y_train)) == 2:
            ap = average_precision_score(self.y_test, y_pred)
            scores["ap"] = ap
        return scores


class EDSEvaluator(Evaluator):
    """
    Evolutionary Distance Similarity Evaluator, analogous to Semantic
    Textual Similarity Evaluator.
    Adapted from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/evaluation/evaluators/STSEvaluator.py
    """

    def __init__(self, embeds1, embeds2, gold_scores, **kwargs):
        super().__init__(**kwargs)
        self.embeds1 = embeds1
        self.embeds2 = embeds2
        self.gold_scores = gold_scores

    def __call__(self):
        embeddings1 = np.array(self.embeds1)
        embeddings2 = np.array(self.embeds2)
        logger.info("Evaluating...")
        cosine_scores = paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        cosine_pearson, _ = pearsonr(self.gold_scores, cosine_scores)
        manhattan_pearson, _ = pearsonr(self.gold_scores, manhattan_distances)
        euclidean_pearson, _ = pearsonr(self.gold_scores, euclidean_distances)

        top_corr = max(
            cosine_pearson,
            manhattan_pearson,
            euclidean_pearson,
        )
        return {
            "cos_sim": cosine_pearson,
            "manhattan": manhattan_pearson,
            "euclidean": euclidean_pearson,
            "top_corr": top_corr,
        }
