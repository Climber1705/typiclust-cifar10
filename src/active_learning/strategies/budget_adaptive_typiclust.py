from typing import List

import numpy as np
import torch
from sklearn.preprocessing import normalize

from src.active_learning.clustering import cluster, cluster_count, uncovered_clusters
from src.active_learning.phase_transition import PhaseTransitionEstimator, compute_lambda
from src.active_learning.scoring import (
    cluster_disagreement_uncertainty,
    hybrid_score,
    model_uncertainty,
    typicality,
)
from src.active_learning.base import Strategy
from src.active_learning.state import State


class BudgetAdaptiveTypiClust(Strategy):

    def __init__(self, n_classes: int, k: int = 20, max_clusters: int = 500, steepness: float = 0.1, default_factor: float = 5.0) -> None:
        self.k = k
        self.max_clusters = max_clusters
        self.steepness = steepness
        self.estimator = PhaseTransitionEstimator(n_classes, default_factor)
        self.cumulative_budget: int = 0

    def query(self, state: State, budget: int, device: torch.device) -> List[int]:
        embeddings = normalize(state.embeddings, norm="l2")
        labeled_indices = np.array(sorted(state.labeled), dtype=int)
        unlabeled_indices = np.array(
            sorted(set(range(len(state.dataset))) - state.labeled), dtype=int
        )


        n_clusters = cluster_count(len(labeled_indices), budget, self.max_clusters)
        cluster_labels = cluster(embeddings, n_clusters)

        free = uncovered_clusters(cluster_labels, labeled_indices, n_clusters)
        if len(free) < budget:
            free = list(range(n_clusters))

        self.lambda_ = compute_lambda(self.cumulative_budget, self.estimator.m0, self.steepness)
        typicality_scores = typicality(embeddings, k=self.k)
        uncertainty_scores = self.compute_uncertainty(state, embeddings, cluster_labels, unlabeled_indices, device)
        scores = hybrid_score(typicality_scores, uncertainty_scores, self.lambda_)

        cluster_sizes = {
            c: int((cluster_labels[unlabeled_indices] == c).sum()) for c in free
        }
        top_clusters = sorted(free, key=lambda c: cluster_sizes[c], reverse=True)[:budget]

        queries: List[int] = []
        for cid in top_clusters:
            mask = cluster_labels[unlabeled_indices] == cid
            if not mask.any():
                continue
            candidates = unlabeled_indices[mask]
            queries.append(int(candidates[np.argmax(scores[candidates])]))

        if len(queries) < budget:
            selected = set(queries)
            extras = sorted(
                (i for i in unlabeled_indices if i not in selected),
                key=lambda i: scores[i],
                reverse=True,
            )
            queries.extend(extras[: budget - len(queries)])

        self.cumulative_budget += len(queries)
        return queries[:budget]

    def update_accuracy(self, validation_accuracy: float) -> None:
        self.estimator.update(self.cumulative_budget, validation_accuracy)

    @property
    def m0_estimate(self) -> float:
        return self.estimator.m0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"labeled={self.cumulative_budget}, "
            f"m0={self.m0_estimate:.1f}, "
            f"lambda={self.lambda_:.3f})"
        )

    def compute_uncertainty(
        self,
        state: State,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        unlabeled_indices: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        if state.model is not None:
            uncertainty_unlabeled = model_uncertainty(
                state.model, state.dataset, unlabeled_indices, device
            )
            uncertainty = np.zeros(len(embeddings), dtype=np.float32)
            uncertainty[unlabeled_indices] = uncertainty_unlabeled
            return uncertainty

        return cluster_disagreement_uncertainty(embeddings, cluster_labels, k=self.k)