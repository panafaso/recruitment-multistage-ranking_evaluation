import numpy as np


def precision_at_k(binary_relevance: np.ndarray, k: int) -> float:
    top_k = binary_relevance[:k]
    return float(top_k.sum() / k)


def recall_at_k(
    binary_relevance: np.ndarray,
    total_relevant: int,
    k: int,
) -> float:
    if total_relevant == 0:
        return 0.0

    return float(
        binary_relevance[:k].sum()
        / total_relevant
    )


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0

    return float(
        2 * precision * recall
        / (precision + recall)
    )


def reciprocal_rank_at_k(
    binary_relevance: np.ndarray,
    k: int,
) -> float:
    relevant_positions = np.flatnonzero(
        binary_relevance[:k] == 1
    )

    if len(relevant_positions) == 0:
        return 0.0

    return float(
        1.0 / (relevant_positions[0] + 1)
    )


def average_precision_at_k(
    binary_relevance: np.ndarray,
    total_relevant: int,
    k: int,
) -> float:
    if total_relevant == 0:
        return 0.0

    top_k = binary_relevance[:k]
    precision_sum = 0.0
    relevant_seen = 0

    for rank, relevance in enumerate(
        top_k,
        start=1,
    ):
        if relevance == 1:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    denominator = min(total_relevant, k)

    return float(
        precision_sum / denominator
    )


def dcg_at_k(
    graded_relevance: np.ndarray,
    k: int,
) -> float:
    top_k = graded_relevance[:k]

    if len(top_k) == 0:
        return 0.0

    gains = np.power(2.0, top_k) - 1.0
    discounts = np.log2(
        np.arange(2, len(top_k) + 2)
    )

    return float(
        np.sum(gains / discounts)
    )


def ndcg_at_k(
    graded_relevance: np.ndarray,
    ideal_relevance: np.ndarray,
    k: int,
) -> float:
    actual_dcg = dcg_at_k(
        graded_relevance,
        k,
    )

    ideal_dcg = dcg_at_k(
        ideal_relevance,
        k,
    )

    if ideal_dcg == 0:
        return 0.0

    return float(
        actual_dcg / ideal_dcg
    )