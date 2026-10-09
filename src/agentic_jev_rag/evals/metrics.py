"""Small, dependency-free metrics."""


def accuracy(pairs: list[tuple]) -> float:
    return sum(expected == predicted for expected, predicted in pairs) / len(pairs) if pairs else 0.0


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (as numpy's default)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
