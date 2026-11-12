import numpy as np

def cut(Z: np.ndarray, labels: list[str], k: int) -> list[list[str]]:
    """Cut the dendrogram into k clusters by undoing the last k - 1 merges."""
    n = len(labels)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}")
    members = {i: [labels[i]] for i in range(n)}
    for step in range(n - k):
        a, b = int(Z[step, 0]), int(Z[step, 1])
        members[n + step] = members.pop(a) + members.pop(b)
    return list(members.values())


def plot_dendrogram(Z, labels: list[str], title: str = "", ax=None):
    """Complete-linkage dendrogram of a labeled distance matrix."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90)
    ax.set_ylabel("complete-linkage distance")
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax
