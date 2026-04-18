"""
Histogram of effective ranks across components.

Given a weight matrix M and a list of feature groups (each = list of input
indices), compute the effective rank of each group's submatrix and plot
the distribution.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np


def effective_rank(block: torch.Tensor) -> float:
    s2 = torch.linalg.svdvals(block).pow(2)
    return (s2.sum().pow(2) / s2.pow(2).sum()).item()


def plot_rank_histogram(M: torch.Tensor, groups: list[list[int]],
                        save_path: str = "rank_histogram.png",
                        bins: int = 30):
    eff_ranks = [effective_rank(M[g]) for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eff_ranks, bins=bins, color="#b23a1f", edgecolor="black")
    ax.set_xlabel("effective rank")
    ax.set_ylabel("number of components")
    ax.set_title(f"Effective rank distribution ({len(groups)} components)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    return eff_ranks


if __name__ == "__main__":
    torch.manual_seed(0)

    # Fake demo: 100 components, each is a 3-row block in a d=50 bottleneck.
    # Make some blocks low effective rank, some high, to show variety.
    n_components, k, d = 100, 3, 50
    M = torch.zeros(n_components * k, d)
    groups = []
    for i in range(n_components):
        # Random target effective rank between 1 and k
        target_rank = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        block = torch.randn(k, d)
        # Force the block to approximately that rank by zeroing tail singular values
        U, S, Vh = torch.linalg.svd(block, full_matrices=False)
        S[target_rank:] *= 0.05
        M[i*k:(i+1)*k] = U @ torch.diag(S) @ Vh
        groups.append(list(range(i*k, (i+1)*k)))

    ranks = plot_rank_histogram(M, groups, save_path="hist.png")
    print(f"Mean effective rank: {np.mean(ranks):.3f}")
    print(f"Min: {min(ranks):.3f}, Max: {max(ranks):.3f}")