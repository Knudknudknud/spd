"""
Analysis utilities for a trained GeometryModel (per-group simplex-volume regression).

Pipeline:
1. Load trained GeometryModel
2. Volume-regression diagnostics (MAE / RMSE / correlation per group)
3. W_in / W_out structural analyses (per-group rank, subspace angles,
   neuron specialization)
4. End-to-end probes: one-hot feature response and canonical-simplex
   group response
5. Save plots to the run directory

Task setup
----------
Group g of rank k_g consumes (k_g + 1) vertices in R^{k_g}, flattened into
k_g · (k_g + 1) features. n_features = sum_g k_g·(k_g+1); n_outputs = len(ranks).
For ranks=[2,2,2] this is 18 features → 3 outputs (3 triangle volumes).

What "correct" structure looks like
-----------------------------------
  Input side  : hidden neurons partition into n_groups subsets;
                cosine-sim of feature embeddings is block-diagonal.
  Output side : each row of W_out reads from a disjoint subset of neurons.
  End-to-end  : a canonical simplex placed in group g drives output g and
                only output g; the group→output matrix is diagonal-dominant.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.experiments.toy_model_of_geometry.models import (
    GeometryModel,
    GeometryModelConfig,
)
from spd.experiments.toy_model_of_geometry.simplex_dataset import SimplexDataset


# =============================================================================
# Configuration
# =============================================================================

RUN_DIR = Path(
    r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\smaller_test"
)

CONFIG = GeometryModelConfig(
    ranks=[2, 2, 2],
    n_hidden=150,
    device="cuda" if torch.cuda.is_available() else "cpu",
    init_bias_to_zero=False,
    output_activation="relu",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Utility Functions
# =============================================================================

def get_group_features(
    ranks: list[int],
) -> tuple[list[list[int]], list[int], int]:
    """Build feature index groups for the simplex task."""
    group_sizes = [k * (k + 1) for k in ranks]
    n_features = sum(group_sizes)

    groups = []
    start = 0
    for size in group_sizes:
        groups.append(list(range(start, start + size)))
        start += size

    return groups, group_sizes, n_features


def build_feature_labels(ranks: list[int]) -> list[str]:
    """Labels of the form  g{group}_v{vertex}_d{dim}."""
    labels = []
    for g_idx, k in enumerate(ranks):
        for v_idx in range(k + 1):
            for d_idx in range(k):
                labels.append(f"g{g_idx}_v{v_idx}_d{d_idx}")
    return labels


def entropy_effective_rank(singular_values: np.ndarray) -> float:
    probabilities = singular_values / singular_values.sum()
    probabilities = probabilities[probabilities > 1e-12]
    entropy = -(probabilities * np.log(probabilities)).sum()
    return float(np.exp(entropy))


def canonical_simplex(k: int) -> np.ndarray:
    """Canonical unit simplex in R^k: origin + standard basis. Volume = 1/k!."""
    verts = np.zeros((k + 1, k))
    for i in range(k):
        verts[i + 1, i] = 1.0
    return verts


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_weights(model: GeometryModel) -> tuple[np.ndarray, np.ndarray]:
    """Return (W1, W2) as numpy arrays.  W1: (n_hidden, n_features), W2: (n_outputs, n_hidden)."""
    W1 = model.linear1.weight.detach().cpu().numpy()
    W2 = model.linear2.weight.detach().cpu().numpy()
    return W1, W2


def forward_numpy_preoutput(
    W1: np.ndarray,
    W2: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """W2 · ReLU(W1 · x), *without* the output ReLU.

    Using the pre-output-ReLU value keeps the routing signal visible even
    when one-hot inputs would be clipped to 0.
    """
    h = np.maximum(W1 @ x, 0.0)
    return W2 @ h


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    run_dir: Path,
    config: GeometryModelConfig,
    device: str,
) -> GeometryModel:
    state_dict = torch.load(run_dir / "geometry.pth", map_location=device)
    model = GeometryModel(config=config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded GeometryModel  ranks={config.ranks}  n_hidden={config.n_hidden}")
    return model


# =============================================================================
# Diagnostics
# =============================================================================

def evaluate_regression(
    model: GeometryModel,
    config: GeometryModelConfig,
    device: str,
    feature_probability: float = 1.0,
    batch_size: int = 20_000,
) -> None:
    """Per-group volume-regression quality, conditioned on whether the group is active."""
    dataset = SimplexDataset(
        dimensions=config.ranks,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="at_least_zero_active",
    )

    batch, labels = dataset.generate_batch(batch_size=batch_size)

    with torch.no_grad():
        preds = model(batch)

    print("=" * 90)
    print(f"Volume regression quality  (feature_probability = {feature_probability})")
    print("=" * 90)
    print(
        f"{'group':<6}{'rank':<6}"
        f"{'MAE':<10}{'RMSE':<10}{'true μ':<10}{'pred μ':<10}{'corr':<8}"
        f"{'|pred| | inactive':<20}"
    )

    for g_idx, k in enumerate(config.ranks):
        true_g = labels[:, g_idx]
        pred_g = preds[:, g_idx]

        active = true_g > 0
        n_active = int(active.sum())

        if n_active >= 2:
            t = true_g[active]
            p = pred_g[active]
            mae = (t - p).abs().mean().item()
            rmse = ((t - p) ** 2).mean().sqrt().item()
            corr = torch.corrcoef(torch.stack([t, p]))[0, 1].item()
            tm = t.mean().item()
            pm = p.mean().item()
        else:
            mae = rmse = corr = tm = pm = float("nan")

        inactive_pred = (
            pred_g[~active].abs().mean().item()
            if (~active).any() else float("nan")
        )

        print(
            f"{g_idx:<6}{k:<6}"
            f"{mae:<10.4f}{rmse:<10.4f}{tm:<10.4f}{pm:<10.4f}{corr:<8.3f}"
            f"{inactive_pred:<20.4f}"
        )

    overall_mae = (labels - preds).abs().mean().item()
    overall_rmse = ((labels - preds) ** 2).mean().sqrt().item()
    print(f"\nOverall MAE:  {overall_mae:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")


def analyze_group_ranks(
    W1: np.ndarray,
    group_features: list[list[int]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-group SVD of W1 submatrix."""
    print("\n" + "=" * 90)
    print("Per-Group Rank Analysis  (W1)")
    print("=" * 90)

    sigma_max_global = np.linalg.svd(W1, compute_uv=False)[0]
    group_bases = []
    group_singular_values = []

    for g_idx, feature_indices in enumerate(group_features):
        W_g = W1[:, feature_indices]
        W_g_norm = W_g / (np.linalg.norm(W_g, axis=0, keepdims=True) + 1e-12)

        U, S, _ = np.linalg.svd(W_g, full_matrices=False)
        group_bases.append(U)
        group_singular_values.append(S)

        ev = (S ** 2) / (S ** 2).sum()
        cum = np.cumsum(ev)

        hard_rank = int((S > 1e-2 * sigma_max_global).sum())
        rank_99 = int((cum < 0.99).sum() + 1)
        entropy_rank = entropy_effective_rank(S)
        directional_rank = entropy_effective_rank(
            np.linalg.svd(W_g_norm, compute_uv=False)
        )

        print(f"group {g_idx} ({len(feature_indices)} features):")
        print(f"  σ:                 {np.round(S, 4)}")
        print(f"  hard_rank:         {hard_rank}")
        print(f"  99%-variance rank: {rank_99}")
        print(f"  entropy_rank:      {entropy_rank:.3f}")
        print(f"  directional_rank:  {directional_rank:.3f}")

    return group_bases, group_singular_values




# =============================================================================
# Plotting Functions
# =============================================================================

def plot_feature_similarity(
    W1: np.ndarray,
    feature_labels: list[str],
    group_features: list[list[int]],
    save_dir: Path,
) -> None:
    """Cosine similarity between feature embeddings (columns of W1).
    Block structure within group boundaries indicates factorization."""
    W_normalized = W1 / (np.linalg.norm(W1, axis=0, keepdims=True) + 1e-12)
    sim = W_normalized.T @ W_normalized

    # boundaries between groups
    boundaries = []
    cum = 0
    for g in group_features[:-1]:
        cum += len(g)
        boundaries.append(cum - 0.5)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(feature_labels)))
    ax.set_xticklabels(feature_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels, fontsize=6)

    for b in boundaries:
        ax.axvline(b, color="black", lw=0.7)
        ax.axhline(b, color="black", lw=0.7)

    ax.set_title("Cosine Similarity of Feature Embeddings  (W1)")
    plt.colorbar(image, ax=ax)
    plt.tight_layout()
    save_figure(fig, save_dir / "feature_similarity.png")



def plot_input_hidden_output_translation(W1, W2, group_features, save_dir,
                                          title=None, threshold=0.05):
    """
    Two stacked heatmaps with the hidden axis shared:

      top    : hidden activation pattern for each input group
               (row g = relu(W1 @ x_g), with x_g the indicator of group g)
      bottom : W2 contribution of each hidden neuron to each output

    Neurons (columns) are sorted by their dominant input group, then by their
    dominant output within that input cluster. Reading a column top-to-bottom
    tells you "input group X activates this neuron, which routes to output Y".
    """
    W1, W2 = np.asarray(W1), np.asarray(W2)
    n_groups = len(group_features)
    d_hid, d_in = W1.shape
    n_out, _ = W2.shape

    # top: hidden activation per input group (group-indicator pushed through W1 + ReLU)
    A = np.zeros((n_groups, d_hid))
    for g, feat_idx in enumerate(group_features):
        A[g] = np.maximum(W1[:, feat_idx].sum(axis=1), 0.0)

    # bottom: positive parts of W2 (post-ReLU output paths)
    B = np.maximum(W2, 0.0)                                  # (n_out, d_hid)

    # neuron-level dominants and threshold
    dom_in  = np.argmax(A, axis=0)
    dom_out = np.argmax(B, axis=0)
    max_in  = A[dom_in,  np.arange(d_hid)]
    max_out = B[dom_out, np.arange(d_hid)]
    keep    = (max_in >= threshold) & (max_out >= threshold)

    # sort: by (dom_in, dom_out), then by combined magnitude inside each cell
    order, v_bounds = [], [0]
    for g_in in range(n_groups):
        cols = np.where((dom_in == g_in) & keep)[0]
        in_mag  = A[g_in, cols]
        out_mag = max_out[cols]                      # peak output mag per neuron
        cols = cols[np.lexsort((-out_mag, -in_mag))] # last key is primary in lexsort
        order.extend(cols.tolist())
        v_bounds.append(len(order))
    order = np.asarray(order, dtype=int)
    A_s, B_s = A[:, order], B[:, order]

    # plot
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 5),
        sharex=True, gridspec_kw=dict(height_ratios=[n_groups, n_out])
    )

    im_t = ax_top.imshow(A_s, aspect="auto", cmap="Blues",
                         vmin=0, vmax=max(A_s.max(), 1e-12))
    ax_top.set_yticks(range(n_groups))
    ax_top.set_yticklabels([f"in_g{g}" for g in range(n_groups)])
    ax_top.set_ylabel("input group")
    ax_top.set_title("relu(W1 · 1_g): hidden activation when input group g is fully on")
    plt.colorbar(im_t, ax=ax_top)

    im_b = ax_bot.imshow(B_s, aspect="auto", cmap="Reds",
                         vmin=0, vmax=max(B_s.max(), 1e-12))
    ax_bot.set_yticks(range(n_out))
    ax_bot.set_yticklabels([f"out{i}" for i in range(n_out)])
    ax_bot.set_ylabel("output")
    ax_bot.set_xlabel(f"neurons (sorted by dominant input → dominant output, "
                      f"kept {len(order)}/{d_hid})")
    ax_bot.set_title("relu(W2): hidden → output contribution")
    plt.colorbar(im_b, ax=ax_bot)

    for b in v_bounds[1:-1]:
        ax_top.axvline(b - 0.5, color="black", lw=1)
        ax_bot.axvline(b - 0.5, color="black", lw=1)

    fig.suptitle(title or "Input → hidden → output translation", y=1.02)
    plt.tight_layout()
    save_figure(fig, save_dir / "w1_w2_input_hidden_output_translation.png")
    plt.close(fig)




def plot_contribution_of_components_to_outputs(W, save_dir, title, group_size=6,
                                                threshold=0.05):
    W = np.asarray(W.T)
    n_out, n_hid = W.shape
    n_groups = n_out // group_size

    W_grouped = W.reshape(n_groups, group_size, n_hid).sum(axis=1)
    W_pos = W_grouped
    dominant = np.argmax(W_pos, axis=0)
    max_contrib = W_pos[dominant, np.arange(n_hid)]   # peak group contribution per neuron
    keep_mask = max_contrib >= threshold              # drop "dead" neurons
    order, boundaries = [], [0]
    for g in range(n_groups):
        cols = np.where((dominant == g) & keep_mask)[0]
        cols = cols[np.argsort(-W_pos[g, cols])]
        order.extend(cols.tolist())
        boundaries.append(len(order))
    order = np.asarray(order, dtype=int)
    W_sorted = W_pos[:, order]


    fig, ax = plt.subplots(figsize=(16, 0.7 * n_groups + 1.5))
    im = ax.imshow(W_sorted, aspect="auto", cmap="Reds",
                   vmin=0.0, vmax=max(W_pos.max(), 1e-12))

    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, color="black", lw=1)

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(
        [f"out{g*group_size}–{(g+1)*group_size-1}" for g in range(n_groups)]
    )
    ax.set_xlabel(f"neurons")
    ax.set_ylabel("output group")
    ax.set_title(title or f"W aggregated into {n_groups} output-groups ")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    title = title.replace(" ", "_").lower() + ".png"

    save_figure(fig, save_dir / title)
    plt.close(fig)


def plot_group_output_matrix(W1, W2, group_features, save_dir, title=None, n_samples=50):
    """
    For each group g:
        - sample x_g from a simplex over its features (Dirichlet)
        - run forward pass
        - average responses across samples

    P[g, k] = expected output k when group g is activated as a simplex distribution.
    """

    W1, W2 = np.asarray(W1), np.asarray(W2)
    n_groups = len(group_features)
    n_out = W2.shape[0]
    d_in = W1.shape[1]

    P = np.zeros((n_groups, n_out))

    for g, feat_idx in enumerate(group_features):
       feat_idx = np.array(feat_idx)
       k = len(feat_idx)
       simplex = np.random.dirichlet(np.ones(k), size=n_samples)   # (n_samples, k)
       X = np.zeros((n_samples, d_in))
       X[:, feat_idx] = simplex
       H = np.maximum(X @ W1.T, 0.0)                                # (n_samples, d_hid)
       Y = H @ W2.T                                                 # (n_samples, n_out)
       P[g] = Y.mean(axis=0)

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(P, cmap="viridis", aspect="auto")

    ax.set_title(title or "Group → output response (simplex sampled)")
    ax.set_xlabel("output")
    ax.set_ylabel("input group")

    ax.set_xticks(range(n_out))
    ax.set_yticks(range(n_groups))

    ax.set_xticklabels([f"out{i}" for i in range(n_out)])
    ax.set_yticklabels([f"g{i}" for i in range(n_groups)])

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    save_figure(fig, save_dir / "group_output_matrix_simplex.png")
    plt.close(fig)

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    config = CONFIG

    model = load_model(
        run_dir=RUN_DIR,
        config=config,
        device=DEVICE,
    )

    W1, W2 = get_weights(model)

    group_features, group_sizes, n_features = get_group_features(config.ranks)
    feature_labels = build_feature_labels(config.ranks)
    n_groups = len(config.ranks)

    # ----- Diagnostics -------------------------------------------------------
    evaluate_regression(
        model=model,
        config=config,
        device=DEVICE,
        feature_probability=1.0,
    )

    group_bases, group_singular_values = analyze_group_ranks(
        W1=W1,
        group_features=group_features,
    )



    # ----- Plots -------------------------------------------------------------
    plot_feature_similarity(
        W1=W1,
        feature_labels=feature_labels,
        group_features=group_features,
        save_dir=RUN_DIR,
    )



    plot_contribution_of_components_to_outputs(
        W=W2.T,
        save_dir=RUN_DIR,
        title="Contribution of hidden neurons to output groups",
        group_size=1,  # W2 is already per-output-group
        threshold=0.05,
    )

    plot_contribution_of_components_to_outputs(
        W=W1,
        save_dir=RUN_DIR,
        title="Contribution of hidden neurons to input groups",
        group_size=group_sizes[0],  # aggregate W1 contributions by input group
        threshold=0.05,
    )

    plot_group_output_matrix(
        W1=W1,
        W2=W2,
        group_features=group_features,
        save_dir=RUN_DIR,
        title="Group → output response (simplex sampled)"
    )
    plot_input_hidden_output_translation(
        W1=W1,
        W2=W2,
        group_features=group_features,
        save_dir=RUN_DIR,
        title="Input → hidden → output translation"
    )
    print(f"\nAll plots saved to: {RUN_DIR}")


if __name__ == "__main__":
    main()