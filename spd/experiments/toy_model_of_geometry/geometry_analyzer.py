"""
Analysis utilities for SPD GeometryModel components.

This analyzes the learned SPD low-rank components directly:
    ΔW = A @ B

rather than the frozen base model weights.

The goal is to compare:
    - vanilla GeometryModel structure
vs
    - learned SPD component structure

in exactly the same analysis pipeline.

Assumptions
-----------
Checkpoint contains:
    components.linear1.A
    components.linear1.B
    components.linear2.A
    components.linear2.B

and each effective component matrix is reconstructed as:
    W = A @ B

If shapes do not match expected dimensions, flip multiplication order.
"""

from pathlib import Path
import einops
from einops import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch

def get_group_features(
    ranks: list[int],
) -> tuple[list[list[int]], list[int], int]:
    group_sizes = [k * (k + 1) for k in ranks]
    n_features = sum(group_sizes)

    groups = []
    start = 0
    for size in group_sizes:
        groups.append(list(range(start, start + size)))
        start += size

    return groups, group_sizes, n_features


# =============================================================================
# Utility
# =============================================================================

def entropy_effective_rank(singular_values: np.ndarray) -> float:
    probabilities = singular_values / singular_values.sum()
    probabilities = probabilities[probabilities > 1e-12]

    entropy = -(probabilities * np.log(probabilities)).sum()

    return float(np.exp(entropy))


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Analyses
# =============================================================================

def analyze_group_ranks(
    W1: np.ndarray,
    group_features: list[list[int]],
) -> None:

    print("\n" + "=" * 90)
    print("Per-Group Rank Analysis")
    print("=" * 90)

    sigma_max_global = np.linalg.svd(W1, compute_uv=False)[0]

    for g_idx, feature_indices in enumerate(group_features):

        W_g = W1[:, feature_indices]

        W_g_norm = W_g / (
            np.linalg.norm(W_g, axis=0, keepdims=True) + 1e-12
        )

        _, S, _ = np.linalg.svd(W_g, full_matrices=False)

        ev = (S ** 2) / (S ** 2).sum()
        cum = np.cumsum(ev)

        hard_rank = int((S > 1e-2 * sigma_max_global).sum())

        rank_99 = int((cum < 0.99).sum() + 1)

        entropy_rank = entropy_effective_rank(S)

        directional_rank = entropy_effective_rank(
            np.linalg.svd(W_g_norm, compute_uv=False)
        )

        print(f"group {g_idx}")
        print(f"  σ:                 {np.round(S, 4)}")
        print(f"  hard_rank:         {hard_rank}")
        print(f"  99%-variance rank: {rank_99}")
        print(f"  entropy_rank:      {entropy_rank:.3f}")
        print(f"  directional_rank:  {directional_rank:.3f}")


def plot_contribution_of_components_to_outputs(
    W,
    save_dir,
    title,
    group_size=6,
    threshold=0.05,
):
    """
    Visualize hidden-neuron contributions to grouped outputs.
    """

    W = np.asarray(W).T

    n_outputs, n_hidden = W.shape
    n_groups = n_outputs // group_size


    #sum over "out", meaning we calculate how much each group contributes to every hidden neuron, i.e what fraction each group contributes, but wihtout the fraction.
    W_grouped = reduce(
        W,
        "(group out) hidden -> group hidden",
        "sum",
        out=group_size,
    )


    #select the index of the group that contributes the most to each hidden neuron.
    dominant_group = np.argmax(W_grouped, axis=0)

    #select the value of the contribution of the dominant group for each hidden neuron.
    dominant_strength = W_grouped[dominant_group,np.arange(n_hidden),]

    #if the dominant feature is "weak", it doesnt do much for the computation.
    keep_mask = dominant_strength >= threshold

    # ------------------------------------------------------------
    # Sort neurons group-by-group
    # ------------------------------------------------------------
    neuron_groups = []

    for g in range(n_groups):
        #select, where each group is g and their strength is above the threshold.
        ids = np.flatnonzero((dominant_group == g) & keep_mask)

        #sort the neurosn in this group by their contribution strength to the group.
        order = np.argsort(W_grouped[g, ids])[::-1]
        neuron_groups.append(ids[order])
    #combine
    neuron_order = np.concatenate(neuron_groups)
    #Reorder columns
    W_sorted = W_grouped[:, neuron_order]



    fig, ax = plt.subplots(
        figsize=(16, 0.7 * n_groups + 1.5)
    )

    im = ax.imshow(
        W_sorted,
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=max(W_grouped.max(), 1e-12),
    )

    # Draw vertical lines seperating the neuron groups
    group_sizes = [len(g) for g in neuron_groups]
    boundaries = np.cumsum([0] + group_sizes)   
    for boundary in boundaries[1:-1]:
        ax.axvline(boundary - 0.5, color="black", lw=1)

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([
        f"out {g * group_size}–{(g + 1) * group_size - 1}"
        for g in range(n_groups)
    ])

    ax.set_xlabel("Hidden neurons")
    ax.set_ylabel("Output groups")
    ax.set_title(title)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    filename = title.replace(" ", "_").lower() + ".png"

    save_figure(fig, save_dir / filename)
    plt.close(fig)


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

    n_groups = len(group_features)
    d_hid, d_in = W1.shape
    n_out, _ = W2.shape

    # top: hidden activation per input group (group-indicator pushed through W1 + ReLU)
    A = np.zeros((n_groups, d_hid))
    for g, feat_idx in enumerate(group_features):
        A[g] = np.maximum(W1[:, feat_idx].sum(axis=1), 0.0)

    # bottom: positive parts of W2 (post-ReLU output paths)
    B = W2              #was np.maximum(W2, 0.0), but this seems nonsense to do?

    # neuron-level dominants and threshold
    dom_in  = np.argmax(A, axis=0)
    dom_out = np.argmax(B, axis=0)
    max_in  = A[dom_in,  np.arange(d_hid)]
    max_out = B[dom_out, np.arange(d_hid)]
    keep    = (max_in >= threshold) & (max_out >= threshold)

    # sort: by (dom_in, dom_out), then by combined magnitude inside each cell
    order, v_bounds, h_bounds = [], [0], [0]
    for g_in in range(n_groups):
        cell_start = len(order)
        for g_out in range(n_out):
            cols = np.where((dom_in == g_in) & (dom_out == g_out) & keep)[0]
            cols = cols[np.argsort(-(A[g_in, cols] + B[g_out, cols]))]
            order.extend(cols.tolist())
        v_bounds.append(len(order))                            # input-group dividers
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




def plot_group_output_matrix(
    W1,
    W2,
    group_features,
    save_dir,
    title=None,
    n_samples=50,
):
    """
    For each input group:
        1. sample a simplex distribution over its features
        2. run forward pass through linear ReLU network
        3. average output responses

    Result:
        P[g, out] = expected output when group g is activated
    """

    n_groups = len(group_features)
    n_out = W2.shape[0]
    d_in = W1.shape[1]

    P = np.zeros((n_groups, n_out))

    # ------------------------------------------------------------
    # group-wise forward simulation
    # ------------------------------------------------------------

    for g, feat_idx in enumerate(group_features):

        feat_idx = np.asarray(feat_idx)
        k = len(feat_idx)

        # construct n_samples random points on a k dim simplex
        simplex = np.random.dirichlet(
            np.ones(k),
            size=n_samples,
        )

        #Create an empty matrix, where we will fill in the simplices.
        X = np.zeros((n_samples, d_in))
        X[:, feat_idx] = simplex

        # forward pass
        H = np.maximum(X @ W1.T, 0.0)   
        Y = H @ W2.T 

        # average response
        P[g] = Y.mean(axis=0)


    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    im = ax.imshow(P, cmap="viridis", aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Output neuron")
    ax.set_ylabel("Input group")

    ax.set_xticks(range(n_out))
    ax.set_yticks(range(n_groups))

    ax.set_xticklabels([f"out{i}" for i in range(n_out)])
    ax.set_yticklabels([f"g{i}" for i in range(n_groups)])

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    save_figure(fig, save_dir / "group_output_matrix_simplex.png")
    plt.close(fig)



def main() -> None:
      
    RUN_DIR = Path(
        r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ranks = [2,2,2]

    state_dict = torch.load(
        RUN_DIR / "model_30000.pth",
        map_location=DEVICE,
    )

    A1 = state_dict["components.linear1.A"]
    B1 = state_dict["components.linear1.B"]

    A2 = state_dict["components.linear2.A"]
    B2 = state_dict["components.linear2.B"]
    W1 = einops.einsum(A1, B1, "d_in C K, C K d_out -> d_out d_in").detach().cpu().numpy()
    W2 = einops.einsum(A2, B2, "d_in C K, C K d_out -> d_out d_in").detach().cpu().numpy()
    

    group_features, _, _ = get_group_features(ranks)

    analyze_group_ranks(
        W1=W1,
        group_features=group_features,
    )

    #Plots that show the magnitude of the contribution of each component group to the hidden neurons or outputs
    plot_contribution_of_components_to_outputs(W1, save_dir=RUN_DIR, group_size=6, title="W1 contribution of input features to hidden neurons")
    plot_contribution_of_components_to_outputs(W2.T, save_dir=RUN_DIR, group_size=1, title="W2 contribution of hidden neurons to outputs")

    plot_group_output_matrix(W1, W2, group_features, save_dir=RUN_DIR, title="Group → output response (forward pass)")

    plot_input_hidden_output_translation(W1, W2, group_features, save_dir=RUN_DIR, title="Input → hidden → output translation")
    #Combined plot, that shows how the input features are routed to the outputs.

    plot_group_output_matrix(W1, W2, group_features, save_dir=RUN_DIR, title="Group → output response (forward pass)")
if __name__ == "__main__":
    main()  