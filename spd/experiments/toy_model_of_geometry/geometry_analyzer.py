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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from image_combiner import combine_images
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

def entropy_effective_rank(singular_values: np.ndarray) -> float:
    probabilities = singular_values / singular_values.sum()
    probabilities = probabilities[probabilities > 1e-12]

    entropy = -(probabilities * np.log(probabilities)).sum()

    return float(np.exp(entropy))


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



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


def plot_input_hidden_output_translation(
    W1,
    W2,
    group_features,
    save_dir,
    title=None,
    threshold=0.05,
):
    """
    Visualize how input groups activate hidden neurons,
    and how those neurons route to outputs.
    """

    n_groups = len(group_features)
    n_hidden = W1.shape[0]
    n_outputs = W2.shape[0]

    # ------------------------------------------------------------
    # Hidden activation per input group
    # ------------------------------------------------------------
    # A[group, hidden]
    #
    # "If input group g is active,
    #  how much does hidden neuron h activate?"
    #
    A = np.zeros((n_groups, n_hidden))
    B = W2


    for g, feature_ids in enumerate(group_features):

        group_input = W1[:, feature_ids]

        # sum feature contributions
        hidden_activation = group_input.sum(axis=1)
        A[g, :] = np.maximum(hidden_activation, 0)

    #For every row, 
    dominant_input = np.argmax(A, axis=0)
    dominant_output = np.argmax(B, axis=0)

    input_strength = A.max(axis=0)
    output_strength = B.max(axis=0)


    keep = (input_strength >= threshold) & (output_strength >= threshold)


    neurons = np.arange(A.shape[1])

    #For each neuron, look up its strength of routing from its 
    routing_strength = A[dominant_input, neurons] * B[dominant_output, neurons]

    # Sort by (input group, output, -strength).
    # lexsort treats the LAST key as primary, so the order in the tuple matters.

    order = np.lexsort((dominant_output, -routing_strength, dominant_input))
    neuron_order = order[keep[order]]

    A_sorted = A[:, neuron_order]
    B_sorted = B[:, neuron_order]

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, (ax_strength, ax_top, ax_bottom) = plt.subplots(
        3, 1,
        figsize=(16, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, n_groups, n_outputs]},
    )

    #top plot
    strengths = routing_strength[neuron_order]
    x = np.arange(len(neuron_order))

    ax_strength.bar(x, strengths, width=1.0, color="0.4", edgecolor="none")
    ax_strength.set_ylabel("Routing\nstrength")
    ax_strength.set_xlim(-0.5, len(neuron_order) - 0.5)
    ax_strength.set_title("Routing strength (input dominant x output dominant)")

    # invisible spacer so this panel's width matches the heatmaps below
    spacer = make_axes_locatable(ax_strength).append_axes("right", size="2%", pad=0.1)
    spacer.axis("off")

    #As
    im_top = ax_top.imshow(A_sorted, aspect="auto", cmap="Blues")
    ax_top.set_ylabel("Input group")
    ax_top.set_title("Input group contribution to hidden neurons")
    ax_top.set_yticks(range(n_groups))
    ax_top.set_yticklabels([f"group {g}" for g in range(n_groups)])

    cax_top = make_axes_locatable(ax_top).append_axes("right", size="2%", pad=0.1)
    plt.colorbar(im_top, cax=cax_top)

    # B
    im_bottom = ax_bottom.imshow(B_sorted, aspect="auto", cmap="Reds")
    ax_bottom.set_ylabel("Output")
    ax_bottom.set_xlabel("Hidden neurons")
    ax_bottom.set_yticks(range(n_outputs))
    ax_bottom.set_yticklabels([f"out {i}" for i in range(n_outputs)])
    ax_bottom.set_title("Hidden neuron contribution to outputs")

    cax_bottom = make_axes_locatable(ax_bottom).append_axes("right", size="2%", pad=0.1)
    plt.colorbar(im_bottom, cax=cax_bottom)

    # ---------------- separators between input groups ----------------
    # Count how many kept neurons fall in each input group, in plot order
    sorted_dominant_input = dominant_input[neuron_order]
    group_sizes = np.bincount(sorted_dominant_input, minlength=n_groups)
    boundaries = np.cumsum(group_sizes)[:-1]   # drop the rightmost edge

    for boundary in boundaries:
        ax_strength.axvline(boundary - 0.5, color="black", lw=1)
        ax_top.axvline(boundary - 0.5, color="black", lw=1)
        ax_bottom.axvline(boundary - 0.5, color="black", lw=1)

    fig.suptitle(title)
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



# def plot_component_input_output_grids(C, group_features, save_dir, title=None,
#                                        threshold=0.05):
#     C = np.asarray(C)
#     n_comp, d_out, d_in = C.shape
#     n_groups = len(group_features)

#     grids = []
#     for feature_ids in group_features:
#         grid = np.zeros((n_comp, d_out))
#         for c in range(n_comp):
#             out_c = C[c][:, feature_ids].sum(axis=1)
#             grid[c] = np.maximum(out_c, 0.0)
#         grids.append(grid)

#     # keep only components whose peak contribution (across all groups) >= threshold
#     peaks = np.array([max(g[c].max() for g in grids) for c in range(n_comp)])
#     keep = np.where(peaks >= threshold)[0]
#     grids = [g[keep] for g in grids]
#     comp_labels = [f"C{c}" for c in keep]
#     n_kept = len(keep)


    

#     vmax = max(max(g.max() for g in grids), 1e-12)   # reverted: global max

#     # widen subplots when there are many output dims so all columns are visible
#     subplot_w = max(2.2, d_out * 0.03)
#     fig, axes = plt.subplots(
#         1, n_groups,
#         figsize=(subplot_w * n_groups + 1.5, 0.18 * n_kept + 2),
#         sharey=True,
#     )
#     if n_groups == 1:
#         axes = [axes]

#     # sparse numeric x-ticks for large d_out; explicit "out{i}" labels for small
#     if d_out <= 20:
#         xticks = list(range(d_out))
#         xticklabels = [f"out{o}" for o in range(d_out)]
#         xlabel = "output"
#     else:
#         step = max(1, d_out // 15)
#         xticks = list(range(0, d_out, step))
#         xticklabels = [str(o) for o in xticks]
#         xlabel = "hidden neuron"

#     for g, (ax, grid) in enumerate(zip(axes, grids)):
#         im = ax.imshow(grid, aspect="auto", cmap="Reds", vmin=0.0, vmax=vmax)
#         ax.set_title(f"input group {g}")
#         ax.set_xlabel(xlabel)
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xticklabels)
#     axes[0].set_ylabel("component")
#     axes[0].set_yticks(range(n_kept))
#     axes[0].set_yticklabels(comp_labels)

#     fig.colorbar(im, ax=axes, shrink=0.85, label="contribution")
#     fig.suptitle(title or f"Component contribution (group one-hot, kept {n_kept}/{n_comp})")
#     save_path = title.replace(" ", "_").lower() + ".png"
#     save_figure(fig, save_dir / save_path)
#     plt.close(fig)



import matplotlib.colors as mcolors


def plot_component_input_output_grids(C, group_features, save_dir, title=None,
                                       threshold=0.05, max_components=12, gamma=0.5):
    C = np.asarray(C)
    n_comp, d_out, d_in = C.shape
    n_groups = len(group_features)

    grids = [
        np.maximum(np.stack([C[c][:, fid].sum(axis=1) for c in range(n_comp)]), 0.0)
        for fid in group_features
    ]
    stacked = np.stack(grids)                              # (n_groups, n_comp, d_out)

       # 1) keep the strongest components by TOTAL magnitude (sorted, strongest first)
    comp_strength = stacked.sum(axis=(0, 2))
    keep_c = np.argsort(-comp_strength)[:max_components]
    keep_c = keep_c[comp_strength[keep_c] > 0]

    # 2) drop dead hidden neurons, but keep them in natural order (no sort)
    hidden_peak = stacked[:, keep_c].max(axis=(0, 1))
    keep_h = np.where(hidden_peak >= threshold)[0]

    grids = [g[np.ix_(keep_c, keep_h)].T for g in grids]   # (n_kept_h, n_kept_c)
    comp_labels = [f"C{c}" for c in keep_c]
    n_kc, n_kh = len(keep_c), len(keep_h)

    vmax = max(max(g.max() for g in grids), 1e-12)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax)  # 4) gamma<1 lifts faint cells

    fig, axes = plt.subplots(
        1, n_groups,
        figsize=(max(2.0, 0.4 * n_kc) * n_groups + 1.5, 0.05 * n_kh + 2),
        sharey=True,
    )
    if n_groups == 1:
        axes = [axes]

    step = max(1, n_kh // 15)
    for g, (ax, grid) in enumerate(zip(axes, grids)):
        im = ax.imshow(grid, aspect="auto", cmap="Reds", norm=norm)
        ax.set_title(f"input group {g}")
        ax.set_xlabel("component")
        ax.set_xticks(range(n_kc))
        ax.set_xticklabels(comp_labels, rotation=90)
    axes[0].set_ylabel("hidden neuron")
    axes[0].set_yticks(range(0, n_kh, step))
    axes[0].set_yticklabels([str(keep_h[i]) for i in range(0, n_kh, step)])

    fig.colorbar(im, ax=axes, shrink=0.85, label="contribution")
    fig.suptitle(title or f"Component contribution (kept {n_kc}/{n_comp} comps, "
                          f"{n_kh}/{d_out} neurons)")
    save_path = title.replace(" ", "_").lower() + ".png"
    save_figure(fig, save_dir / save_path)
    plt.close(fig)



def main() -> None:
      
    run_dirs = [
        r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2",
        r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4",
        r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5",
        r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8",
        r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\differnet_minimalities\0.1",
        r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\differnet_minimalities\0.001",
        r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\differnet_minimalities\0.0001",
        r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\differnet_minimalities\0.00001",
        r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\differnet_minimalities\0.000001",

    ]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    for run_dir in run_dirs:
        RUN_DIR = Path(
            run_dir)
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
        C1 = einops.einsum(A1, B1, "d_in C K, C K d_out -> C d_out d_in").detach().cpu().numpy()
        C2 = einops.einsum(A2, B2, "d_in C K, C K d_out -> C d_out d_in").detach().cpu().numpy()

        group_features, _, _ = get_group_features(ranks)

        #Plots that show the magnitude of the contribution of each component group to the hidden neurons or outputs
        
        plot_contribution_of_components_to_outputs(W1, save_dir=RUN_DIR, group_size=6, title="W1 contribution of input features to hidden neurons")
        plot_contribution_of_components_to_outputs(W2.T, save_dir=RUN_DIR, group_size=1, title="W2 contribution of hidden neurons to outputs")

        plot_group_output_matrix(W1, W2, group_features, save_dir=RUN_DIR, title="Group → output response (forward pass)")

        plot_input_hidden_output_translation(W1, W2, group_features, save_dir=RUN_DIR, title="Input → hidden → output translation")
        #Combined plot, that shows how the input features are routed to the outputs.

        plot_group_output_matrix(W1, W2, group_features, save_dir=RUN_DIR, title="Group → output response (forward pass)")

        plot_component_input_output_grids(C1, group_features, save_dir=RUN_DIR, title="w1_Component contribution to hidden neurons (group one-hot)")
        plot_component_input_output_grids(C2, group_features, save_dir=RUN_DIR, title="w2_Component contribution to outputs (group one-hot)")



    # #Combine the above generated plots into one plot:

    # group_images = [[
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2\w1_w2_input_hidden_output_translation.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4\w1_w2_input_hidden_output_translation.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\w1_w2_input_hidden_output_translation.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8\w1_w2_input_hidden_output_translation.png"],
    #             [
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2\group_output_matrix_simplex.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4\group_output_matrix_simplex.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\group_output_matrix_simplex.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8\group_output_matrix_simplex.png"
    #             ],
    #             [
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2\causal_importances_upper_leaky_30000.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4\causal_importances_upper_leaky_30000.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\causal_importances_upper_leaky_30000.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8\causal_importances_upper_leaky_30000.png"
    #             ],
    #             [
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2\w1_contribution_of_input_features_to_hidden_neurons.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4\w1_contribution_of_input_features_to_hidden_neurons.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\w1_contribution_of_input_features_to_hidden_neurons.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8\w1_contribution_of_input_features_to_hidden_neurons.png",
    #             ],
    #             [
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2\w2_contribution_of_hidden_neurons_to_outputs.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4\w2_contribution_of_hidden_neurons_to_outputs.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5\w2_contribution_of_hidden_neurons_to_outputs.png",
    #             r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8\w2_contribution_of_hidden_neurons_to_outputs.png"
    #             ],

    # ]
    # #Its scrappy but its good enough.
    # orientations = ["vertical", "vertical", "horizontal", "vertical", "vertical"]
    # names = ["input_hidden_output_translation", "group_output_matrix_simplex", "causal_importances", "w1_contribution_of_input_features_to_hidden_neurons", "w2_contribution_of_hidden_neurons_to_outputs"]

    # for group_images, orientation, name in zip(group_images, orientations, names):
    #     combine_images(
    #         group_images,
    #         save_path= name,
    #         orientation=orientation,
    #         size=6,
    #         dpi=300
    #     )

if __name__ == "__main__":
    main()  