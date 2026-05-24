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
from  matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable




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


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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

    A = np.zeros((n_groups, n_hidden))
    B = W2


    for g, feature_ids in enumerate(group_features):
        #(hidden, features)
        group_input = W1[:, feature_ids]

        #sum over the feature group, to find the total activation of each hidden neuron from this group.
        hidden_activation = group_input.sum(axis=1)
        A[g, :] = np.maximum(hidden_activation, 0)

    #Across every group, pick the one that maximally activates it.
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




def compute_io_flows(C1, C2, group_features):

    read_per_group = []

    #C1 = (C, d_hid, d_in)
    #C2 = (C, d_out, d_hid)
    for g in group_features:
        C1_g = C1[:, :, g]          # (C, d_hid, |g|)
        C1_g_sum = np.sum(C1_g, axis=-1)     # (C, d_hid)
        read_per_group.append(C1_g_sum)

    read = np.stack(read_per_group, axis=-1)  # (C, d_hid, n_groups)
    read = np.sum(read, axis=-2)              # (C, n_groups)

    write = C2.sum(axis=-1)  # (C2, n_outputs)

    in_h = np.sum(C1, axis=-1)   # (C, d_hid)
    out_h = np.sum(C2, axis=-2)  # (C, d_hid)
    overlap = in_h @ out_h.T            # (C, C)
    
    return read, overlap, write


def pick_components(read, write, coverage=0.95, min_mass=0):
    """
    Drop any subcomponent below `min_mass`, then keep the fewest survivors making
    up `coverage` of the remaining mass. C1 ranked by total read (into hiddens),
    C2 by total write (into outputs).
    read: (C1, n_groups)   write: (C2, n_outputs)
    """
    #contrib1 = read.sum(axis=1)      # (C1,) total read per input subcomponent
    #contrib2 = write.sum(axis=1)     # (C2,) total write per output subcomponent
    
    #Resorted to this version as the components would cancel each other out otherwise.
    contrib1 = np.maximum(read, 0).sum(axis=1)      # (C1,) total positive read per input subcomponent
    contrib2 = np.maximum(write, 0).sum(axis=1)     # (C2,) total positive write per output subcomponent

    def cover(v):
        live = np.where(v >= min_mass)[0]
        if live.size == 0:
            raise ValueError("No components meet the minimum mass requirement.")
        
        order = live[np.argsort(-v[live])]
        cum = np.cumsum(v[order])
        total = cum[-1]

        if total <= 0:
            raise ValueError("Total mass is zero or negative.")
        
        n = np.searchsorted(cum, coverage * total) + 1
        return order[:n]

    return cover(contrib1), cover(contrib2)


def build_columns(read, write, n_groups, n_outputs, c1, c2):
    return [
        {
            "title": "Input groups",
            "ids": [i for i in range(n_groups)],
            "imp": read.sum(0),
            "order": list(range(n_groups)),
        },
        {
            "title": "W2 subcomponents",
            "ids": [str(i) for i in c1],
            "imp": read.sum(1),
            "order": list(range(len(c1))),    
        },
        {
            "title": "W1 subcomponents",
            "ids": [str(i) for i in c2],
            "imp": write.sum(1),
            "order": list(range(len(c2))),
        },
        {
            "title": "Output groups",
            "ids": [i for i in range(n_outputs)],
            "imp": write.sum(0),
            "order": list(range(n_outputs)),
        },
    ]


def barycenter_sweep(columns, edges, sweeps=8):
    """
    Some sorting algorithm from stack exchange. Not too important,
    as it only serves to disentangle the visualization.
    """
    sizes = [len(c["ids"]) for c in columns]

    for c in columns:
        c["order"] = list(range(len(c["ids"])))

    for _ in range(sweeps):
        for col, nbr, mat in edges:

            nrank = np.empty(sizes[nbr])
            nrank[columns[nbr]["order"]] = np.arange(sizes[nbr])

            w = mat.sum(1)

            bc = (mat @ nrank) / np.where(w > 0, w, 1)

            cur = np.empty(sizes[col])
            cur[columns[col]["order"]] = np.arange(sizes[col])

            bc = np.where(w > 0, bc, cur)

            columns[col]["order"] = list(np.argsort(bc, kind="stable"))

    return columns

def layout_column(imp, order, gap=0.03):

    n = len(imp)

    #Every "node" is the same size.
    total_mass = 1 - gap * (n - 1)
    each = total_mass / n
    h = np.full(n, each)


    y = 1.0
    centers = np.empty(n)
    for i in order:
        centers[i] = y - (h[i] / 2)
        y -= h[i] + gap

    return centers, h


def render_io_chain(columns, flows, save_path, title=None, edge_frac=0.05):
    xs = [0, 1, 2, 3]
    bw = 0.045

    flow_colors = ["#378ADD", "#7F77DD", "#D85A30"]

    fig, ax = plt.subplots(figsize=(13, 7))

    def ribbon(x0, y0, x1, y1, lw, color, alpha):
        xm = (x0 + x1) / 2
        path = MplPath(
            [(x0, y0), (xm, y0), (xm, y1), (x1, y1)],
            [MplPath.MOVETO, MplPath.CURVE4,
             MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(PathPatch(
            path,
            fill=False,
            lw=lw,
            edgecolor=color,
            alpha=alpha,
            capstyle="round",
        ))

    # flows
    for i, M in enumerate(flows):
        max_flow = np.abs(M).max()
        if max_flow <= 0:
            raise ValueError("Flow matrix has non-positive maximum value, cannot scale ribbons.")
        #horizontal starting, end points bw= beam width
        x0 = xs[i] + bw / 2
        x1 = xs[i + 1] - bw / 2

        threshold = edge_frac * max_flow
        active_sources, active_targets = np.where(np.abs(M) >= threshold)

        for s, d in zip(active_sources, active_targets):

            flow_value = M[s, d]
            r = flow_value / max_flow #this normalizes to 0 <= 1 <= 1
            r= np.abs(r)

            ribbon(
                x0,
                columns[i]["centers"][s],
                x1,
                columns[i + 1]["centers"][d],
                0.5 + 7 * r,
                flow_colors[i],
                0.15 + 0.5 * np.abs(r),
            )

        # flows
    # for i, M in enumerate(flows):
    #     max_flow = M.max()
        
    #     #horizontal starting, end points bw= beam width
    #     x0 = xs[i] + bw / 2
    #     x1 = xs[i + 1] - bw / 2

        
    #     #threshold = edge_frac * max_flow
    #     #active_sources, active_targets = np.where(M > threshold)
    #     active_sources, active_targets = np.indices(M.shape).reshape(2, -1)
    #     for s, d in zip(active_sources, active_targets):

    #         flow_value = M[s, d]
    #         r = flow_value / max_flow #this normalizes to 0 <= 1 <= 1


    #         ribbon(
    #             x0,
    #             columns[i]["centers"][s],
    #             x1,
    #             columns[i + 1]["centers"][d],
    #             np.abs(0.5 + 7 * r),
    #             flow_colors[i],
    #             np.minimum(1.0,0.3 +0.5 * np.abs(r))
    #         )

    # nodes
    for i, col in enumerate(columns):
        for j, label in enumerate(col["ids"]):
            yc = col["centers"][j]
            h = col["heights"][j]

            ax.add_patch(Rectangle(
                (xs[i] - bw / 2, yc - h / 2),
                bw, h,
                facecolor="#B5D4F4" if i < 2 else "#F5C4B3",
                edgecolor="#185FA5" if i < 2 else "#993C1D",
                lw=0.8,
                zorder=3,
            ))

            ax.text(xs[i], yc, label, ha="center", va="center", fontsize=8)

        ax.text(xs[i], 1.06, col["title"], ha="center", fontsize=11)

    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-0.1, 1.13)
    ax.axis("off")

    if title:
        fig.suptitle(title, y=0.9)

    save_figure(fig, save_path)
    plt.close(fig)


def plot_io_routing_chain(
    C1, C2, group_features, save_dir,
    title=None, coverage=0.5,
    edge_frac=0.01, sort_nodes=True, sweeps=20, min_mass=0.05,
):
    read, overlap, write = compute_io_flows(C1, C2, group_features)
    #Read has dim (C,_n groups), overlap ahs dim (C1, C2), write has dim (C2, n_outputs)

    #Select only the components that matter significantly
    c1, c2 = pick_components(
        read, write,
        coverage=coverage, min_mass=min_mass,
    )

    #pick components that satisfy minimum mass
    read = read[c1, :]        
    write = write[c2, :]   
    
    #Likewise for the overlap, only the components that carry mass.
    overlap = overlap[np.ix_(c1, c2)]

    columns = build_columns(
        read, write,
        len(group_features),
        C2.shape[1],
        c1, c2,
    )


    if sort_nodes:
        edges = [
        (1, 0, read),
        (2, 1, overlap.T),
        (2, 3, write),
        (1, 2, overlap),
    ]
        columns = barycenter_sweep(columns, edges, sweeps)
  
    for col in columns:
        col["centers"], col["heights"] = layout_column(col["imp"], col["order"])

    render_io_chain(
        columns,
        [read.T, overlap, write],
        save_dir / "io_routing_chain.png",
        title,
        edge_frac,
    )


    
def main() -> None:
      
    run_dirs = [
        #minimaities
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.1", "(minimality 1e-1)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.01", "(minimality 1e-2)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.001", "(minimality 1e-3)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.0001", "(minimality 1e-4)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.00001", "(minimality 1e-5)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.000001", "(minimality 1e-6)"),
        #Rank plots:
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_1", "(Rank 1)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_2", "(Rank 2)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_3", "(Rank 3)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_4", "(Rank 4)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_5", "(Rank 5)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_6", "(Rank 6)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_7", "(Rank 7)"),
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\rank_plots\rank_8", "(Rank 8)")
    ]
    
    model_dir = r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\smaller_test"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    for run_dir, run_title in run_dirs:
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
        
        #Components
        C1 = einops.einsum(A1, B1, "d_in C K, C K d_out -> C d_out d_in").detach().cpu().numpy()
        C2 = einops.einsum(A2, B2, "d_in C K, C K d_out -> C d_out d_in").detach().cpu().numpy()

        group_features, _, _ = get_group_features(ranks)

        plot_group_output_matrix(W1, W2, group_features, save_dir=RUN_DIR, title="Group output matrix " + run_title)
        plot_io_routing_chain(C1, C2, group_features, save_dir=RUN_DIR, title="Group to output routing " + run_title, coverage=0.9, edge_frac=0.05, min_mass=0.05)
    
    
    model_dir = Path(model_dir)
    state_dict = torch.load(
        model_dir / "geometry.pth",
        map_location=DEVICE,
    )
    print(state_dict.keys())
    W1 = state_dict["linear1.weight"].detach().cpu().numpy()
    W2 = state_dict["linear2.weight"].detach().cpu().numpy()
    ranks = [2,2,2]
    group_features, _, _ = get_group_features(ranks)

    plot_input_hidden_output_translation(W1, W2, group_features, save_dir=model_dir, title="Layered translation from input groups to outputs")
    # plot_io_routing_chain(
    #     A1, B1, group_features, model_dir,
    #     title="Input → hidden → output routing chain (components, smaller test)",
    #     components=False,
    # )
if __name__ == "__main__":
    main()  