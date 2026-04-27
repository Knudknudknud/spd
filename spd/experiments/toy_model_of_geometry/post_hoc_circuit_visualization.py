"""Diagnose a trained SPD decomposition for the Geometry task:
which components fire for which groups?

Usage:
    Edit DECOMPOSITION_DIR and RANKS below, then run.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

from spd.experiments.toy_model_of_geometry.models import GeometryModel
from spd.experiments.toy_model_of_geometry.simplex_dataset import SimplexDataset
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_causal_importances
from spd.configs import Config


# =============== EDIT THESE ===============
FOLDER_NAME = "nmasks1_stochrecon1.00e+00_stochreconlayer1.00e+00_p1.00e+00_impmin1.00e-05_C30_sd20_lr1.00e-03_bs2048__input-dim24_hid128_ranks2-2-2-2_20260421_200454_913"
DECOMPOSITION_DIR = r"C:\\Users\\Knud\\uni\\spd\\spd\\experiments\\toy_model_of_geometry\\out\\" + FOLDER_NAME

RANKS = [2, 2, 2, 2]
CHECKPOINT_NAME = "model_40000.pth"
N_SAMPLES = 512
MAGNITUDE = 0.75
# ==========================================


def compute_groups(ranks):
    """For ranks=[2,2,2,2] → [[0..5], [6..11], [12..17], [18..23]]."""
    groups = []
    start = 0
    for k in ranks:
        size = k * (k + 1)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def load_models(decomposition_dir, device):
    decomposition_dir = Path(decomposition_dir)
    with open(decomposition_dir / "final_config.yaml") as f:
        config = Config(**yaml.safe_load(f))

    target_model, _ = GeometryModel.from_pretrained(config.pretrained_model_path)
    target_model.to(device)
    target_model.eval()

    component_model = ComponentModel(
        base_model=target_model,
        target_module_patterns=config.target_module_patterns,
        C=config.C,
        k=config.k,
        n_ci_mlp_neurons=config.n_ci_mlp_neurons,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )
    checkpoint = torch.load(
        decomposition_dir / CHECKPOINT_NAME,
        map_location=device,
        weights_only=True,
    )
    component_model.load_state_dict(checkpoint)
    component_model.to(device)
    component_model.eval()
    return component_model, config


def get_importances(component_model, x):
    module_names = [p.replace("-", ".") for p in component_model.components.keys()]
    _, pre_weight_acts = component_model.forward_with_pre_forward_cache_hooks(
        x, module_names=module_names
    )
    pre_weight_acts = {k.replace(".", "-"): v for k, v in pre_weight_acts.items()}
    As = {name: comp.A for name, comp in component_model.components.items()}
    ci, _ = calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        As=As,
        gnan=component_model.gnan,
    )
    return ci


def evaluate_per_feature(component_model, ranks, groups, magnitude, n_samples, device):
    """For each input feature i, probe with only that feature active at `magnitude`.
    Returns {layer_name: (n_features, C)} matrix."""
    n_features = sum(k * (k + 1) for k in ranks)
    results = {}

    for i in range(n_features):
        x = torch.zeros(n_samples, n_features, device=device)
        x[:, i] = torch.rand(n_samples, device=device) * magnitude

        with torch.no_grad():
            ci = get_importances(component_model, x)

        for layer_name, importance in ci.items():
            if layer_name not in results:
                results[layer_name] = []
            results[layer_name].append(importance.mean(dim=0).detach().cpu().numpy())

    for layer_name in results:
        results[layer_name] = np.stack(results[layer_name], axis=0)  # (n_features, C)
    return results


def plot_heatmap(results, ranks, groups, savepath):
    """Heatmap: rows = input features, cols = components. One subplot per layer.
    Separator lines between groups."""
    n_layers = len(results)
    n_features = sum(k * (k + 1) for k in ranks)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 6),
                              constrained_layout=True)
    if n_layers == 1:
        axes = [axes]

    # map feature index -> group index
    feature_to_group = {}
    for g_idx, group in enumerate(groups):
        for f in group:
            feature_to_group[f] = g_idx

    for ax, (layer_name, mat) in zip(axes, results.items()):
        vmax = max(mat.max(), 1e-8)
        im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_title(layer_name)
        ax.set_xlabel("Component index")
        ax.set_ylabel("Input feature")
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([f"{i} (g{feature_to_group[i]})" for i in range(n_features)])
        ax.set_xticks(range(mat.shape[1]))

        # Draw horizontal separator lines between groups
        for i in range(1, n_features):
            if feature_to_group[i] != feature_to_group[i - 1]:
                ax.axhline(i - 0.5, color='cyan', linewidth=1.5)

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Component importance per input feature (magnitude={MAGNITUDE})", fontsize=13)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def print_group_component_assignment(results, threshold=0.1):
    """For each group, list which components fire strongly."""
    print("\n=== GROUP → COMPONENT ASSIGNMENT ===")
    for layer_name, mat in results.items():
        print(f"\n  Layer: {layer_name}")
        for g_idx in range(mat.shape[0]):
            row = mat[g_idx]
            row_norm = row / max(row.max(), 1e-8)
            active = [(c, row[c]) for c in range(len(row)) if row_norm[c] > threshold]
            active_str = ", ".join(f"c{c}={v:.2f}" for c, v in active)
            print(f"    Group {g_idx}: {active_str}")


def check_component_overlap(results, threshold=0.1):
    """For each pair of groups, count how many components they share."""
    print("\n=== COMPONENT OVERLAP BETWEEN GROUPS ===")
    for layer_name, mat in results.items():
        print(f"\n  Layer: {layer_name}")
        n_groups = mat.shape[0]
        row_norms = mat / np.maximum(mat.max(axis=1, keepdims=True), 1e-8)
        active_sets = [set(np.where(row_norms[g] > threshold)[0]) for g in range(n_groups)]

        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                shared = active_sets[i] & active_sets[j]
                print(f"    Groups {i} & {j}: {len(active_sets[i])} vs {len(active_sets[j])} active, "
                      f"shared = {len(shared)} {sorted(shared) if shared else ''}")


def probe_ablation(component_model, target_model, ranks, groups, device, n_samples=256):
    """Check: does ablating each component's importance hurt specific groups?"""
    print("\n=== ABLATION PROBE (drop one component, check which group degrades) ===")
    print("Not implemented yet — optional deeper check.")
    # Would need access to SPD's masking machinery to zero out components.
    # Can be added if needed.


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    component_model, config = load_models(DECOMPOSITION_DIR, device)
    target_model = component_model.model
    target_model.eval()
    groups = compute_groups(RANKS)

    print(f"C = {config.C}, k = {config.k}")
    print(f"RANKS = {RANKS}")
    print(f"Groups (input feature slices):")
    for g_idx, g in enumerate(groups):
        print(f"  Group {g_idx} (dim {RANKS[g_idx]}): features {g[0]}..{g[-1]}")

    results = evaluate_per_feature(
    component_model, RANKS, groups,
    magnitude=MAGNITUDE, n_samples=N_SAMPLES, device=device,
    )
    fig = plot_heatmap(results, RANKS, groups, savepath="per_feature_importances.png")
    print("\nSaved plot: per_feature_importances.png")

    print_group_component_assignment(results, threshold=0.1)
    check_component_overlap(results, threshold=0.1)

    plt.show()