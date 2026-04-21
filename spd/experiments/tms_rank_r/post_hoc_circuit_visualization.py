"""Diagnose a trained SPD decomposition: which components fire for which features?

Usage:
    Edit DECOMPOSITION_DIR and GROUPS below, then run.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

from spd.experiments.tms_rank_r.models import TMSModel, TMSModelConfig
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_causal_importances
from spd.configs import Config


# =============== EDIT THESE ===============

FOLDER_NAME = "nmasks1_stochrecon1.00e+00_stochreconlayer1.00e+00_p1.00e+00_impmin5.00e-03_C30_sd20_lr1.00e-03_bs2048_ft8_hid6hid-layers0_20260420_222043_030"

DECOMPOSITION_DIR = r"C:\\Users\\Knud\\uni\\spd\\spd\\experiments\\tms_rank_r\\out\\" + FOLDER_NAME
GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]   # your pair structure
N_FEATURES = 8
MAGNITUDE = 0.75
CHECKPOINT_NAME = "model_40000.pth"
N_ROTATIONS = 12   # number of angles to probe per circle
# ==========================================

from pathlib import Path
import yaml
from spd.configs import Config

decomposition_dir = Path(DECOMPOSITION_DIR)
with open(decomposition_dir / "final_config.yaml") as f:
    config = Config(**yaml.safe_load(f))

print("SPD was decomposing:", config.pretrained_model_path)

def load_models(decomposition_dir: str, device: str):
    decomposition_dir = Path(decomposition_dir)

    with open(decomposition_dir / "final_config.yaml") as f:
        config = Config(**yaml.safe_load(f))

    target_model, _ = TMSModel.from_pretrained(config.pretrained_model_path)
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


def evaluate_per_feature(component_model, n_features, magnitude, n_samples, device):
    """Probe each input coordinate individually."""
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
        results[layer_name] = np.stack(results[layer_name], axis=0)
    return results


def plot_heatmap(results, groups, n_features, savepath):
    n_layers = len(results)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4),
                              constrained_layout=True)
    if n_layers == 1:
        axes = [axes]

    feature_to_group = {}
    for g_idx, group in enumerate(groups):
        for f in group:
            feature_to_group[f] = g_idx

    for ax, (layer_name, mat) in zip(axes, results.items()):
        im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_title(layer_name)
        ax.set_xlabel("Component index")
        ax.set_ylabel("Input feature")
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([f"{i} (g{feature_to_group.get(i,'?')})" for i in range(n_features)])
        ax.set_xticks(range(mat.shape[1]))

        for i in range(1, n_features):
            if feature_to_group.get(i) != feature_to_group.get(i - 1):
                ax.axhline(i - 0.5, color='cyan', linewidth=1.5)

    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"Importance per input feature (magnitude={MAGNITUDE})", fontsize=13)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def print_component_summary(component_model):
    """For each component in each layer, show which input features it reads from."""
    for name, comp in component_model.components.items():
        print(f"\n=== {name} ===")
        A = comp.A.detach().cpu()
        C = A.shape[1]
        B = comp.B.detach().cpu()

        for c in range(C):
            W_c = A[:, c, :] @ B[c, :, :]
            input_norms = W_c.abs().sum(dim=1)
            active = (input_norms > 0.05).nonzero().flatten().tolist()
            print(f"  Component {c}: "
                  f"reads from features {active}  "
                  f"(norms: {[f'{v:.2f}' for v in input_norms.tolist()]})")


def print_target_sanity(target_model, groups, n_features, device):
    """Sanity: show what the target model does on single-group inputs (corner point)."""
    print("\n=== Target model sanity check (corner probe) ===")
    for group in groups:
        x = torch.zeros(1, n_features, device=device)
        x[0, group] = MAGNITUDE
        with torch.no_grad():
            out = target_model(x)
        print(f"  Group {group}: out = {[f'{v:.3f}' for v in out[0].tolist()]}")


def probe_circle_rotations(target_model, groups, n_features, n_rotations, device):
    print(f"\n=== Circle rotation probe ({n_rotations} angles per group, unit circle) ===")
    thetas = np.linspace(0, 2 * np.pi, n_rotations, endpoint=False)

    # Compute bias (zero input response)
    x_zero = torch.zeros(1, n_features, device=device)
    with torch.no_grad():
        bias_out = target_model(x_zero)

    for group in groups:
        if len(group) != 2:
            continue

        pair_response_norms = []
        for theta in thetas:
            x = torch.zeros(1, n_features, device=device)
            x[0, group[0]] = np.cos(theta)
            x[0, group[1]] = np.sin(theta)
            with torch.no_grad():
                out = target_model(x)
            response = out - bias_out   # subtract bias
            pair_response_norms.append(response[0, group].norm().item())

        pair_response_norms = np.array(pair_response_norms)
        print(f"  Group {group}:")
        print(f"    pair response norm — mean={pair_response_norms.mean():.3f}, "
              f"std={pair_response_norms.std():.3f}, "
              f"min={pair_response_norms.min():.3f}, max={pair_response_norms.max():.3f}")
        ratio = pair_response_norms.max() / max(pair_response_norms.min(), 1e-6)
        print(f"    pair response max/min ratio = {ratio:.2f}  (≈1 rank-2, ≫1 rank-1)")

def plot_circle_rotations(target_model, groups, n_features, magnitude,
                            n_rotations, device, savepath):
    """Plot output pair norm vs. rotation angle for each group."""
    thetas = np.linspace(0, 2 * np.pi, n_rotations, endpoint=False)
    fig, ax = plt.subplots(figsize=(7, 5))

    for group in groups:
        if len(group) != 2:
            continue
        pair_out_norms = []
        for theta in thetas:
            x = torch.zeros(1, n_features, device=device)
            x[0, group[0]] = magnitude * np.cos(theta)
            x[0, group[1]] = magnitude * np.sin(theta)
            with torch.no_grad():
                out = target_model(x)
            pair_out_norms.append(out[0, group].norm().item())
        ax.plot(thetas, pair_out_norms, marker='o', label=f"group {group}")

    ax.axhline(magnitude, color='gray', linestyle='--',
               label=f"input norm ({magnitude})")
    ax.set_xlabel("Rotation angle (radians)")
    ax.set_ylabel("Output pair norm")
    ax.set_title("Circle rotation probe — output norm stability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def probe_circle_reconstruction(model, groups, n_features, n_rotations, device):
    print(f"\n=== Circle reconstruction probe ({n_rotations} angles) ===")
    thetas = np.linspace(0, 2 * np.pi, n_rotations, endpoint=False)

    for group in groups:
        if len(group) != 2:
            continue

        recon_errors = []
        norm_errors = []
        leakage = []

        for theta in thetas:
            x = torch.zeros(1, n_features, device=device)
            x[0, group[0]] = np.cos(theta)
            x[0, group[1]] = np.sin(theta)

            with torch.no_grad():
                out = model(x)

            out_pair = out[0, group]
            target = x[0, group]

            # 1. reconstruction error (should be ~0)
            recon_errors.append((out_pair - target).norm().item())

            # 2. norm preservation (should be ~1)
            norm_errors.append(abs(out_pair.norm().item() - 1.0))

            # 3. leakage to other coords (should be ~0)
            mask = torch.ones(n_features, dtype=torch.bool, device=device)
            mask[group] = False
            leakage.append(out[0, mask].norm().item())

        recon_errors = np.array(recon_errors)
        norm_errors = np.array(norm_errors)
        leakage = np.array(leakage)

        print(f"  Group {group}:")
        print(f"    recon error — mean={recon_errors.mean():.4f}, max={recon_errors.max():.4f}")
        print(f"    norm error  — mean={norm_errors.mean():.4f}, max={norm_errors.max():.4f}")
        print(f"    leakage     — mean={leakage.mean():.4f}, max={leakage.max():.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    component_model, config = load_models(DECOMPOSITION_DIR, device)
    target_model = component_model.model
    target_model.eval()

    print(f"C = {config.C}, k = {config.k}")
    print(f"Groups = {GROUPS}")

    print_target_sanity(target_model, GROUPS, N_FEATURES, device)

    probe_circle_rotations(
        target_model, GROUPS, N_FEATURES, n_rotations=N_ROTATIONS, device=device,
    )

    plot_circle_rotations(
        target_model, GROUPS, N_FEATURES,
        magnitude=MAGNITUDE, n_rotations=N_ROTATIONS, device=device,
        savepath="circle_rotation_probe.png",
    )
    print("Saved plot: circle_rotation_probe.png")

    results = evaluate_per_feature(
        component_model, N_FEATURES,
        magnitude=MAGNITUDE, n_samples=256, device=device,
    )

    print_component_summary(component_model)

    fig = plot_heatmap(results, GROUPS, N_FEATURES, savepath="per_feature_importances.png")
    print("\nSaved plot: per_feature_importances.png")

    probe_circle_reconstruction(
        target_model, GROUPS, N_FEATURES, n_rotations=N_ROTATIONS, device=device,
    )
    plt.show()