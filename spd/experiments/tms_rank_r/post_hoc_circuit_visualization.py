import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

from spd.experiments.tms.models import TMSModel, TMSModelConfig
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_causal_importances
from spd.configs import Config


def load_models(decomposition_dir: str, device: str = "cpu"):
    decomposition_dir = Path(decomposition_dir)

    with open(decomposition_dir / "final_config.yaml") as f:
        config = Config(**yaml.safe_load(f))

    # Hardcode the TMS config — you know these values
    tms_model_config = TMSModelConfig(
        n_features=6,
        n_hidden=3,
        n_hidden_layers=0,
        tied_weights=True,
        device=device,
        init_bias_to_zero=False,
    )

    target_model = TMSModel(config=tms_model_config)
    target_model.load_state_dict(
        torch.load(config.pretrained_model_path, map_location=device, weights_only=True)
    )
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
        decomposition_dir / "model_40000.pth", map_location=device, weights_only=True
    )
    component_model.load_state_dict(checkpoint)
    component_model.to(device)
    component_model.eval()

    return component_model, config

def get_importances_for_input(component_model, x, config):
    """Run a single input through the model and get causal importances."""
    module_names = [p.replace("-", ".") for p in component_model.components.keys()]

    _, pre_weight_acts = component_model.forward_with_pre_forward_cache_hooks(
        x, module_names=module_names
    )

    # Rename keys to match component names
    pre_weight_acts = {k.replace(".", "-"): v for k, v in pre_weight_acts.items()}

    As = {name: comp.A for name, comp in component_model.components.items()}

    causal_importances, _ = calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        As=As,
        gnan=component_model.gnan,
    )

    return causal_importances


def evaluate_group_importances(component_model, config, groups, n_features, 
                                magnitude=0.75, n_samples=256, device="cpu"):
    results = {}

    for g_idx, group_indices in enumerate(groups):
        # Create a batch, not a single sample
        x = torch.zeros(n_samples, n_features, device=device)
        for i in range(n_samples):
            x[i, group_indices] = torch.rand(len(group_indices), device=device) * magnitude

        with torch.no_grad():
            ci = get_importances_for_input(component_model, x, config)

        for layer_name, importance in ci.items():
            if layer_name not in results:
                results[layer_name] = []
            # Average importance across the batch
            results[layer_name].append(importance.mean(dim=0).detach().cpu().numpy())

    for layer_name in results:
        results[layer_name] = np.stack(results[layer_name], axis=0)

    return results


def plot_group_importances(results, groups):
    """Heatmap: rows = groups, columns = components."""
    n_layers = len(results)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    group_labels = [str(g) for g in groups]

    for ax, (layer_name, importance_matrix) in zip(axes, results.items()):
        im = ax.imshow(importance_matrix, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_title(layer_name)
        ax.set_xlabel("Component index")
        ax.set_ylabel("Input group")
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(group_labels, fontsize=8)
        ax.set_xticks(range(importance_matrix.shape[1]))

    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle("Importance by input group", fontsize=14)
    fig.tight_layout()
    return fig


def plot_shared_feature_comparison(results, groups, n_features):
    """For shared features, show how importance shifts by group context."""
    feature_to_groups = {}
    for g_idx, group in enumerate(groups):
        for f in group:
            feature_to_groups.setdefault(f, []).append(g_idx)
    shared_features = {f: gs for f, gs in feature_to_groups.items() if len(gs) > 1}

    figs = []
    for layer_name, importance_matrix in results.items():
        n_shared = len(shared_features)
        fig, axes = plt.subplots(1, n_shared, figsize=(4 * n_shared, 3))
        if n_shared == 1:
            axes = [axes]

        for ax, (feat, group_indices) in zip(axes, shared_features.items()):
            n_components = importance_matrix.shape[1]
            x_pos = np.arange(n_components)
            width = 0.8 / len(group_indices)

            for i, g_idx in enumerate(group_indices):
                ax.bar(
                    x_pos + i * width,
                    importance_matrix[g_idx],
                    width=width,
                    label=f"group {groups[g_idx]}",
                    alpha=0.8,
                )

            ax.set_title(f"Feature {feat}")
            ax.set_xlabel("Component")
            ax.set_ylabel("Importance")
            ax.legend(fontsize=7)
            ax.set_xticks(x_pos)

        fig.suptitle(f"Shared feature importances — {layer_name}", fontsize=12)
        fig.tight_layout()
        figs.append(fig)

    return figs


def plot_component_circuits(results, groups, n_features):
    """Per component: which features does it handle, for each input group."""
    figs = []
    for layer_name, importance_matrix in results.items():
        n_components = importance_matrix.shape[1]
        fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 3))
        if n_components == 1:
            axes = [axes]

        for comp_idx, ax in enumerate(axes):
            x_pos = np.arange(len(groups))
            ax.bar(x_pos, importance_matrix[:, comp_idx])
            ax.set_title(f"Component {comp_idx}")
            ax.set_xlabel("Input group")
            ax.set_ylabel("Importance")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(g) for g in groups], fontsize=7, rotation=45)
            ax.set_ylim(0, 1)

        fig.suptitle(f"Which input groups activate each component — {layer_name}", fontsize=12)
        fig.tight_layout()
        figs.append(fig)

    return figs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Point this to your decomposition output directory
    decomposition_dir = "C:\\Users\\Knud\\uni\\spd\\spd\\experiments\\tms_rank_r\\out\\nmasks1_stochrecon1.00e+00_stochreconlayer1.00e+00_p1.00e+00_impmin3.00e-03_C4_sd0_lr1.00e-03_bs2048_ft6_hid3hid-layers0_20260412_202911_421"
    component_model, config = load_models(decomposition_dir, device)

    groups = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    n_features = 6



    target_model = component_model.model
    target_model.eval()
    groups = [[0,1,2], [1,2,3], [2,3,4], [3,4,5]]

    for g_idx, group in enumerate(groups):
        x = torch.zeros(1, 6, device=device)
        x[0, group] = 0.75
        with torch.no_grad():
            out = target_model(x)
        print(f"Group {group}: input={x[0].tolist()}")
        print(f"  output={[f'{v:.3f}' for v in out[0].tolist()]}")
        print()
    results = evaluate_group_importances(
        component_model, config, groups, n_features,
        magnitude=0.75, device=device,
    )

    # Check what each component's weight slice looks like
    for name, comp in component_model.components.items():
        print(f"\n=== {name} ===")
        A = comp.A.detach().cpu()  # (d_in, C, k)
        B = comp.B.detach().cpu()  # (C, k, d_out)
        
        for c in range(4):
            W_c = A[:, c, :] @ B[c, :, :]  # (d_in, d_out)
            # Which input features does this component use?
            input_norms = W_c.abs().sum(dim=1)  # (d_in,)
            print(f"  Component {c} input feature norms: {[f'{v:.3f}' for v in input_norms.tolist()]}")

    fig1 = plot_group_importances(results, groups)
    fig1.savefig("group_importances.png", dpi=150)

    figs = plot_shared_feature_comparison(results, groups, n_features)
    for i, fig in enumerate(figs):
        fig.savefig(f"shared_feature_comparison_{i}.png", dpi=150)

    figs = plot_component_circuits(results, groups, n_features)
    for i, fig in enumerate(figs):
        fig.savefig(f"component_circuits_{i}.png", dpi=150)

    plt.show()