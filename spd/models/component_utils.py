from collections.abc import Mapping

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
import torch.nn as nn
from spd.utils import extract_batch_data
import torch_geometric as pyg



def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.

    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """
    stochastic_masks = []
    for _ in range(n_mask_samples):
        stochastic_masks.append(
            {layer: ci + (1 - ci) * torch.rand_like(ci) for layer, ci in causal_importances.items()}
        )
    return stochastic_masks


def calc_ci_l_zero(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    cutoff: float = 1e-2,
) -> dict[str, float]:
    """Calculate the L0 loss on the causal importances, summed over the C dimension."""
    ci_l_zero = {}
    for layer_name, ci in causal_importances.items():
        mean_dims = tuple(range(ci.ndim - 1))
        ci_l_zero[layer_name] = (ci > cutoff).float().mean(dim=mean_dims).sum().item()
    return ci_l_zero


def component_activation_statistics(
    model: ComponentModel,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_steps: int,
    device: str,
    threshold: float = 0.1,
) -> tuple[dict[str, float], dict[str, Float[Tensor, " C"]]]:
    """Get the number and strength of the masks over the full dataset."""
    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    n_tokens = {module_name.replace("-", "."): 0 for module_name in components}
    total_n_active_components = {module_name.replace("-", "."): 0 for module_name in components}
    component_activation_counts = {
        module_name.replace("-", "."): torch.zeros(model.C, device=device)
        for module_name in components
    }
    data_iter = iter(dataloader)
    for _ in range(n_steps):
        # --- Get Batch --- #
        batch = extract_batch_data(next(data_iter))
        batch = batch.to(device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        As = {module_name: v.A for module_name, v in components.items()}

        causal_importances, _ = calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            As=As,
            gates=gates,
            detach_inputs=False,
        )
        for module_name, ci in causal_importances.items():
            # mask (batch, pos, C) or (batch, C)
            n_tokens[module_name] += ci.shape[:-1].numel()

            # Count the number of components that are active above the threshold
            active_components = ci > threshold
            total_n_active_components[module_name] += int(active_components.sum().item())

            sum_dims = tuple(range(ci.ndim - 1))
            component_activation_counts[module_name] += active_components.sum(dim=sum_dims)

    # Show the mean number of components
    mean_n_active_components_per_token: dict[str, float] = {
        module_name: (total_n_active_components[module_name] / n_tokens[module_name])
        for module_name in components
    }
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]] = {
        module_name: component_activation_counts[module_name] / n_tokens[module_name]
        for module_name in components
    }

    return mean_n_active_components_per_token, mean_component_activation_counts


def lower_leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    return torch.where(x > 0, torch.clamp(x, max=1), alpha * x)


def upper_leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    # TODO: Make more memory efficient
    return torch.where(x > 1, 1 + alpha * (x - 1), F.relu(x))


# def calc_causal_importances(
#     pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
#     As: Mapping[str, Float[Tensor, "d_in C"]],
#     gates: Mapping[str, Gate | GateMLP],
#     detach_inputs: bool = False,
# ) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:
#     """Calculate component activations and causal importances in one pass to save memory.

#     Args:
#         pre_weight_acts: The activations before each layer in the target model.
#         As: The A matrix at each layer.
#         gates: The gates to use for the mask.
#         detach_inputs: Whether to detach the inputs to the gates.

#     Returns:
#         Tuple of (causal_importances, causal_importances_upper_leaky) dictionaries for each layer.
#     """
#     causal_importances = {}
#     causal_importances_upper_leaky = {}

#     for param_name in pre_weight_acts:
#         acts = pre_weight_acts[param_name]

#         if not acts.dtype.is_floating_point:
#             # Embedding layer
#             component_act = As[param_name][acts]
#         else:
#             # Linear layer
#             component_act = einops.einsum(acts, As[param_name], "... d_in, d_in C -> ... C")

#         gate_input = component_act.detach() if detach_inputs else component_act
#         gate_output = gates[param_name](gate_input)
#         causal_importances[param_name] = lower_leaky_relu(gate_output)
#         causal_importances_upper_leaky[param_name] = upper_leaky_relu(gate_output)

#     return causal_importances, causal_importances_upper_leaky

def _construct_edge_index(all_gate_outputs: dict, device: torch.device) -> Tensor:
    total_nodes = sum(v.shape[-1] for v in all_gate_outputs.values())
    all_nodes = torch.arange(total_nodes, device=device)
    src = all_nodes.repeat(total_nodes)
    dst = all_nodes.repeat_interleave(total_nodes)
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    return edge_index


def _construct_edge_index_forward_only(all_gate_outputs: dict, device: torch.device) -> Tensor:
    total_nodes = sum(v.shape[-1] for v in all_gate_outputs.values())
    all_nodes = torch.arange(total_nodes, device=device)
    src = all_nodes.repeat(total_nodes)
    dst = all_nodes.repeat_interleave(total_nodes)
    mask = src < dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    return edge_index


def _construct_node_distances_forward_only(all_gate_outputs: dict, device: torch.device) -> Tensor:
    layer_ids = []
    for layer_idx, name in enumerate(all_gate_outputs.keys()):
        C = all_gate_outputs[name].shape[-1]
        layer_ids.extend([layer_idx] * C)
    
    layer_ids = torch.tensor(layer_ids, dtype=torch.float, device=device)
    
    # (total_nodes, total_nodes) — distance from row to col
    node_distances = (layer_ids.unsqueeze(1) - layer_ids.unsqueeze(0))
    node_distances = node_distances.masked_fill(node_distances < 0, 1e6)  # block backward only
    
    return node_distances

def _construct_node_distances(all_gate_outputs: dict, device: torch.device) -> Tensor:
    layer_ids = []
    for layer_idx, name in enumerate(all_gate_outputs.keys()):
        C = all_gate_outputs[name].shape[-1]
        layer_ids.extend([layer_idx] * C)
    layer_ids = torch.tensor(layer_ids, dtype=torch.float, device=device).unsqueeze(-1)
    node_distances = torch.cdist(layer_ids, layer_ids, p=1)
    return node_distances


def _remove_same_layer_edges(edge_index: Tensor, node_distances: Tensor) -> Tensor:
    edge_src, edge_dst = edge_index[0], edge_index[1]
    #iterate both src and dst together and check if they are in the same layer using node_distances
    #if the result is 0, then they are in the same layer and we want to remove that edge
    layer_diff = node_distances[edge_src, edge_dst]
    cross_layer_mask = layer_diff > 0
    return edge_index[:, cross_layer_mask]


# def calc_causal_importances(
#     pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
#     As: Mapping[str, Float[Tensor, "d_in C"]],
#     gates: Mapping[str, Gate | GateMLP],
#     detach_inputs: bool = False,
#     allow_same_layer_connections: bool = True,
#     use_gnn: bool = True,
# ) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:
#     causal_importances = {}
#     causal_importances_upper_leaky = {}

#     # First pass: collect activations
#     all_gate_outputs = {}
#     for param_name in pre_weight_acts:
#         acts = pre_weight_acts[param_name]
#         if not acts.dtype.is_floating_point:
#             component_act = As[param_name][acts]
#         else:
#             component_act = einops.einsum(acts, As[param_name], "... d_in, d_in C -> ... C")
#         gate_input = component_act.detach() if detach_inputs else component_act

#         if use_gnn:
#             all_gate_outputs[param_name] = gate_input  # raw activations
#         else:
#             all_gate_outputs[param_name] = gates[param_name](gate_input)  # original gate

#     # GNN pass — runs once over all layers
#     if use_gnn:
#         #had device issues, so just infer it...
#         device = next(iter(all_gate_outputs.values())).device
#         #Bidirectional edges
#         edge_index = _construct_edge_index(all_gate_outputs, device)
#         node_distances = _construct_node_distances(all_gate_outputs, device)

#         #Forward only edges
#         #edge_index = _construct_edge_index_forward_only(all_gate_outputs, device)
#         #node_distances = _construct_node_distances_forward_only(all_gate_outputs, device)

#         if not allow_same_layer_connections:
#             edge_index = _remove_same_layer_edges(edge_index, node_distances)
#             node_distances = node_distances.masked_fill(node_distances == 0, 1e6)

#         gnn = gates.get("active_module", None)
#         if gnn is None:
#             raise ValueError("GNN gate not found in gates dictionary under key 'active_module'")

#         normalization_matrix = node_distances.sum(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(node_distances)
#         node_distances = node_distances.requires_grad_(True)
#         node_feats = torch.cat([
#             all_gate_outputs[n].reshape(-1, all_gate_outputs[n].shape[-1]).mean(dim=0)
#             for n in pre_weight_acts
#         ], dim=0).unsqueeze(-1)

#         # Normalize so inputs aren't near-zero
#         node_feats = (node_feats - node_feats.mean()) / (node_feats.std() + 1e-8)

#         # print("node_feats requires_grad:", node_feats.requires_grad)
#         # print("node_feats grad_fn:", node_feats.grad_fn)

#         graph_data = pyg.data.Data(
#             x=node_feats,
#             edge_index=edge_index,
#             node_distances=node_distances,
#             normalization_matrix=normalization_matrix,
#         )

#         gnn_out = gnn(graph_data)  # (total_nodes, 1)

#         offset = 0
#         for param_name in pre_weight_acts:
#             C = all_gate_outputs[param_name].shape[-1]
#             layer_out = gnn_out[offset:offset + C].squeeze(-1)  # (C,)
#             #Residual, f(x) + x - might be dumb?
#             all_gate_outputs[param_name] = all_gate_outputs[param_name] + layer_out
#             offset += C

#     for param_name, gate_output in all_gate_outputs.items():
#         causal_importances[param_name] = lower_leaky_relu(gate_output)
#         causal_importances_upper_leaky[param_name] = upper_leaky_relu(gate_output)

#     return causal_importances, causal_importances_upper_leaky


def calc_causal_importances(
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
    As: Mapping[str, Float[Tensor, "d_in C"]],
    gates: Mapping[str, Gate | GateMLP],
    detach_inputs: bool = False,
    allow_same_layer_connections: bool = True,
    use_gnn: bool = True,
) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:
    causal_importances = {}
    causal_importances_upper_leaky = {}

    # First pass: collect activations
    all_gate_outputs = {}
    for param_name in pre_weight_acts:
        acts = pre_weight_acts[param_name]
        if not acts.dtype.is_floating_point:
            component_act = As[param_name][acts]
        else:
            component_act = einops.einsum(acts, As[param_name], "... d_in, d_in C -> ... C")
        gate_input = component_act.detach() if detach_inputs else component_act

        if use_gnn:
            all_gate_outputs[param_name] = gate_input  # raw activations
        else:
            all_gate_outputs[param_name] = gates[param_name](gate_input)  # original gate

    if use_gnn:
        #had device issues, so just infer it...
        device = next(iter(all_gate_outputs.values())).device

        #Bidirectional edges
        edge_index = _construct_edge_index(all_gate_outputs, device)
        node_distances = _construct_node_distances(all_gate_outputs, device)

        #Forward only edges
        #edge_index = _construct_edge_index_forward_only(all_gate_outputs, device)
        #node_distances = _construct_node_distances_forward_only(all_gate_outputs, device)

        if not allow_same_layer_connections:
            edge_index = _remove_same_layer_edges(edge_index, node_distances)
            node_distances = node_distances.masked_fill(node_distances == 0, 1e6)

        gnn = gates.get("active_module", None)
        if gnn is None:
            raise ValueError("GNN gate not found in gates dictionary under key 'active_module'")


        #Average activation over the batch, perhaps this is insufficient? Will test next week.
        #I believe it has problems with batch and GNAN as they expect different dimensions.
        #Would matter more with multiple features? Talk to lukas.
        # Build one feature per node (component) for the GNAN
        #Idk some issue with gnan not taking batches, and 
        #since its a graph it prolly has no quick fix, i'll rewatch the youtube series.
        # per_layer_means = []
        # for n in pre_weight_acts:
        #     acts = all_gate_outputs[n]                                          # (batch, pos, C)
        #     flat = acts.reshape(acts.shape[0] * acts.shape[1], acts.shape[2])   # (batch*pos, C)
        #     mean = flat.mean(dim=0)                                             # (C,) — avg activation per component
        #     per_layer_means.append(mean)

        # Collect the activations from each layer into a list
        per_layer = []
        for n in pre_weight_acts:
            acts = all_gate_outputs[n]   # (1, C) for this layer
            acts = acts.squeeze(0)       # (C,) — remove the pos dimension
            per_layer.append(acts)


        # Glue all the (C,) vectors end to end into one long vector
        # e.g. if layers have C=64, C=32, C=64 this gives (160,)
        node_feats = torch.cat(per_layer, dim=0)

        # Add a feature dimension so each node has 1 feature
        # (160,) -> (160, 1), which is what PyG expects: (num_nodes, num_features)
        node_feats = node_feats.unsqueeze(-1)

        # squeeze:    (0.5, 0.8, 0.2)          # shape (3,)

        # unsqueeze:  [[0.5],                   # shape (3, 1)
            #         [0.8],                   # node 0: feature = 0.5
            #         [0.2]]                   # node 1: feature = 0.8
        #                                 # node 2: feature = 0.2

        print(f"[GNAN] node_feats shape = {node_feats.shape}")
        print(f"[GNAN] node_feats min={node_feats.min().item():.4f}, max={node_feats.max().item():.4f}, mean={node_feats.mean().item():.4f}")
        print(f"[GNAN] edge_index shape = {edge_index.shape}, num edges = {edge_index.shape[1]}")
        print(f"[GNAN] node_distances shape = {node_distances.shape}")

        graph_data = pyg.data.Data(
        x=node_feats,
        edge_index=edge_index,
        node_distances=node_distances,
        )

        gnn_out = gnn(graph_data)
        print(f"[GNAN] gnn_out shape = {gnn_out.shape}")
        print(f"[GNAN] gnn_out min={gnn_out.min().item():.4f}, max={gnn_out.max().item():.4f}, mean={gnn_out.mean().item():.4f}")

        offset = 0
        for param_name in pre_weight_acts:
            C = all_gate_outputs[param_name].shape[-1]
            layer_out = gnn_out[offset:offset + C].squeeze(-1)
            print(f"[GNAN] Residual '{param_name}': layer_out shape = {layer_out.shape}, pre-residual acts shape = {all_gate_outputs[param_name].shape}")
            all_gate_outputs[param_name] = all_gate_outputs[param_name] + layer_out
            print(f"[GNAN] Residual '{param_name}': post-residual shape = {all_gate_outputs[param_name].shape}")
            offset += C
        return causal_importances, causal_importances_upper_leaky