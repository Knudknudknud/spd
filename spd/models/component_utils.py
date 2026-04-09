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


def _construct_edge_index_backwards_only(all_gate_outputs: dict, device: torch.device) -> Tensor:
    total_nodes = sum(v.shape[-1] for v in all_gate_outputs.values())
    all_nodes = torch.arange(total_nodes, device=device)
    src = all_nodes.repeat(total_nodes)
    dst = all_nodes.repeat_interleave(total_nodes)
    mask = src < dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    return edge_index


def _construct_node_distances_backwards_only(all_gate_outputs: dict, device: torch.device) -> Tensor:
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
    node_distances = layer_ids.unsqueeze(0) - layer_ids.unsqueeze(1)

    return node_distances


def _remove_same_layer_edges(edge_index: Tensor, node_distances: Tensor) -> Tensor:
    edge_src, edge_dst = edge_index[0], edge_index[1]
    #iterate both src and dst together and check if they are in the same layer using node_distances
    #if the result is 0, then they are in the same layer and we want to remove that edge
    layer_diff = node_distances[edge_src, edge_dst]
    cross_layer_mask = layer_diff > 0
    return edge_index[:, cross_layer_mask]



def calc_causal_importances(
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
    As: Mapping[str, Float[Tensor, "d_in C"]],
    gates: Mapping[str, Gate | GateMLP],
    detach_inputs: bool = False,
    allow_same_layer_connections: bool = True,
    use_gnn: bool = True,
    negative_distance: bool = True,
) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:

    #Big note:
    #The batching solution is okay, because we always sample all
    #nodes i think? If not it could still work as we always calculate the distances anew,
    #We dont actually use the graph structure for anything but the distances.
    causal_importances = {}
    causal_importances_upper_leaky = {}

    # First pass: collect activations
    all_gate_outputs = {}
    all_gate_feats = {}   #added to get the features too.
    #Pre_weight_acts is the dictionary of the outputs of the given layer
    for param_name in pre_weight_acts:
        acts = pre_weight_acts[param_name]
        if not acts.dtype.is_floating_point:
            component_act = As[param_name][acts]
        else:
            #The A matrix components, for each layer (C, d_in, k)
            A = As[param_name]
            #Take the products (A * features),
            #spits out a tensor of shape (batch, C, k)
            component_act_k = einops.einsum(acts, A, "... d_in, d_in C k -> ... C k")
            #For now collapse over the k, might still be a hack,
            #ask Lukas.
            #Probably just feed both values into the gnn -> change input dim to k
            component_act = component_act_k.mean(dim=-1)

        gate_input = component_act.detach() if detach_inputs else component_act
        gate_feats = component_act_k.detach() if detach_inputs else component_act_k

        if use_gnn:
            #ad the gate (mean output over k) and the features to their respective dictionaries.
            all_gate_outputs[param_name] = gate_input    # (batch, C) — for gating
            all_gate_feats[param_name] = gate_feats      # (batch, C, k) — for GNAN
        else:
            all_gate_outputs[param_name] = gates[param_name](gate_input)

    if use_gnn:
        device = next(iter(all_gate_outputs.values())).device

        #This implementation is highly inefficient for these models,
        #As the graph is fully determined on the irst iteration
        #And as such we could cache it instead of reconstructing it every time. But it works for now, and it's easier to read.
        #Likewise we dont even use the edge index, lol.
        edge_index = _construct_edge_index(all_gate_outputs, device)
        
        #Below gives an option to allow negative distances,
        #That way the gnn can learn to ignore one direction of edges
        #If it proves to be beneficial.
        node_distances = _construct_node_distances(all_gate_outputs, device)


        #only backward connections
        #edge_index = _construct_edge_index_backwards_only(all_gate_outputs, device)
        #node_distances = _construct_node_distances_backwards_only(all_gate_outputs, device)
        
        #Gnn requires distance normalization, move it into the gnn code later.
        
        #Allow negative distances, but keep the sign, by doing sign(x) * 1/(1+|x|)
        if negative_distance:
            #print("Using negative distances with sign preservation for GNN")
            sign = torch.sign(node_distances)
            sign[node_distances == 0] = 1.0  # convert 0 to 1, to give it distance 1.
            node_distances = sign * 1.0 / (1.0 + node_distances.abs())

        else:
            node_distances =  1.0 / (1.0 + node_distances.abs())



        if not allow_same_layer_connections:
            edge_index = _remove_same_layer_edges(edge_index, node_distances)
            node_distances = node_distances.masked_fill(node_distances == 0, 1e6)




        #Temporary hack, to add the gnn without changing the rest of the code.
        gnn = gates.get("active_module", None)
        if gnn is None:
            raise ValueError("GNN gate not found in gates dictionary under key 'active_module'")

        #List[Batch, C, k]
        per_layer_feats = [all_gate_feats[n] for n in pre_weight_acts]
        #Concatenate C dimensions together, to get a tensor of shape (batch, sum_C, k)
        all_feats = torch.cat(per_layer_feats, dim=-2)
        #This ofcourse requires that the order is always the same, of both distances
        #and iteration and so fourth. Ive assumed that it is the case.
        gnn_out = gnn.forward_batched(
            x_batch=all_feats,
            dist_batch=node_distances,
        ).squeeze(-1)
        #Squeeze folds it into [batch, sum_C], instead of [batch, sum_C, 1]
        #The 1 is the output dimension of the gnn, which is always 1

        #This part still has a fairly large issue,
        #There is an outblock where the x + gnn_out was changed to just gnn_out.
        #The problem is that every neuron in the gnn layer, gets the exact same output
        #We should add another linear layer only for the x value, to distinguish them.
        #Likewise only adding x, does not currently give a percentage value between 0 and 1,
        #And is fundamentally different from the paper.
        offset = 0
        for param_name in pre_weight_acts:
            C = all_gate_outputs[param_name].shape[-1]
            layer_out = gnn_out[:, offset:offset + C]  # (batch, C)
            #Changed it to be just the gnn not as a residual, as a test. worked horribly.
            all_gate_outputs[param_name] =  all_gate_outputs[param_name] + layer_out
            offset += C


    for param_name, gate_output in all_gate_outputs.items():
        causal_importances[param_name] = lower_leaky_relu(gate_output)
        causal_importances_upper_leaky[param_name] = upper_leaky_relu(gate_output)

    return causal_importances, causal_importances_upper_leaky