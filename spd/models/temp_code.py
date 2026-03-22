
    # # GNN pass — runs once over all layers
    # if use_gnn:
    #     #had device issues, so just infer it...
    #     device = next(iter(all_gate_outputs.values())).device
    #     #Bidirectional edges
    #     edge_index = _construct_edge_index(all_gate_outputs, device)
    #     node_distances = _construct_node_distances(all_gate_outputs, device)

    #     #Forward only edges
    #     #edge_index = _construct_edge_index_forward_only(all_gate_outputs, device)
    #     #node_distances = _construct_node_distances_forward_only(all_gate_outputs, device)

    #     if not allow_same_layer_connections:
    #         edge_index = _remove_same_layer_edges(edge_index, node_distances)
    #         node_distances = node_distances.masked_fill(node_distances == 0, 1e6)

    #     gnn = gates.get("active_module", None)
    #     if gnn is None:
    #         raise ValueError("GNN gate not found in gates dictionary under key 'active_module'")

    #     # FIX: compute normalization only over valid (non-blocked) entries
    #     # Previously, 1e6 sentinel values dominated the row sums, squashing
    #     # all real distances to ~0 after division
    #     valid_mask = (node_distances < 1e5).float()
    #     valid_distances = node_distances * valid_mask
    #     normalization_matrix = valid_distances.sum(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(node_distances)



    #     # Per-sample features: each token gets its own node features
    #     # instead of averaging over the batch (which made the GNN output
    #     # a constant shared across all samples)
    #     first_name = next(iter(pre_weight_acts))
    #     leading_shape = all_gate_outputs[first_name].shape[:-1]  # e.g. (batch, pos)

    #     node_feats_per_sample = torch.cat([
    #         all_gate_outputs[n] for n in pre_weight_acts
    #     ], dim=-1)  # (..., total_nodes)

    #     total_nodes = node_feats_per_sample.shape[-1]
    #     S = node_feats_per_sample[..., 0].numel()  # product of leading dims
    #     node_feats_flat = torch.sigmoid(node_feats_per_sample.reshape(S, total_nodes))

    #     # fs[0] is a scalar→scalar MLP, batch all S*N scalars at once
    #     fs_input = node_feats_flat.reshape(-1, 1)                        # (S*N, 1)
    #     fs_out = gnn.fs[0](fs_input)                                     # (S*N, out_channels)
    #     fs_out = fs_out.reshape(S, total_nodes, gnn.out_channels)        # (S, N, out)

    #     # rho processes distances — computed ONCE, same for all samples
    #     m_dist = gnn.rho(node_distances.flatten().view(-1, 1))
    #     m_dist = m_dist.view(total_nodes, total_nodes, gnn.out_channels)
    #     if gnn.normalize_rho:
    #         m_dist = m_dist / normalization_matrix.unsqueeze(-1)

    #     # Batched matmul: rho(distances) @ fs(features) for each sample
    #     fx_perm = fs_out.permute(0, 2, 1).unsqueeze(-1)                  # (S, out, N, 1)
    #     m_dist_perm = m_dist.permute(2, 0, 1).unsqueeze(0)               # (1, out, N, N)
    #     mf = torch.matmul(m_dist_perm, fx_perm).squeeze(-1)              # (S, out, N)
    #     gnn_out = mf.permute(0, 2, 1)                                    # (S, N, out)

    #     # Reshape back to (..., total_nodes) and split per layer
    #     gnn_out = gnn_out.squeeze(-1).reshape(*leading_shape, total_nodes)

    #     offset = 0
    #     for param_name in pre_weight_acts:
    #         C = all_gate_outputs[param_name].shape[-1]
    #         layer_out = gnn_out[..., offset:offset + C]  # (..., C)
    #         #Residual, f(x) + x - might be dumb?
    #         all_gate_outputs[param_name] = all_gate_outputs[param_name] + layer_out
    #         offset += C

    # for param_name, gate_output in all_gate_outputs.items():
    #     causal_importances[param_name] = lower_leaky_relu(gate_output)
    #     causal_importances_upper_leaky[param_name] = upper_leaky_relu(gate_output)

    # return causal_importances, causal_importances_upper_leaky
    # GNN pass — runs once over all layers