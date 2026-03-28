The gnn tends to keep all components alive?
Yes, within the same layer. Look at the GNAN computation:
gnn_out[i] = Σ_j  rho(d_ij) * fs(x_j)
Two nodes i1 and i2 in the same layer have identical distances to every other node. So:
gnn_out[i1] = Σ_j  rho(d_{i1,j}) * fs(x_j)
gnn_out[i2] = Σ_j  rho(d_{i2,j}) * fs(x_j)
These are the same sum — d_{i1,j} = d_{i2,j} for all j because i1 and i2 are in the same layer. Every component in the same layer gets the exact same GNAN correction.
That's the fundamental problem. The GNAN can't distinguish between components within a layer. It can only say "all 200 components in layer 0 get +0.3."





The simplest fix: add a self-processing path that doesn't go through the aggregation. A small MLP that processes each node's own features independently:
python# In TensorGNAN.__init__, add:
self.self_mlp = nn.Sequential(
    nn.Linear(in_channels, hidden_channels),
    nn.ReLU(),
    nn.Linear(hidden_channels, out_channels),
)
python# In forward_batched, after computing mf:
mf = mf.sum(dim=3).permute(0, 2, 1)           # (S, N, C) — cross-layer term, same within layer

# Self term — different per node because each has different features
self_out = self.self_mlp(x_batch)               # (S, N, out_channels) — unique per node

return mf + self_out                            # cross-layer context + within-layer discrimination
```

Now the output is:
```
gnn_out[i] = self_mlp(x_i) + Σ_j rho(d_ij) * fs(x_j)
              ↑                    ↑
         unique per node     same for same-layer nodes
The self_mlp sees each node's own activation and can distinguish "component 5 is strongly active" from "component 150 is inactive" — even within the same layer. The aggregation sum provides cross-layer context on top.