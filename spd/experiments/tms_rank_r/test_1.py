"""
Dependent residual-MLP dataset, in the spirit of Braun et al. 2025
(toy model of compressed computation), but with feature pairs that live on
a 2D circle instead of being independent scalars.

Model the dataset is built for:
    h0 = W_E @ x                      (residual stream, dim d_resid)
    h1 = h0 + W_out @ ReLU(W_in @ h0) (one residual MLP layer)
    y  = W_U @ h1   with W_U = W_E^T

Target: y_i = x_i + ReLU(x_i)  for each feature i.

Standard Braun setup: each x_i is an independent sparse scalar in [-1, 1].

Our extension: some features come in PAIRS that are constrained to a circle.
For a pair (i, j), when active we set
    x_i = r * cos(theta),  x_j = r * sin(theta)
with theta ~ Uniform[0, 2*pi) and r ~ Uniform[0, 1].
The two coordinates are dependent (they lie on a 1D curve in 2D when r is
fixed) and uncorrelated. Each pair forms a rank-2 feature block whose
covariance has rank 2 but support is geometrically constrained.

This dataset only generates inputs and the target y_i = x_i + ReLU(x_i).
The TMS / residual-MLP architecture itself is built elsewhere.
"""
import torch


class DependentResidualMLPDataset:
    def __init__(self, n_scalars: int = 96, n_pairs: int = 2,
                 feature_probability: float = 0.05, device: str = "cpu"):
        self.n_scalars = n_scalars
        self.n_pairs = n_pairs
        self.n_features = n_scalars + 2 * n_pairs
        self.feature_probability = feature_probability
        self.device = device

        # Coordinate layout: scalars first, then pair blocks.
        # scalar features: indices 0 .. n_scalars-1
        # pair blocks:     [n_scalars, n_scalars+1], [n_scalars+2, n_scalars+3], ...
        self.scalar_indices = list(range(n_scalars))
        self.pair_indices = [
            [n_scalars + 2*p, n_scalars + 2*p + 1] for p in range(n_pairs)
        ]

    def __len__(self) -> int:
        return 2**31

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(batch_size, self.n_features, device=self.device)

        # Scalar features: sparse uniform in [-1, 1]
        if self.n_scalars > 0:
            mask = (torch.rand(batch_size, self.n_scalars, device=self.device)
                    < self.feature_probability).float()
            vals = 2 * torch.rand(batch_size, self.n_scalars, device=self.device) - 1
            x[:, self.scalar_indices] = mask * vals

        # Pair features: each pair fires together on a circle in 2D
        for pair in self.pair_indices:
            mask = (torch.rand(batch_size, 1, device=self.device)
                    < self.feature_probability).float()  # (B, 1) shared by both
            theta = 2 * torch.pi * torch.rand(batch_size, 1, device=self.device)
            r = torch.rand(batch_size, 1, device=self.device)
            xi = r * torch.cos(theta)
            xj = r * torch.sin(theta)
            x[:, pair[0]:pair[0]+1] = mask * xi
            x[:, pair[1]:pair[1]+1] = mask * xj

        # Target: y_i = x_i + ReLU(x_i)   applied elementwise
        y = x + torch.relu(x)
        return x, y


if __name__ == "__main__":
    torch.manual_seed(0)

    ds = DependentResidualMLPDataset(n_scalars=96, n_pairs=2,
                                     feature_probability=0.05)
    x, y = ds.generate_batch(8)
    print("n_features =", ds.n_features)
    print("scalar block: indices", ds.scalar_indices[:5], "...")
    print("pair blocks :", ds.pair_indices)
    print()

    # Verify pair structure on a large batch with prob=1
    ds2 = DependentResidualMLPDataset(n_scalars=0, n_pairs=1,
                                      feature_probability=1.0)
    x2, _ = ds2.generate_batch(10_000)
    pair = ds2.pair_indices[0]
    norms = x2[:, pair].norm(dim=1)
    print("Pair norm statistics (should be ~Uniform[0,1] since r is uniform):")
    print(f"  mean={norms.mean():.4f}, std={norms.std():.4f}, "
          f"min={norms.min():.4f}, max={norms.max():.4f}")
    cov = (x2[:, pair].T @ x2[:, pair]) / x2.shape[0]
    print("Empirical 2x2 covariance of the pair (should be near-diagonal):")
    print(cov)

    # Verify y = x + ReLU(x)
    print("\nTarget check: y == x + relu(x)?",
          torch.allclose(y, x + torch.relu(x)))