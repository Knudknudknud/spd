import torch


class SphericalFeatureDataset:
    def __init__(self, ranks: list[int], feature_probability: float = 0.05,
                 device: str = "cpu"):
        self.ranks = ranks
        self.feature_probability = feature_probability
        self.device = device
        self.n_features = sum(ranks)

        # Build coordinate slices for each feature: [0], [1], [2,3], [4,5,6], ...
        self.groups = []
        start = 0
        for k in ranks:
            self.groups.append(list(range(start, start + k)))
            start += k

    def __len__(self) -> int:
        return 2**31

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = torch.zeros(batch_size, self.n_features, device=self.device)

        for group_indices in self.groups:
            k = len(group_indices)
            mask = (torch.rand(batch_size, 1, device=self.device) < self.feature_probability).float()

            if k == 1:
                vals = torch.rand(batch_size, 1, device=self.device)
            else:
                # Uniform on S^(k-1): normalize a Gaussian
                vals = torch.randn(batch_size, k, device=self.device)
                vals = vals / vals.norm(dim=1, keepdim=True)

            batch[:, group_indices] += vals * mask

        return batch, batch.clone()


if __name__ == "__main__":
    torch.manual_seed(67)

    # Small batch to eyeball
    ds = SphericalFeatureDataset(ranks=[2,2,2,2], feature_probability=0.3)
    x, y = ds.generate_batch(4)
    print("n_features =", ds.n_features)
    print("groups =", ds.groups)
    print("x =")
    print(x)
    print("x == y?", torch.equal(x, y))

    # Large batch to check properties
    ds = SphericalFeatureDataset(ranks=[2, 2, 2, 2], feature_probability=1.0)
    x, _ = ds.generate_batch(10_000)

    print("\nNorm check (all features always active, prob=1.0):")
    for i, group in enumerate(ds.groups):
        k = len(group)
        norms = x[:, group].norm(dim=1)
        print(f"  feature {i} (rank {k}): mean norm = {norms.mean():.4f}, "
              f"std = {norms.std():.2e}")

    print("\nSparsity check (feature_probability=0.1):")
    ds = SphericalFeatureDataset(ranks=[1, 1, 2, 3], feature_probability=0.1)
    x, _ = ds.generate_batch(10_000)
    for i, group in enumerate(ds.groups):   
        active = (x[:, group].abs().sum(dim=1) > 0).float().mean()
        print(f"  feature {i}: active fraction = {active:.4f} (expected 0.1)")