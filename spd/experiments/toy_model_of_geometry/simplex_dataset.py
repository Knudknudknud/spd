import math
import torch
from torch import Tensor


class SimplexDataset:
    def __init__(self, dimensions: list[int], device: str = "cpu",
                 data_generation_type: str = "at_least_zero_active",
                 feature_probability: float = 0.75):
        self.dimensions = dimensions  # e.g. [4, 2] -> 4-simplex and 2-simplex
        self.device = device
        self.data_generation_type = data_generation_type
        self.feature_probability = feature_probability

        # each k-simplex has (k+1) vertices in R^k -> k*(k+1) numbers
        self.group_sizes = [k * (k + 1) for k in dimensions]
        self.n_features = sum(self.group_sizes)
        self.output_dim = len(dimensions)

        # slice for each group in the flat row
        self.groups = []
        start = 0
        for size in self.group_sizes:
            self.groups.append(list(range(start, start + size)))
            start += size

    def __len__(self) -> int:
        return 2**31

    def _volume_from_vertices(self, vertices: Tensor) -> Tensor:
        """vertices: (..., k+1, k). Returns (...,) volumes."""
        edges = vertices[..., 1:, :] - vertices[..., :1, :]  # (..., k, k)
        gram = edges @ edges.transpose(-1, -2)  # (..., k, k)
        k = edges.shape[-2]
        return torch.sqrt(torch.clamp(torch.det(gram), min=0)) / math.factorial(k)

    def generate_batch(self, batch_size: int):
        batch = torch.zeros(batch_size, self.n_features, device=self.device)
        labels = torch.zeros(batch_size, self.output_dim, device=self.device)

        # figure out which rows have which groups active
        if self.data_generation_type == "at_least_zero_active":
            group_active = torch.rand(batch_size, len(self.dimensions), device=self.device) < self.feature_probability
        elif self.data_generation_type == "exactly_one_active":
            chosen = torch.randint(len(self.dimensions), (batch_size,), device=self.device)
            group_active = torch.zeros(batch_size, len(self.dimensions), dtype=torch.bool, device=self.device)
            group_active[torch.arange(batch_size), chosen] = True
        else:
            raise ValueError("not a valid data_generation_type for Simplex")

        for g_idx, (k, group) in enumerate(zip(self.dimensions, self.groups)):
            active_rows = group_active[:, g_idx].nonzero(as_tuple=True)[0]
            if active_rows.numel() == 0:
                continue

            num_active = active_rows.numel()

            # sample (k+1) vertices in R^k, uniform in [0,1]^k
            vertices = torch.rand(num_active, k + 1, k, device=self.device)

            # flatten vertices into the row slice for this group
            batch[active_rows[:, None], group] = vertices.reshape(num_active, -1)

            # compute labels
            labels[active_rows, g_idx] = self._volume_from_vertices(vertices)

        return batch, labels


if __name__ == "__main__":
    torch.manual_seed(67)

    # sanity checks against known formulas

    # 1. unit triangle (2-simplex) with vertices (0,0), (1,0), (0,1) -> area 0.5
    ds = SimplexDataset([2])
    vertices = torch.tensor([[[[0., 0.], [1., 0.], [0., 1.]]]])  # (1, 1, 3, 2)
    vol = ds._volume_from_vertices(vertices)
    print(f"unit triangle volume: {vol.item():.4f} (expected 0.5000)")
    assert abs(vol.item() - 0.5) < 1e-6

    # 2. unit tetrahedron (3-simplex) with vertices at origin + standard basis -> volume 1/6
    vertices = torch.tensor([[[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]])
    ds3 = SimplexDataset([3])
    vol = ds3._volume_from_vertices(vertices)
    print(f"unit tetrahedron volume: {vol.item():.4f} (expected {1/6:.4f})")
    assert abs(vol.item() - 1/6) < 1e-6

    # 3. degenerate simplex (all vertices coincident) -> volume 0
    vertices = torch.zeros(1, 1, 4, 3)
    vol = ds3._volume_from_vertices(vertices)
    print(f"degenerate volume: {vol.item():.4f} (expected 0.0000)")
    assert vol.item() < 1e-6

    # 4. end-to-end: generate a batch and verify shapes + label consistency
    ds = SimplexDataset([4, 2], feature_probability=1.0)  # force all groups active
    x, labels = ds.generate_batch(8)
    assert x.shape == (8, 4 * 5 + 2 * 3), f"unexpected batch shape: {x.shape}"
    assert labels.shape == (8, 2), f"unexpected labels shape: {labels.shape}"
    assert (labels > 0).all(), "all groups should be active with feature_probability=1.0"
    print(f"batch shape: {x.shape}, labels shape: {labels.shape}")
    print(f"sample labels:\n{labels}")

    # 5. ablated groups should have 0 labels
    ds = SimplexDataset([4, 2], feature_probability=0.0)  # force all ablated
    x, labels = ds.generate_batch(8)
    assert (labels == 0).all(), "all labels should be 0 when nothing is active"
    assert (x == 0).all(), "batch should be all zeros when nothing is active"
    print("ablation test passed")

    print("\nall tests passed")