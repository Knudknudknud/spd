import math
import torch
from torch import Tensor


class SimplexDataset:
    def __init__(self, dimensions: list[int], device: str = "cpu",
                 data_generation_type: str = "at_least_zero_active",
                 feature_probability: float = 0.75,
                 degeneracy_eps: float = 1e-6,
                 max_resample_attempts: int = 100):
        self.dimensions = dimensions
        self.device = device
        self.data_generation_type = data_generation_type
        self.feature_probability = feature_probability
        self.degeneracy_eps = degeneracy_eps
        self.max_resample_attempts = max_resample_attempts

        self.group_sizes = [k * (k + 1) for k in dimensions]
        self.n_features = sum(self.group_sizes)
        self.output_dim = len(dimensions)

        self.groups = []
        start = 0
        for size in self.group_sizes:
            self.groups.append(list(range(start, start + size)))
            start += size

    def __len__(self) -> int:
        return 2**31

    def _edge_dets(self, vertices: Tensor) -> Tensor:
        """vertices: (..., k+1, k). Returns (...,) signed determinants of edge matrix."""
        edges = vertices[..., 1:, :] - vertices[..., :1, :]
        return torch.det(edges)

    def _volume_from_vertices(self, vertices: Tensor) -> Tensor:
        """vertices: (..., k+1, k). Returns (...,) volumes via |det| / k!."""
        edges = vertices[..., 1:, :] - vertices[..., :1, :]
        k = edges.shape[-2]
        return torch.abs(torch.det(edges)) / math.factorial(k)

    def _sample_valid_simplices(self, num: int, k: int) -> Tensor:
        """Sample `num` non-degenerate (k+1, k) simplices in [0,1]^k via rejection."""
        vertices = torch.rand(num, k + 1, k, device=self.device)

        for _ in range(self.max_resample_attempts):
            dets = self._edge_dets(vertices)
            bad = torch.abs(dets) < self.degeneracy_eps
            n_bad = int(bad.sum().item())
            if n_bad == 0:
                return vertices
            vertices[bad] = torch.rand(n_bad, k + 1, k, device=self.device)

        raise RuntimeError(
            f"failed to sample non-degenerate simplices after "
            f"{self.max_resample_attempts} attempts (k={k}, eps={self.degeneracy_eps})"
        )

    def generate_batch(self, batch_size: int):
        batch = torch.zeros(batch_size, self.n_features, device=self.device)
        labels = torch.zeros(batch_size, self.output_dim, device=self.device)

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
            vertices = self._sample_valid_simplices(num_active, k)

            batch[active_rows[:, None], group] = vertices.reshape(num_active, -1)
            labels[active_rows, g_idx] = self._volume_from_vertices(vertices)

        return batch, labels


if __name__ == "__main__":
    torch.manual_seed(67)

    # sanity: known formulas still work
    ds = SimplexDataset([2])
    vertices = torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]])
    assert abs(ds._volume_from_vertices(vertices).item() - 0.5) < 1e-6
    print("unit triangle: pass")

    ds3 = SimplexDataset([3])
    vertices = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
    assert abs(ds3._volume_from_vertices(vertices).item() - 1/6) < 1e-6
    print("unit tetrahedron: pass")

    # rejection actually rejects degenerate inputs
    degenerate = torch.zeros(1, 4, 3)
    dets = ds3._edge_dets(degenerate)
    assert torch.abs(dets).item() < 1e-6
    print("degenerate detection: pass")

    # end-to-end shapes
    ds = SimplexDataset([4, 2], feature_probability=1.0)
    x, labels = ds.generate_batch(8)
    assert x.shape == (8, 4 * 5 + 2 * 3)
    assert labels.shape == (8, 2)
    assert (labels > 0).all()
    print("batch shapes & all-positive labels: pass")

    # volume distribution: how spread out are the labels?
    print("\n--- volume distribution per dimension ---")
    for k in [2, 3, 4, 5]:
        ds_k = SimplexDataset([k], feature_probability=1.0)
        _, labels = ds_k.generate_batch(10000)
        vols = labels[:, 0]
        print(f"k={k}: min={vols.min():.2e}  median={vols.median():.2e}  "
              f"max={vols.max():.2e}  mean={vols.mean():.2e}  std={vols.std():.2e}")
        # log-spread tells us how many orders of magnitude the volumes span
        log_vols = torch.log10(vols.clamp(min=1e-30))
        print(f"      log10 range: [{log_vols.min():.2f}, {log_vols.max():.2f}]  "
              f"(spans {log_vols.max() - log_vols.min():.1f} orders of magnitude)")

    print("\nall tests passed")