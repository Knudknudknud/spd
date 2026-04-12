import torch

class SharedCircuitDataset:
    def __init__(self, n_features: int = 6, group_size: int = 3, stride: int = 3,
                 feature_probability: float = 0.05, device: str = "cpu", n_independent: int = 0):
        self.n_features = n_features
        self.group_size = group_size
        self.feature_probability = feature_probability
        self.device = device
        self.n_independent = n_independent

        # Build overlapping groups: [0,1,2], [1,2,3], [2,3,4], ...
        self.groups = []
        for start in range(0, n_features - group_size + 1, stride):
            self.groups.append(list(range(start, start + group_size)))
        self.n_groups = len(self.groups)

    def __len__(self) -> int:
        return 2**31

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = torch.zeros(batch_size, self.n_features, device=self.device)

        # For each sample, each group fires independently with feature_probability
        for g, group_indices in enumerate(self.groups):
            mask = (torch.rand(batch_size, 1, device=self.device) < self.feature_probability).float()
            vals = torch.rand(batch_size, len(group_indices), device=self.device)
            batch[:, group_indices] += vals * mask

        return batch, batch.clone()