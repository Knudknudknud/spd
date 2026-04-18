from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset


class PairedResidualMLPDataset(ResidualMLPDataset):
    """Like ResidualMLPDataset, but some features come in 2D circle pairs.

    For each pair (i, j):
      Input:  (x_i, x_j) = r * (cos t, sin t), r ~ U[0,1], t ~ U[0, 2pi)
              shared activation mask between the two coords
      Label:  y_{ij} = x_{ij} + ReLU(u . x_{ij}) * u
              where u is a fixed random unit vector in R^2 per pair
    """

    def __init__(
        self,
        n_features: int,
        feature_probability: float,
        device: str,
        pair_indices: list[list[int]],
        label_fn_seed: int | None = None,
        **kwargs,
    ):
        super().__init__(
            n_features=n_features,
            feature_probability=feature_probability,
            device=device,
            calc_labels=True,
            label_type="act_plus_resid",
            act_fn_name="relu",
            label_fn_seed=label_fn_seed,
            **kwargs,
        )
        self.pair_indices = pair_indices

        # One fixed random unit vector per pair
        gen = torch.Generator(device=device)
        if label_fn_seed is not None:
            gen.manual_seed(label_fn_seed + 1)
        u = torch.randn(len(pair_indices), 2, generator=gen, device=device)
        self.pair_directions = u / u.norm(dim=1, keepdim=True)

    def generate_batch(self, batch_size: int):
        batch, labels = super().generate_batch(batch_size)

        for p, (i, j) in enumerate(self.pair_indices):
            active = (torch.rand(batch_size, device=self.device)
                      < self.feature_probability).float()
            r = torch.rand(batch_size, device=self.device)
            t = 2 * torch.pi * torch.rand(batch_size, device=self.device)

            xi = active * r * torch.cos(t)
            xj = active * r * torch.sin(t)
            batch[:, i] = xi
            batch[:, j] = xj

            u = self.pair_directions[p]
            x_pair = torch.stack([xi, xj], dim=1)
            proj = x_pair @ u
            y_pair = x_pair + F.relu(proj).unsqueeze(1) * u.unsqueeze(0)
            labels[:, i] = y_pair[:, 0]
            labels[:, j] = y_pair[:, 1]

        return batch, labels
