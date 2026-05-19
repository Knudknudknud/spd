from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor


class QuantizationDataset:
    def __init__(
        self,
        n_features: int,
        feature_probability: float,
        device: str,
        label_type: Literal["scalar", "vector"],
        n_groups: int = 1,
        data_generation_type: Literal[
            "exactly_one_active", "at_least_zero_active"
        ] = "at_least_zero_active",
    ):
        """Sparse feature dataset for the residual MLP parity task.

        Args:
            n_features: Total number of features (control bits + data bits).
            feature_probability: Probability that a control bit is active when
                data_generation_type is 'at_least_zero_active'.
            device: Device to allocate tensors on.
            label_type: Type of labels to generate. Vector returns one label per group, scalar returns the sum of the vector labels.
            n_groups: Number of groups. The first n_groups features are control bits;
                the remaining n_features - n_groups are data bits.
            data_generation_type: How control bits are sampled.
        """
        # For now assume every group is the same length.
        assert n_features % n_groups == 0, "n_features must be divisible by n_groups"
        assert n_features / n_groups >= 3, (
            "To have a non-trivial parity, each group must have more than 2 features. "
            "The first n_groups features are the control bits."
        )

        self.n_features = n_features
        self.n_groups = n_groups
        self.feature_probability = feature_probability
        self.device = device
        self.data_generation_type = data_generation_type
        self.bits_per_group = (n_features - n_groups) // n_groups
        self.label_type = label_type

    def compute_labels(
        self,
        batch: Float[Tensor, "batch n_features"],
    ) -> Float[Tensor, "batch ..."]:

        batch_size = batch.shape[0]

        # First n_groups
        control_bits = batch[:, : self.n_groups]
        #Remaining n_features - n_groups are data bits
        data_bits = batch[:, self.n_groups :]

        # Reshape data into groups
        data_bits = data_bits.reshape(
            batch_size,
            self.n_groups,
            self.bits_per_group,
        )
        #sum over all bits in each group
        parity_sum = torch.einsum("b g k -> b g", data_bits)
        #for each entry in [batch, group], compute parity of each group
        parity_per_group = parity_sum % 2

        # Element wise product
        gated = control_bits * parity_per_group

        # Return depending on label type
        if self.label_type == "scalar":
            return torch.einsum("b g -> b", gated)

        elif self.label_type == "vector":
            return gated
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")
        
    def generate_batch(
        self,
        batch_size: int,
    ) -> tuple[Float[Tensor, "batch n_features"], Tensor]:

        # Create empty batch
        batch = torch.zeros(batch_size,self.n_features,device=self.device,
        )

        if self.data_generation_type == "exactly_one_active":

            #random int between 0 and n_groups-1, for each row in the batch. Shape: (batch_size,)
            active_indices = torch.randint(low=0,high=self.n_groups,size=(batch_size,),device=self.device,)

            # Create control-bit matrix, Shape: (batch_size, n_groups)
            control_bits = torch.zeros(batch_size,self.n_groups,device=self.device,)

            # Arranges indices to [0,1,2,3,4,5,..., batch_size-1] 
            row_indices = torch.arange(batch_size)

            #For batch element i, set control bit at active_indices[i] to 1
            control_bits[row_indices, active_indices] = 1.0

        elif self.data_generation_type == "at_least_zero_active":

            # Randomly activate each control bit independently
            random_values = torch.rand(
                batch_size,
                self.n_groups,
                device=self.device,
            )

            control_bits = (
                random_values < self.feature_probability
            )

        else:
            raise ValueError(
                f"Unknown data_generation_type: {self.data_generation_type}"
            )

        # Put the control_bits to be the first n_group columns
        # [Batches, n_features], so pick the first n_group columns and set them to control_bits
        batch[:, : self.n_groups] = control_bits


        #The remaining n_features - n_groups are data bits, which are random bits independent of the control bits
        random_data = torch.rand(batch_size,self.n_features - self.n_groups,device=self.device,)
        #sampled independently from Bernoulli(0.5)
        data_bits = (random_data < 0.5).float()

        # Put data bits after control bits
        batch[:, self.n_groups :] = data_bits

        labels = self.compute_labels(batch)

        return batch, labels
    def __len__(self) -> int:
        return 2**31


if __name__ == "__main__":
    dataset = QuantizationDataset(
        n_features=3,
        n_groups=1,
        feature_probability=0.5,
        device="cpu",
        data_generation_type="at_least_zero_active",
        label_type="vector",
    )


    print("Generating batch for at_least_zero_active:")
    batch, labels = dataset.generate_batch(batch_size=2)
    print("Batch shape:", batch.shape)
    print("Labels shape:", labels.shape)
    print("Batch:", batch)
    print("Labels:", labels)


    print("\nGenerating batch for exactly_one_active:")
    dataset.data_generation_type = "exactly_one_active"
    batch, labels = dataset.generate_batch(batch_size=2)
    print("Batch shape:", batch.shape)
    print("Labels shape:", labels.shape)
    print("Batch:", batch)
    print("Labels:", labels)


    print("\nGenerating batch for scalar labels:")
    dataset.label_type = "scalar"
    batch, labels = dataset.generate_batch(batch_size=1)
    print("Batch shape:", batch.shape)
    print("Labels shape:", labels.shape)
    print("Batch:", batch)
    print("Labels:", labels)