"""Trains a residual linear model on one-hot input vectors."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

import einops
import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from spd.data_utils import DatasetGeneratedDataLoader
from spd.experiments.toy_model_of_quantization.models import QuantizationMLP, QuantizationMLPConfig
from spd.experiments.toy_model_of_quantization.quantization_dataset import QuantizationDataset
from spd.log import logger
from spd.utils import get_lr_schedule_fn, set_seed
from spd.wandb_utils import init_wandb

wandb.require("core")


class QuantizationTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None  # The name of the wandb project (if None, don't log to wandb)
    seed: int = 0
    quantization_mlp_config: QuantizationMLPConfig
    loss_type: Literal["mse", "bce"] = "mse" 
    feature_probability: PositiveFloat
    importance_val: float | None = None
    data_generation_type: Literal[
        "exactly_one_active",  "at_least_zero_active"
    ] = "at_least_zero_active"
    label_type: Literal["scalar", "vector"] = "vector"
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    lr: PositiveFloat
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    n_batches_final_losses: PositiveInt = 1


def loss_function(
    out: Float[Tensor, "batch n_features"] | Float[Tensor, "batch d_embed"],
    labels: Float[Tensor, "batch n_features"],
    config: QuantizationTrainConfig,
) -> Float[Tensor, "batch n_features"] | Float[Tensor, "batch d_embed"]:
    
    if config.loss_type == "mse":
        
        loss = (out - labels) ** 2
    elif config.loss_type == "bce":
        loss = F.binary_cross_entropy_with_logits(out, labels, reduction="none")
    else:
        raise ValueError(f"Invalid loss_type: {config.loss_type}")
    return loss


def train(
    config: QuantizationTrainConfig,
    model: QuantizationMLP,
    trainable_params: list[nn.Parameter],
    dataloader: DatasetGeneratedDataLoader[
        tuple[
            Float[Tensor, "batch n_features"],
            Float[Tensor, "batch n_features"],
        ]
    ],
    device: str,
    out_dir: Path,
    run_name: str,
) -> Float[Tensor, ""]:
    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, name=run_name)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = out_dir / "quantization_mlp_train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    logger.info(f"Saved config to {config_path}")
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")

    # Save the coefficients used to generate the labels
    assert isinstance(dataloader.dataset, QuantizationDataset)

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.01)

    # Add this line to get the lr_schedule_fn
    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule)

    pbar = tqdm(range(config.steps), total=config.steps)
    for step, (batch, labels) in zip(pbar, dataloader, strict=False):
        if step >= config.steps:
            break

        # Add this block to update the learning rate
        current_lr = config.lr * lr_schedule_fn(step, config.steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        
        optimizer.zero_grad()

        batch: Float[Tensor, "batch n_features"] = batch.to(device)
        labels: Float[Tensor, "batch n_features"] = labels.to(device)
        out = model(batch)
        loss: Float[Tensor, "batch n_features"] = loss_function(
            out, labels, config
        )
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        if step % config.print_freq == 0:
            with torch.no_grad():
                # per-group loss

                # per-group accuracy (thresholded)
                preds = (out > 0).float()
                acc_per_group = (preds == labels).float().mean(dim=0)  # [n_groups] 

            tqdm.write(
                f"step {step}: loss={loss.item():.2e}, lr={current_lr:.2e}, "
                f"acc_mean={acc_per_group.mean().item():.3f}")
            if config.wandb_project:
                wandb.log({"loss": loss.item(), "lr": current_lr}, step=step)

    model_path = out_dir / "quantization.pth"
    torch.save(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    print(f"Saved model to {model_path}")

    # Calculate final losses by averaging many batches
    final_losses = []
    for _ in range(config.n_batches_final_losses):
        batch, labels = next(iter(dataloader))
        batch = batch.to(device)
        labels = labels.to(device)
        out = model(batch)
        loss = loss_function(out, labels, config)
        loss = loss.mean()
        final_losses.append(loss)
    final_losses = torch.stack(final_losses).mean().cpu().detach()
    print(f"Final losses: {final_losses.numpy()}")
    return final_losses


def run_train(config: QuantizationTrainConfig, device: str) -> Float[Tensor, ""]:
    cfg = config.quantization_mlp_config
    
    if config.loss_type == "bce" and config.label_type == "scalar":
        raise ValueError("BCE with scalar, is not binary in its current version, so use MSE instead.")
    run_name = (
        f"quant_mlp_d-model{cfg.d_hid}_"
        f"_seed{config.seed}"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"

    if config.label_type == "scalar":
        d_out = 1
    elif config.label_type == "vector":
        d_out = cfg.n_groups    
    model = QuantizationMLP(config=cfg, d_out=d_out).to(device)



    dataset = QuantizationDataset(
        n_features=cfg.n_features,
        n_groups=cfg.n_groups,
        feature_probability=config.feature_probability,
        device=device,
        data_generation_type=config.data_generation_type,
        label_type=config.label_type,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    final_losses = train(
        config=config,
        model=model,
        trainable_params=[p for p in model.parameters() if p.requires_grad],
        dataloader=dataloader,
        device=device,
        out_dir=out_dir,
        run_name=run_name,
    )
    return final_losses


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = QuantizationTrainConfig(
        wandb_project="spd-train-quantization-mlp",
        seed=0,
        quantization_mlp_config=QuantizationMLPConfig(
            n_features=12,
            n_groups=4,
            d_hid=4,
            in_bias=False,
            out_bias=False,
        ),
    
        loss_type="bce",
        feature_probability=0.25,
        importance_val=1,
        data_generation_type="at_least_zero_active",
        label_type="vector",
        batch_size=2048,
        steps=10000,  
        print_freq=1000,
        lr=3e-3,
        lr_schedule="cosine",
    )

    set_seed(config.seed)

    run_train(config, device)

