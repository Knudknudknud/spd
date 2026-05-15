import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import einops
import torch
import torch.nn.functional as F
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.log import logger
from spd.module_utils import init_param_
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class QuantizationPaths(BaseModel):
    """Paths to output files from a QuantizationMLP training run."""

    quantization_train_config: Path
    checkpoint: Path


class QuantizationMLPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_features: PositiveInt
    n_groups: PositiveInt
    d_hid: PositiveInt
    in_bias: bool
    out_bias: bool

class QuantizationMLP(nn.Module):
    def __init__(self, config: QuantizationMLPConfig, d_out: int):
        super().__init__()
        self.config = config
        self.d_out = d_out
        self.act_fn = nn.ReLU()
        self.mlp_in = nn.Linear(config.n_features, config.d_hid, bias=config.in_bias)
        self.mlp_out = nn.Linear(config.d_hid, d_out, bias=config.out_bias)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        mid = self.act_fn(self.mlp_in(x))
        return self.mlp_out(mid)

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> QuantizationPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run)

        run_dir = fetch_wandb_run_dir(run.id)

        quantization_train_config_path = download_wandb_file(
            run, run_dir, "quantization_mlp_train_config.yaml"
        )
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        logger.info(f"Downloaded checkpoint from {checkpoint_path}")
        return QuantizationPaths(
            quantization_train_config=quantization_train_config_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    def from_pretrained(
        cls, path: ModelPath
    ) -> tuple["QuantizationMLP", dict[str, Any]]:
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            paths = QuantizationPaths(
                quantization_train_config=Path(path).parent / "quantization_mlp_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.quantization_train_config) as f:
            quantization_train_config_dict = yaml.safe_load(f)

        quantization_config = QuantizationMLPConfig(
            **quantization_train_config_dict["quantization_mlp_config"]
        )
        label_type = quantization_train_config_dict["label_type"]
        d_out = 1 if label_type == "scalar" else quantization_config.n_groups

        quantization_mlp = cls(config=quantization_config, d_out=d_out)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        quantization_mlp.load_state_dict(params)

        return quantization_mlp, quantization_train_config_dict
