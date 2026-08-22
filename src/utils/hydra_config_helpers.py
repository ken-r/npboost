"""
This module provides helper functions for loading and instantiating Hydra configurations.
"""

import os
import logging
from typing import List, Optional, Tuple, Union

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from src.data.data_layout import CONFIG_FOLDER
from src.data.datasets import BundleFactory, DataBundle
from src.utils.helpers import set_seed

logger = logging.getLogger(__name__)


def disable_wandb_logging(cfg: DictConfig) -> DictConfig:
    """
    Disable W&B logging.

    """
    if OmegaConf.select(cfg, "wandb.enabled") is not None:
        cfg.wandb.enabled = False

    if OmegaConf.select(cfg, "model.wandb_enabled") is not None:
        cfg.model.wandb_enabled = False

    return cfg



def instantiate_omega_conf(yaml_path, **additional_kwargs):
    """Instantiates an object from an OmegaConf YAML file."""
    config = OmegaConf.load(yaml_path, **additional_kwargs)
    return instantiate(config, **additional_kwargs)


def get_synthetic_data_provider(experiment_name, fixed_effect=None, random_effect=None, noise=None, seeds: list=None, additional_overrides=None):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_FOLDER)):
        overrides = [
            "data=synthetic/zero_data", # We start with zero data so that our fixed, random and noise effects are just added on top of that.
            f"data.experiment.name={experiment_name}"
        ]
        if fixed_effect is not None:
            overrides.append(f"data/synthetic/fixed_effect={fixed_effect}")
        if random_effect is not None:
            overrides.append(f"data/synthetic/random_effect={random_effect}")
        if noise is not None:
            overrides.append(f"data/synthetic/noise={noise}")

        if seeds is not None:
            overrides.append(f"data.generation_seed_list={seeds}")

        if additional_overrides is not None:
            overrides.extend(additional_overrides)

        cfg = compose(config_name="config", overrides=overrides)
        provider = instantiate(cfg.data.provider)
    return provider


def get_experiment_data_provider(experiment_name, experiment_type, additional_overrides=None):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_FOLDER)):
        overrides = [
            f"data={experiment_type}/{experiment_name}",
        ]

        if additional_overrides is not None:
            overrides.extend(additional_overrides)

        cfg = compose(config_name="config", overrides=overrides)
        provider = instantiate(cfg.data.provider)
    return provider


def ensure_available_device(cfg: DictConfig) -> DictConfig:
    """
    Keep notebook/script configs from accidentally requesting CUDA on CPU-only machines.

    """
    device = OmegaConf.select(cfg, "device")
    model_device = OmegaConf.select(cfg, "model.device")
    requested_device = model_device or device

    if requested_device is None:
        return cfg

    if str(requested_device).lower().startswith("cuda") and not torch.cuda.is_available():
        if device is not None:
            cfg.device = "cpu"
        if model_device is not None:
            cfg.model.device = "cpu"

    return cfg

def load_hydra_config(
    overrides: Optional[List[str]] = None,
    config_name: str = "config",
    disable_wandb: bool = True
) -> DictConfig:
    """
    Programmatically loads the Hydra configuration with custom overrides.
    Resolves variable interpolations and handles GlobalHydra restarts safely.

    Args:
        overrides: List of override strings (e.g. ['model=npboost', 'task=few_shot'])
        config_name: Main configuration filename (default: "config")
        disable_wandb: If True, forces W&B logging to be disabled.

    Returns:
        cfg: The fully resolved OmegaConf DictConfig.
    """
    if overrides is None:
        overrides = []

    # Reset GlobalHydra instance if it is already initialized (essential for Jupyter notebooks)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_dir_str = str(CONFIG_FOLDER.resolve())
    logger.info(f"Initializing Hydra from config directory: {config_dir_str}")

    with initialize_config_dir(version_base=None, config_dir=config_dir_str):
        cfg = compose(config_name=config_name, overrides=overrides)
        OmegaConf.resolve(cfg)

        if disable_wandb:
            disable_wandb_logging(cfg)

        ensure_available_device(cfg)

        return cfg

