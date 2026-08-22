"""CLI wrapper for dense TabICL prediction generation."""

from __future__ import annotations

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

IN_CONTEXT_TASK_NAME = "in_context"
FEW_SHOT_TASK_NAME = "few_shot"
TABICL_MODEL_NAMES = ["tabicl_taskwise", "tabicl_pooled"]
GROUP_ID_OFFSET = 100000


def _cfg_list(cfg: DictConfig, key: str, default: list[str]) -> list[str]:
    value = cfg.get(key, default)
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        return [value]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _validate_choices(values: list[str], valid_values: list[str], field_name: str) -> None:
    invalid_values = sorted(set(values) - set(valid_values))
    if invalid_values:
        raise ValueError(
            f"Invalid {field_name}: {invalid_values}. "
            f"Expected values from {valid_values}."
        )


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )
    from src.evaluation.dense_tabicl_predictions import generate_dense_tabicl_prediction_frames

    tasks = _cfg_list(cfg, "tasks", ["all"])
    model_names = _cfg_list(cfg, "model_names", TABICL_MODEL_NAMES)
    _validate_choices(tasks, [IN_CONTEXT_TASK_NAME, FEW_SHOT_TASK_NAME, "all"], "tasks")
    _validate_choices(model_names, TABICL_MODEL_NAMES, "model_names")

    generate_dense_tabicl_prediction_frames(
        data_name=cfg.get("data_name", "gp_gaussian"),
        fixed_effect_name=cfg.get("fixed_effect_name", "steps_1D"),
        split_seed=int(cfg.get("split_seed", 0)),
        tasks=tasks,
        model_names=model_names,
        n_tasks=int(cfg.get("n_tasks", 3)),
        n_support=int(cfg.get("n_support", 20)),
        n_query=int(cfg.get("n_query", 400)),
        group_id_offset=int(cfg.get("group_id_offset", GROUP_ID_OFFSET)),
        interval_level=float(cfg.get("interval_level", 0.95)),
        n_estimators=int(cfg.get("n_estimators", 8)),
        n_jobs=int(cfg.get("n_jobs", 1)),
        save=True,
        overwrite=bool(cfg.get("overwrite", False)),
    )


if __name__ == "__main__":
    main()
