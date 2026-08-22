"""Refit selected runs and optionally save models plus prediction artifacts.

The script starts from the parameter CSVs written by the result-reporting code.
For the selected model/task/seed it rebuilds a local Hydra config and applies
the CSV hyperparameters. The original W&B config is only loaded when explicitly
requested. Saved artifacts are meant for publication plots and auditing.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import logging
from pathlib import Path
from statistics import NormalDist

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.evaluation.model_predictions import GroupedModelPrediction, ModelPrediction
from src.constants import (VALIDATION_SPLIT_NAME, TEST_SPLIT_NAME, IN_CONTEXT_TASK_NAME, FEW_SHOT_TASK_NAME,
                           TUNED_HYPERPARAMS_PER_MODEL)
from src.utils.helpers import set_seed
from src.data.datasets import BundleFactory, ModelDataAdapter
from src.evaluation.crps import crps_eval
from src.data.data_layout import (
    ADDITIONAL_RESULT_FOLDER,
    RESULT_FOLDER,
    PROJECT_ROOT,
    RealDataLayout,
    SyntheticDataLayout,
)

from src.data.datasets import BaseDataset
from src.utils.helpers import isolated_torch_rng

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)

DEFAULT_PREDICTION_INTERVAL_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_PREDICTION_INTERVAL_SAMPLES = None

# Saved labels are kept stable for existing prediction artifacts.
INTERVAL_METHOD_SAMPLE_QUANTILE = "sample_quantile"
INTERVAL_METHOD_CRPS_SAMPLE_QUANTILE = "crps_sample_quantile"
INTERVAL_METHOD_NORMAL_MEAN_STD = "gaussian_mean_std"
SAMPLED_INTERVAL_METHODS = {
    INTERVAL_METHOD_SAMPLE_QUANTILE,
    INTERVAL_METHOD_CRPS_SAMPLE_QUANTILE,
}

CSV_RUN_METADATA_COLUMNS = {
    "task_name",
    "data_split_seed",
    "fixed_effect_type",
    "feature_dimension",
    "best_validation_score",
    "run_id",
    "test_rmse",
    "test_crps",
    "best_boosting_round",
    "epochs_trained",
}


def _json_ready(value):
    """Convert NumPy containers and scalars into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _series_value_if_present(row: pd.Series, key: str):
    if key not in row or _is_missing_csv_value(row[key]):
        return None
    return _json_ready(row[key])


def _extract_metric_fields(mapping) -> dict:
    metric_keys = [
        "best_validation_score",
        "test_rmse",
        "test_crps",
        "best_boosting_round",
        "epochs_trained",
    ]
    metrics = {}
    for key in metric_keys:
        if isinstance(mapping, pd.Series):
            value = _series_value_if_present(mapping, key)
        elif mapping is not None and key in mapping:
            value = _json_ready(mapping[key])
        else:
            value = None
        if value is not None:
            metrics[key] = value
    return metrics


def _initial_run_info_from_params(row: pd.Series) -> dict:
    """Return the compact performance summary available in the params CSV."""
    info = {
        "validation_score": _json_ready(row["best_validation_score"]),
        "test_rmse": _json_ready(row["test_rmse"]),
    }
    if pd.notna(row["test_crps"]):
        info["test_crps"] = _json_ready(row["test_crps"])
    return info


def _wandb_run_reference(wandb_run, wandb_project_name: str, wandb_run_id: str) -> dict:
    raw_run_path = getattr(wandb_run, "path", None)
    run_path = (
        "/".join(str(part) for part in raw_run_path)
        if isinstance(raw_run_path, (list, tuple))
        else raw_run_path
    )
    return {
        "project": wandb_project_name,
        "run_id": wandb_run_id,
        "path": run_path or f"{wandb_project_name}/{wandb_run_id}",
        "url": getattr(wandb_run, "url", None),
        "name": getattr(wandb_run, "name", None),
    }


def _distribution_device(distribution: torch.distributions.Distribution) -> torch.device | None:
    try:
        mean = distribution.mean
    except NotImplementedError:
        return None
    return mean.device if isinstance(mean, torch.Tensor) else None


def _sample_distribution(
    distribution: torch.distributions.Distribution,
    sample_shape: torch.Size,
    seed: int,
) -> torch.Tensor:
    with isolated_torch_rng(seed, device=_distribution_device(distribution)):
        return distribution.sample(sample_shape)


def _interval_column_names(nominal_level: float) -> tuple[str, str]:
    suffix = f"{float(nominal_level):.2f}"
    return f"lower_{suffix}", f"upper_{suffix}"


def _has_group_distributions(predictions) -> bool:
    return (
        isinstance(predictions, GroupedModelPrediction)
        and predictions.predictive_distributions is not None
        and len(predictions.predictive_distributions) > 0
    )


def _quantiles_by_group(
    group_ids: np.ndarray,
    samples_by_group: dict,
    nominal_levels: tuple[float, ...],
) -> dict[str, np.ndarray]:
    n_rows = len(group_ids)
    interval_columns = {
        column: np.full(n_rows, np.nan, dtype=np.float32)
        for level in nominal_levels
        for column in _interval_column_names(level)
    }

    for group_id, samples in samples_by_group.items():
        group_mask = group_ids == group_id
        if not np.any(group_mask):
            continue

        samples_array = samples.detach().cpu().numpy() if isinstance(samples, torch.Tensor) else np.asarray(samples)
        for level in nominal_levels:
            alpha = 1.0 - level
            lower_col, upper_col = _interval_column_names(level)
            interval_columns[lower_col][group_mask] = np.quantile(
                samples_array,
                q=alpha / 2,
                axis=0,
            ).astype(np.float32)
            interval_columns[upper_col][group_mask] = np.quantile(
                samples_array,
                q=1 - alpha / 2,
                axis=0,
            ).astype(np.float32)

    return interval_columns


def _sample_interval_quantiles(
    predictions: GroupedModelPrediction,
    nominal_levels: tuple[float, ...],
    n_decoder_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    pred = predictions.reorder_to_original()
    group_ids = np.asarray(pred.group_ids).reshape(-1)
    samples_by_group = {}
    for group_id, distribution in pred.predictive_distributions.items():
        raw_samples = _sample_distribution(
            distribution,
            torch.Size([n_decoder_samples]),
            seed=seed,
        )
        # (S, G, B, T, 1) -> (S*G, T)
        flattened_samples = (
            torch.flatten(raw_samples, start_dim=0, end_dim=1)
            .squeeze(-1)
            .squeeze(1)
            .detach()
        )
        samples_by_group[group_id] = flattened_samples

    return _quantiles_by_group(group_ids, samples_by_group, nominal_levels)


def _sample_interval_quantiles_from_crps_samples(
    predictions: GroupedModelPrediction,
    nominal_levels: tuple[float, ...],
    crps_samples_by_group: dict,
) -> dict[str, np.ndarray]:
    pred = predictions.reorder_to_original()
    group_ids = np.asarray(pred.group_ids).reshape(-1)
    return _quantiles_by_group(group_ids, crps_samples_by_group, nominal_levels)


def _gaussian_interval_quantiles(
    pred: ModelPrediction | GroupedModelPrediction,
    nominal_levels: tuple[float, ...],
) -> dict[str, np.ndarray]:
    mean = np.asarray(pred.mean).reshape(-1).astype(np.float32)
    std = np.clip(np.asarray(pred.std).reshape(-1), a_min=0.0, a_max=None).astype(np.float32)
    interval_columns = {}
    for level in nominal_levels:
        z_score = NormalDist().inv_cdf(0.5 + float(level) / 2)
        lower_col, upper_col = _interval_column_names(level)
        interval_columns[lower_col] = (mean - z_score * std).astype(np.float32)
        interval_columns[upper_col] = (mean + z_score * std).astype(np.float32)
    return interval_columns


def _prediction_interval_quantiles(
    predictions: ModelPrediction | GroupedModelPrediction,
    nominal_levels: tuple[float, ...],
    n_decoder_samples: int,
    seed: int,
    crps_samples_by_group: dict | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    levels = tuple(float(level) for level in nominal_levels)
    if any(level <= 0.0 or level >= 1.0 for level in levels):
        raise ValueError("All prediction interval levels must lie strictly between 0 and 1.")
    if n_decoder_samples < 1:
        raise ValueError("prediction_interval_samples must be at least 1.")

    pred = predictions.reorder_to_original() if isinstance(predictions, GroupedModelPrediction) else predictions
    if _has_group_distributions(predictions):
        if crps_samples_by_group is not None:
            return _sample_interval_quantiles_from_crps_samples(
                predictions=predictions,
                nominal_levels=levels,
                crps_samples_by_group=crps_samples_by_group,
            ), INTERVAL_METHOD_CRPS_SAMPLE_QUANTILE
        return _sample_interval_quantiles(
            predictions=predictions,
            nominal_levels=levels,
            n_decoder_samples=n_decoder_samples,
            seed=seed,
        ), INTERVAL_METHOD_SAMPLE_QUANTILE
    return _gaussian_interval_quantiles(pred, levels), INTERVAL_METHOD_NORMAL_MEAN_STD


def _prediction_frame(
    predictions: ModelPrediction | GroupedModelPrediction,
    target_data: BaseDataset,
    split_name: str,
    interval_levels: tuple[float, ...],
    interval_samples: int,
    interval_seed: int = 0,
    crps_samples_by_group: dict | None = None,
) -> pd.DataFrame:
    """Create a row-aligned prediction dataframe for one validation or test split."""
    pred = predictions.reorder_to_original() if isinstance(predictions, GroupedModelPrediction) else predictions
    n_rows = len(target_data.response)

    def optional_column(values):
        if values is None:
            return np.full(n_rows, np.nan)
        return np.asarray(values).reshape(-1)

    frame = pd.DataFrame({
        "split": split_name,
        "row_index": np.arange(n_rows),
        "group_id": np.asarray(pred.group_ids).reshape(-1),
        "y_true": np.asarray(target_data.response).reshape(-1),
        "mean": np.asarray(pred.mean).reshape(-1),
        "std": np.asarray(pred.std).reshape(-1),
        "fixed_effect": optional_column(pred.fixed_effect),
        "random_effect": optional_column(pred.random_effect),
    })
    interval_columns, interval_method = _prediction_interval_quantiles(
        predictions=predictions,
        nominal_levels=interval_levels,
        n_decoder_samples=interval_samples,
        seed=interval_seed,
        crps_samples_by_group=crps_samples_by_group,
    )
    for column, values in interval_columns.items():
        frame[column] = values
    frame["interval_method"] = interval_method
    frame["interval_samples_per_latent"] = interval_samples if interval_method in SAMPLED_INTERVAL_METHODS else 0
    return frame


def _target_npbd_for_crps(model, target_data: BaseDataset):
    if model.name in ["npboost", "anpboost"]:
        return ModelDataAdapter.to_npboost_data(target_data)
    return ModelDataAdapter.to_np_data(target_data)


def _prediction_splits(model, data_bundle):
    """Return the target/context data pairs needed for predictions on each task split."""
    if model.task.name == IN_CONTEXT_TASK_NAME:
        context_data = None if model.is_gaussian_prediction else data_bundle.train
        return {
            VALIDATION_SPLIT_NAME: (data_bundle.validation, context_data),
            TEST_SPLIT_NAME: (data_bundle.test, context_data),
        }
    if model.task.name == FEW_SHOT_TASK_NAME:
        return {
            VALIDATION_SPLIT_NAME: (data_bundle.validation_query, data_bundle.validation_support),
            TEST_SPLIT_NAME: (data_bundle.test_query, data_bundle.test_support),
        }
    raise ValueError(f"Unknown task name: {model.task.name}")


def save_prediction_artifacts(
    model,
    data_bundle,
    results,
    model_folder,
    model_file_name,
    log_wandb: bool,
    metadata: dict,
    log: logging.Logger,
    interval_levels: tuple[float, ...] = DEFAULT_PREDICTION_INTERVAL_LEVELS,
    interval_samples: int | None = DEFAULT_PREDICTION_INTERVAL_SAMPLES,
):
    """Save row-aligned validation/test predictions and a metrics JSON file.

    Each parquet file contains one row per target observation with ``y_true``,
    ``mean``, ``std``, optional fixed/random-effect components, and interval
    columns such as ``lower_0.95``/``upper_0.95``. For grouped NP predictions,
    intervals reuse the same samples as CRPS so the stored intervals and logged
    probabilistic score are based on the same random draw.
    """
    output_dir = Path(model_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = _prediction_splits(model, data_bundle)
    saved_files = []
    requested_interval_samples = int(interval_samples if interval_samples is not None else model.crps_samples)
    grouped_interval_uses_crps_samples = False
    replayed_crps_by_split = {}

    for split_name, (target_data, context_data) in splits.items():
        model._set_prediction_seed(split_name)
        predictions = model.predict(target_data=target_data, context_data=context_data)
        crps_samples_by_group = None
        split_interval_samples = requested_interval_samples
        if _has_group_distributions(predictions):
            split_interval_samples = model.crps_samples
            grouped_interval_uses_crps_samples = True
            target_npbd = _target_npbd_for_crps(model, target_data)
            model._set_evaluation_seed(split_name)
            replayed_crps, crps_samples_by_group = crps_eval(
                predictions,
                target_npbd,
                model.crps_samples,
                model.device,
                return_samples=True,
            )
            replayed_crps_by_split[split_name] = replayed_crps
            reported_crps = results.get(split_name, {}).get("crps") if isinstance(results.get(split_name), dict) else None
            if reported_crps is None or pd.isna(reported_crps):
                log.info(
                    f"Replayed {split_name} CRPS from saved-interval samples: {replayed_crps:.12g}"
                )
            else:
                log.info(
                    f"Replayed {split_name} CRPS from saved-interval samples: {replayed_crps:.12g} "
                    f"(experiment runner reported {reported_crps:.12g}; abs diff {abs(replayed_crps - reported_crps):.3g})"
                )
            if requested_interval_samples != model.crps_samples:
                log.warning(
                    f"Ignoring prediction_interval_samples={requested_interval_samples} for {split_name} grouped intervals; "
                    f"using model.crps_samples={model.crps_samples} so intervals reuse the CRPS samples."
                )
        pred_df = _prediction_frame(
            predictions=predictions,
            target_data=target_data,
            split_name=split_name,
            interval_levels=interval_levels,
            interval_samples=split_interval_samples,
            interval_seed=model._prediction_seed(split_name),
            crps_samples_by_group=crps_samples_by_group,
        )
        pred_path = output_dir / f"{model_file_name}-{split_name.lower()}-predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
        saved_files.append(pred_path)

    metrics_path = output_dir / f"{model_file_name}-metrics.json"
    results = dict(results)
    if metadata:
        results["source_run"] = metadata.get("source_run")
        results["original_performance"] = metadata.get("original_performance")
    results["prediction_interval_artifacts"] = {
        "nominal_levels": list(interval_levels),
        "sample_quantile_samples_per_latent": model.crps_samples if grouped_interval_uses_crps_samples else requested_interval_samples,
        "requested_sample_quantile_samples_per_latent": requested_interval_samples,
        "grouped_intervals_reuse_crps_samples": grouped_interval_uses_crps_samples,
        "replayed_crps_from_interval_samples": replayed_crps_by_split,
        "sample_quantile_note": (
            "Grouped predictive distributions store latent components. Their saved "
            "intervals reuse the same response samples as CRPS; Gaussian models use "
            "mean/std Normal quantiles."
        ),
    }
    with metrics_path.open("w") as f:
        json.dump(_json_ready(results), f, indent=2)
    saved_files.append(metrics_path)
    log.info(f"Saved prediction artifacts to {output_dir}")

    if log_wandb:
        artifact = wandb.Artifact(
            name=f"{model_file_name}-predictions",
            type="predictions",
            metadata=_json_ready(metadata),
        )
        for path in saved_files:
            artifact.add_file(str(path), name=path.name)
        wandb.log_artifact(artifact)


def _is_missing_csv_value(value) -> bool:
    missing = pd.isna(value)
    # pd.isna returns an array for array-likes; CSV cells only need scalar missing checks.
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _hydra_override_literal(value) -> str:
    literal_value = value.item() if isinstance(value, np.generic) else value
    if isinstance(literal_value, float) and literal_value.is_integer():
        return str(int(literal_value))
    return str(literal_value)


def _csv_hydra_override(run_to_fit: pd.Series, key: str) -> str | None:
    if key not in run_to_fit or _is_missing_csv_value(run_to_fit[key]):
        return None
    return f"{key}={_hydra_override_literal(run_to_fit[key])}"


def _data_config_override(local_project_name):
    """Return the Hydra data override for a local results project name."""
    config_dir = PROJECT_ROOT / "configs" / "data"
    for data_type in ("real", "synthetic"):
        if (config_dir / data_type / f"{local_project_name}.yaml").is_file():
            return f"{data_type}/{local_project_name}"
    raise ValueError(f"No data config for local project '{local_project_name}'.")


def _current_repo_cfg(
    local_project_name,
    model_name,
    task_name,
    data_split_seed,
    fixed_effect_name,
    run_to_fit,
    validation_metric,
    log,
):
    """Build a config from local files and apply CSV hyperparameters."""
    data_override = _data_config_override(local_project_name)
    overrides = [
        f"data={data_override}",
        f"model={model_name}",
        f"task={task_name}",
        f"data.experiment.split_data_seed={data_split_seed}",
        f"model.validation_metric={validation_metric}",
    ]
    if data_override.startswith("synthetic/"):
        if not fixed_effect_name:
            raise ValueError("For synthetic data, fixed_effect_name must be provided.")
        overrides.append(f"data/synthetic/fixed_effect={fixed_effect_name}")

    applied = ["model.validation_metric"]
    for key in TUNED_HYPERPARAMS_PER_MODEL.get(model_name, []):
        if key in CSV_RUN_METADATA_COLUMNS:
            continue
        override = _csv_hydra_override(run_to_fit, key)
        if override is not None:
            overrides.append(override)
            applied.append(key)

    config_dir = PROJECT_ROOT / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)

    cfg.wandb.enabled = False
    log.info(f"Using current repo config with CSV hyperparameters: {applied}")
    return cfg


def _model_results_csv_path(local_project_name: str, model_name: str) -> Path:
    """Find the parameter CSV for a model across known result folders."""
    candidates = [
        RESULT_FOLDER / local_project_name,
        ADDITIONAL_RESULT_FOLDER / local_project_name,
    ]
    for result_dir in candidates:
        csv_path = result_dir / f"{model_name}_params.csv"
        if csv_path.exists():
            return csv_path
    searched = ", ".join(str(path / f"{model_name}_params.csv") for path in candidates)
    raise FileNotFoundError(f"Could not find parameter CSV for {model_name}. Searched: {searched}")


def _load_run_row(
    local_project_name: str,
    model_name: str,
    task_name: str,
    data_split_seed: int,
    fixed_effect_name: str | None,
    log: logging.Logger,
) -> pd.Series:
    model_results_csv_file = _model_results_csv_path(local_project_name, model_name)
    log.info(f"Getting the run id from {model_results_csv_file}")
    all_runs = pd.read_csv(model_results_csv_file)

    candidate_runs = all_runs[
        (all_runs["task_name"] == task_name) &
        (all_runs["data_split_seed"] == data_split_seed)
    ]

    has_synthetic_columns = {"fixed_effect_type", "feature_dimension"}.issubset(all_runs.columns)
    if has_synthetic_columns:
        if not fixed_effect_name:
            raise ValueError("For synthetic data, fixed_effect_name must be provided.")
        fixed_effect_type, feature_dimension_label = fixed_effect_name.split("_")
        feature_dimension = int(feature_dimension_label[:-1])
        candidate_runs = candidate_runs[
            (candidate_runs["fixed_effect_type"] == fixed_effect_type) &
            (candidate_runs["feature_dimension"] == feature_dimension)
        ]

    if len(candidate_runs) == 0:
        raise ValueError(
            "No runs found for the specified model/task/split/fixed-effect criteria."
        )
    if len(candidate_runs) > 1:
        raise ValueError(
            "Multiple runs found for the specified model/task/split/fixed-effect criteria. "
            "Please check the CSV file."
        )

    return candidate_runs.iloc[0]


def _resolve_config(
    wandb_project_name: str,
    local_project_name: str,
    model_name: str,
    task_name: str,
    data_split_seed: int,
    fixed_effect_name: str | None,
    run_to_fit: pd.Series,
    validation_metric: str,
    force_wandb_run: bool,
    log: logging.Logger,
) -> tuple[DictConfig, dict, dict, str]:
    wandb_run_id = run_to_fit["run_id"]
    source_run = {
        "project": wandb_project_name,
        "run_id": wandb_run_id,
        "path": f"{wandb_project_name}/{wandb_run_id}",
        "url": None,
        "name": None,
    }
    original_performance = {
        "csv": _extract_metric_fields(run_to_fit),
        "wandb_summary": {},
    }

    if not force_wandb_run:
        cfg = _current_repo_cfg(
            local_project_name=local_project_name,
            model_name=model_name,
            task_name=task_name,
            data_split_seed=data_split_seed,
            fixed_effect_name=fixed_effect_name,
            run_to_fit=run_to_fit,
            validation_metric=validation_metric,
            log=log,
        )
        return cfg, source_run, original_performance, "current_repo_with_csv_hyperparams"

    api = wandb.Api()
    wandb_run = api.run(f"{wandb_project_name}/{wandb_run_id}")
    cfg = OmegaConf.create(wandb_run.config)
    source_run = _wandb_run_reference(wandb_run, wandb_project_name, wandb_run_id)
    try:
        original_performance["wandb_summary"] = _extract_metric_fields(dict(wandb_run.summary))
    except Exception as summary_exc:
        log.warning(f"Could not read summary metrics for wandb run '{source_run['path']}': {summary_exc}")
    source_run_url_suffix = f" ({source_run['url']})" if source_run.get("url") else ""
    log.info(
        f"Loaded config from wandb run: {source_run['path']}{source_run_url_suffix}"
    )
    return cfg, source_run, original_performance, "wandb"


def _build_model_path(
    local_project_name: str,
    model_name: str,
    fixed_effect_name: str | None,
    task,
    data_split_seed: int,
    is_synthetic: bool,
):
    if is_synthetic:
        return SyntheticDataLayout.get_model_path(
            experiment_name=local_project_name,
            model_name=model_name,
            fixed_effect_name=fixed_effect_name,
            task=task,
            split_data_seed=data_split_seed,
        )
    return RealDataLayout.get_model_path(
        experiment_name=local_project_name,
        model_name=model_name,
        task=task,
        split_data_seed=data_split_seed,
    )


def fit_and_save_model(
    wandb_project_name,
    local_project_name,
    model_name,
    task_name,
    data_split_seed,
    fixed_effect_name=None,
    save_model=True,
    save_predictions=None,
    prediction_interval_samples=DEFAULT_PREDICTION_INTERVAL_SAMPLES,
    validation_metric="rmse",
    force_wandb_run=False,
):
    """Refit one CSV-selected run and optionally save the fitted model.

    ``local_project_name`` points to the local result folder containing
    ``{model_name}_params.csv``. The selected row provides the run id and tuned
    hyperparameters. By default prediction artifacts are saved only when the
    model is saved for split seed 0.
    """

    log = logging.getLogger(local_project_name)

    run_to_fit = _load_run_row(
        local_project_name=local_project_name,
        model_name=model_name,
        task_name=task_name,
        data_split_seed=data_split_seed,
        fixed_effect_name=fixed_effect_name,
        log=log,
    )
    wandb_run_id = run_to_fit["run_id"]
    original_validation_score = run_to_fit["best_validation_score"]
    cfg, source_run, original_performance, config_source = _resolve_config(
        wandb_project_name=wandb_project_name,
        local_project_name=local_project_name,
        model_name=model_name,
        task_name=task_name,
        data_split_seed=data_split_seed,
        fixed_effect_name=fixed_effect_name,
        run_to_fit=run_to_fit,
        validation_metric=validation_metric,
        force_wandb_run=force_wandb_run,
        log=log,
    )

    log.info(f"Source run reference: {_json_ready(source_run)}")
    log.info(f"Original performance reference: {_json_ready(original_performance)}")

    is_synthetic = (cfg.data.type == "synthetic")

    if not torch.cuda.is_available() and cfg.device == "cuda":
        cfg.device = "cpu"


    log_wandb = cfg.wandb.enabled
    should_save_predictions = (save_model and data_split_seed == 0) if save_predictions is None else save_predictions

    cfg.wandb.enabled = log_wandb

    if log_wandb:
        if is_synthetic:
            wandb_run_name = f"{model_name}-{local_project_name}-{fixed_effect_name}-{task_name}-seed{data_split_seed}"
        else:
            wandb_run_name = f"{model_name}-{local_project_name}-{task_name}-seed{data_split_seed}"
        
        wandb.init(
            project="refit-models",
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=wandb_run_name,
            config=OmegaConf.to_container(cfg),
        )

    experiment_seed = cfg.data.experiment.seed


    log.info("Starting experiment.")

    set_seed(experiment_seed)

    data_provider = instantiate(cfg.data.provider)
    task = instantiate(cfg.task)

    if is_synthetic:
        split_df = data_provider.fetch_split_df(
            task=task,
            seed=data_split_seed)
    else:
        split_df = data_provider.fetch_split_df(
            task=task,
            response_column=cfg.data.experiment.response_name,
            seed=data_split_seed)

    model_folder, model_file_name = _build_model_path(
        local_project_name=local_project_name,
        model_name=model_name,
        fixed_effect_name=fixed_effect_name,
        task=task,
        data_split_seed=data_split_seed,
        is_synthetic=is_synthetic,
    )

    data_bundle = BundleFactory.create_bundle(split_df, task)

    # For split seed 0, we plot intermediate results for debugging and visualization (same as in run_experiment script).
    plot_intermediate = (data_split_seed == 0)

    model = instantiate(cfg.model, experiment_seed=experiment_seed, task=task, plot_intermediate=plot_intermediate)
    

    results = model.run(data_bundle=data_bundle)
    if save_model:
        model.save_model(model_folder, model_file_name)
        log.info(f"Model {model_file_name} saved to {model_folder}")
    else:
        log.info("save_model=false; fitted model was not saved.")

    if should_save_predictions:
        save_prediction_artifacts(
            model=model,
            data_bundle=data_bundle,
            results=results,
            model_folder=model_folder,
            model_file_name=model_file_name,
            log_wandb=log_wandb,
            metadata={
                "model_name": model_name,
                "task_name": task_name,
                "data_split_seed": data_split_seed,
                "fixed_effect_name": fixed_effect_name,
                "local_project_name": local_project_name,
                "original_wandb_project": wandb_project_name,
                "original_run_id": wandb_run_id,
                "source_run": source_run,
                "original_performance": original_performance,
                "config_source": config_source,
                "validation_metric": validation_metric,
            },
            log=log,
            interval_samples=prediction_interval_samples,
        )
    
    if log_wandb:
        wandb.log({"validation_results": results[VALIDATION_SPLIT_NAME],
                   "test_results": results[TEST_SPLIT_NAME],
                   "validation_metric": validation_metric,
                   "original_wandb_project": wandb_project_name,
                   "original_run_id": wandb_run_id,
                   "original_validation_score": original_validation_score,
                   "source_run": _json_ready(source_run),
                   "original_performance": _json_ready(original_performance)})
        wandb.finish()
    log.info(f"Final results: {_json_ready(results)}")
    log.info(f"Initial run info: {_initial_run_info_from_params(run_to_fit)}")
    log.info("Experiment completed.")

@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    fit_and_save_model(
        wandb_project_name=cfg.wandb_project_name,
        local_project_name=cfg.local_project_name,
        model_name=cfg.model_name,
        task_name=cfg.task_name,
        data_split_seed=int(cfg.data_split_seed),
        fixed_effect_name=cfg.get("fixed_effect_name", None),
        save_model=cfg.get("save_model", True),
        save_predictions=cfg.get("save_predictions", None),
        prediction_interval_samples=cfg.get("prediction_interval_samples", DEFAULT_PREDICTION_INTERVAL_SAMPLES),
        validation_metric=cfg.get("validation_metric", "rmse"),
        force_wandb_run=cfg.get("force_wandb_run", False),
    )


if __name__ == "__main__":
    main()
    
