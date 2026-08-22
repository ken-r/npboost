"""Dense TabICL prediction generation shared by scripts and notebooks.

These helpers create and store prediction frames for publication plots.
They can then be used to generate plots in notebooks without needing to refit the models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from src.constants import (
    FEW_SHOT_TASK_NAME,
    GROUPING_COLUMN_NAME,
    IN_CONTEXT_TASK_NAME,
    SPLIT_COLUMN_NAME,
    SYNTHETIC_FEATURE_COLUMN_PREFIX,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
)
from src.data.data_layout import PROJECT_ROOT, MODEL_FOLDER
from src.data.datasets import BundleFactory
from src.data.synthetic_dense_prediction_data import (
    FEW_SHOT_IRRELEVANT_SEED,
    FEW_SHOT_NOISE_SEED,
    FEW_SHOT_RANDOM_EFFECT_SEED,
    FEW_SHOT_SUPPORT_SEED,
    GROUP_ID_OFFSET,
    IN_CONTEXT_DENSE_NOISE_SEED,
    PLOT_GENERATION_SEED,
    dense_few_shot_plot_split_df,
    dense_in_context_plot_split_df,
)


LOGGER = logging.getLogger(__name__)

TABICL_MODEL_LABELS = {
    "tabicl_taskwise": "TabICL_T",
    "tabicl_pooled": "TabICL_P",
}
TABICL_MODEL_NAMES = ["tabicl_taskwise", "tabicl_pooled"]


def _tabicl_model_classes():
    from src.training.experiment_runner import (
        TabICLPooledExperiment,
        TabICLTaskwiseExperiment,
    )

    return {
        "tabicl_taskwise": TabICLTaskwiseExperiment,
        "tabicl_pooled": TabICLPooledExperiment,
    }


def compose_task_config(data_name: str, fixed_effect_name: str, task_name: str, split_seed: int):
    """Compose CPU-only Hydra config used for dense TabICL plot predictions."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                f"data=synthetic/{data_name}",
                f"data/synthetic/fixed_effect={fixed_effect_name}",
                f"task={task_name}",
                f"data.experiment.split_data_seed={split_seed}",
                "wandb.enabled=false",
                "device=cpu",
            ],
        )


def make_tabicl_runner(
    model_name: str,
    *,
    experiment_seed: int,
    task,
    n_estimators: int,
    n_jobs: int,
    crps_alphas=None,
):
    """Create a TabICL baseline for local prediction."""
    runner_cls = _tabicl_model_classes()[model_name]
    return runner_cls(
        name=model_name,
        experiment_seed=experiment_seed,
        task=task,
        wandb_enabled=False,
        n_estimators=n_estimators,
        device="cpu",
        crps_alphas=crps_alphas,
        model_params={"n_jobs": n_jobs},
    )


def _tabicl_interpolate_quantile(
    quantiles: np.ndarray,
    quantile_alphas: np.ndarray,
    alpha: float,
) -> np.ndarray:
    order = np.argsort(quantile_alphas)
    sorted_alphas = np.asarray(quantile_alphas, dtype=float)[order]
    sorted_quantiles = np.asarray(quantiles, dtype=float)[:, order]
    return np.asarray([np.interp(alpha, sorted_alphas, row) for row in sorted_quantiles])


def prediction_frame_from_tabicl_runner(
    *,
    model_name: str,
    task,
    train_data,
    context_data,
    target_data,
    feature: str,
    interval_level: float,
    experiment_seed: int,
    n_estimators: int,
    n_jobs: int,
    split_name: str = TEST_SPLIT_NAME,
) -> pd.DataFrame:
    """Return row-aligned TabICL predictions for one target split.

    The returned frame contains the plotting feature, ``y_true``, ``mean``,
    an approximate ``std`` derived from the requested interval, true synthetic
    components, group-level RMSE/CRPS, and interval columns such as
    ``lower_0.95``/``upper_0.95``.
    """
    runner = make_tabicl_runner(
        model_name,
        experiment_seed=experiment_seed,
        task=task,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    runner.fit(train_data)
    runner._set_prediction_seed(split_name)
    predictions = runner.predict(
        target_data=target_data,
        context_data=context_data,
        random_state=runner._prediction_seed(split_name),
    )

    quantiles = np.asarray(predictions.quantiles, dtype=float)
    quantile_alphas = np.asarray(predictions.quantile_alphas, dtype=float)
    lower_col = f"lower_{interval_level:.2f}"
    upper_col = f"upper_{interval_level:.2f}"
    alpha = 1.0 - interval_level
    lower = _tabicl_interpolate_quantile(quantiles, quantile_alphas, alpha / 2.0)
    upper = _tabicl_interpolate_quantile(quantiles, quantile_alphas, 1.0 - alpha / 2.0)
    z_score = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
    std = (upper - lower) / (2.0 * z_score)
    std[~np.isfinite(std)] = np.nan

    frame = pd.DataFrame(
        {
            "row_index": np.arange(len(target_data.response)),
            "group_id": np.asarray(predictions.group_ids).reshape(-1),
            "model_name": model_name,
            "model_label": TABICL_MODEL_LABELS.get(model_name, model_name),
            feature: target_data.features[feature].to_numpy(),
            "y_true": target_data.response.to_numpy(),
            "mean": np.asarray(predictions.mean).reshape(-1),
            "std": std,
            "fixed_effect": np.nan,
            "random_effect": np.nan,
            "true_fixed_effect": target_data.fixed_effect_part.to_numpy(),
            "true_random_effect": target_data.random_effect_part.to_numpy(),
            "true_noise": target_data.noise_part.to_numpy(),
            lower_col: lower,
            upper_col: upper,
            "interval_method": "tabicl_quantile",
        }
    )
    frame["true_signal"] = frame["true_fixed_effect"] + frame["true_random_effect"]

    group_metrics = runner.evaluate_by_group(predictions, target_data).set_index(GROUPING_COLUMN_NAME)
    frame["group_rmse"] = frame["group_id"].map(group_metrics["rmse"])
    frame["group_crps"] = frame["group_id"].map(group_metrics["crps"])
    return frame


def prediction_frames_from_tabicl_runners(
    *,
    model_names,
    task,
    train_data,
    context_data,
    target_data,
    feature: str,
    interval_level: float,
    experiment_seed: int,
    n_estimators: int,
    n_jobs: int,
    split_name: str = TEST_SPLIT_NAME,
) -> pd.DataFrame:
    """Stack dense prediction frames for several TabICL variants."""
    return pd.concat(
        [
            prediction_frame_from_tabicl_runner(
                model_name=model_name,
                task=task,
                train_data=train_data,
                context_data=context_data,
                target_data=target_data,
                feature=feature,
                interval_level=interval_level,
                experiment_seed=experiment_seed,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                split_name=split_name,
            )
            for model_name in model_names
        ],
        ignore_index=True,
    )


def dense_prediction_output_path(experiment_name: str, fixed_effect_name: str, task_name: str, split_seed: int) -> Path:
    """Return the standard parquet path for dense TabICL prediction artifacts."""
    return (
        MODEL_FOLDER
        / experiment_name
        / "models"
        / fixed_effect_name
        / f"dense_model_predictions-{task_name}-seed{split_seed}.parquet"
    )


def prepare_in_context_predictions(args) -> tuple[pd.DataFrame, Path]:
    """Build dense predictions for stored in-context groups.

    The helper recovers each selected group's saved random-effect realization
    and evaluates TabICL on a dense test grid for those same groups.
    """
    cfg = compose_task_config(args.data_name, args.fixed_effect_name, IN_CONTEXT_TASK_NAME, args.split_seed)
    provider = instantiate(cfg.data.provider)
    task = instantiate(cfg.task)
    feature = f"{SYNTHETIC_FEATURE_COLUMN_PREFIX}0"

    if provider.fixed_effect.feature_dimension != 1:
        raise ValueError("Dense line predictions are only supported for 1D fixed effects.")

    base_split_df = provider.fetch_split_df(task=task, seed=args.split_seed)
    raw_df = provider.fetch_raw_df(seed=args.split_seed)
    groups = sorted(
        base_split_df.loc[
            base_split_df[SPLIT_COLUMN_NAME] == TRAIN_SPLIT_NAME,
            GROUPING_COLUMN_NAME,
        ].unique()
    )[: args.n_tasks]

    dense_split_df, recovery_diagnostics = dense_in_context_plot_split_df(
        provider,
        base_split_df,
        raw_df,
        groups=groups,
        seed=PLOT_GENERATION_SEED,
        noise_seed=IN_CONTEXT_DENSE_NOISE_SEED,
        n_query=args.n_query,
        feature=feature,
    )
    LOGGER.info("In-context GP recovery diagnostics:\n%s", recovery_diagnostics.to_string(index=False))

    model_split_df = dense_split_df.drop(
        columns=["true_signal", "plot_generation_seed", "noise_generation_seed", "plot_only"],
        errors="ignore",
    )
    tabicl_model_split_df = pd.concat(
        [
            model_split_df[model_split_df[SPLIT_COLUMN_NAME] == TRAIN_SPLIT_NAME],
            model_split_df[model_split_df[SPLIT_COLUMN_NAME] == TEST_SPLIT_NAME],
        ],
        ignore_index=True,
        sort=False,
    )
    tabicl_bundle = BundleFactory.create_bundle(tabicl_model_split_df, task)
    pred_df = prediction_frames_from_tabicl_runners(
        model_names=args.model_names,
        task=task,
        train_data=tabicl_bundle.train,
        context_data=tabicl_bundle.train,
        target_data=tabicl_bundle.test,
        feature=feature,
        interval_level=args.interval_level,
        experiment_seed=cfg.data.experiment.seed,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
    pred_df.insert(0, "task_name", IN_CONTEXT_TASK_NAME)
    return pred_df, dense_prediction_output_path(args.data_name, args.fixed_effect_name, IN_CONTEXT_TASK_NAME, args.split_seed)


def prepare_few_shot_predictions(args) -> tuple[pd.DataFrame, Path]:
    """Build dense predictions for fresh few-shot plot groups.

    The sampled plot groups are checked not to overlap with training groups, so
    the frame visualizes held-out-task behavior and does not reuse a training task.
    """
    cfg = compose_task_config(args.data_name, args.fixed_effect_name, FEW_SHOT_TASK_NAME, args.split_seed)
    provider = instantiate(cfg.data.provider)
    task = instantiate(cfg.task)
    feature = f"{SYNTHETIC_FEATURE_COLUMN_PREFIX}0"

    if provider.fixed_effect.feature_dimension != 1:
        raise ValueError("Dense line predictions are only supported for 1D fixed effects.")

    dense_split_df = dense_few_shot_plot_split_df(
        provider,
        random_effect_seed=FEW_SHOT_RANDOM_EFFECT_SEED,
        noise_seed=FEW_SHOT_NOISE_SEED,
        support_seed=FEW_SHOT_SUPPORT_SEED,
        irrelevant_seed=FEW_SHOT_IRRELEVANT_SEED,
        n_tasks=args.n_tasks,
        n_support=args.n_support,
        n_query=args.n_query,
        group_id_offset=args.group_id_offset,
    )

    training_split_df = provider.fetch_split_df(task=task, seed=args.split_seed)
    training_groups = set(
        training_split_df.loc[
            training_split_df[SPLIT_COLUMN_NAME] == TRAIN_SPLIT_NAME,
            GROUPING_COLUMN_NAME,
        ].unique()
    )
    plot_groups = set(dense_split_df[GROUPING_COLUMN_NAME].unique())
    if not plot_groups.isdisjoint(training_groups):
        raise ValueError("Dense few-shot plot groups overlap with training groups.")

    model_split_df = dense_split_df.drop(
        columns=[
            "true_signal",
            "plot_generation_seed",
            "random_effect_generation_seed",
            "noise_generation_seed",
            "support_generation_seed",
            "plot_only",
        ],
        errors="ignore",
    )
    dense_bundle = BundleFactory.create_bundle(model_split_df, task)
    training_bundle = BundleFactory.create_bundle(training_split_df, task)

    pred_df = prediction_frames_from_tabicl_runners(
        model_names=args.model_names,
        task=task,
        train_data=training_bundle.train,
        context_data=dense_bundle.test_support,
        target_data=dense_bundle.test_query,
        feature=feature,
        interval_level=args.interval_level,
        experiment_seed=cfg.data.experiment.seed,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
    pred_df.insert(0, "task_name", FEW_SHOT_TASK_NAME)
    return pred_df, dense_prediction_output_path(args.data_name, args.fixed_effect_name, FEW_SHOT_TASK_NAME, args.split_seed)


def save_prediction_frame(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    """Write one dense prediction frame to parquet."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    LOGGER.info("Saved %s rows to %s", len(frame), path)


def generate_dense_tabicl_prediction_frames(
    *,
    data_name: str = "gp_gaussian",
    fixed_effect_name: str = "steps_1D",
    split_seed: int = 0,
    tasks: tuple[str, ...] | list[str] = ("all",),
    model_names: tuple[str, ...] | list[str] = tuple(TABICL_MODEL_NAMES),
    n_tasks: int = 3,
    n_support: int = 20,
    n_query: int = 400,
    group_id_offset: int = GROUP_ID_OFFSET,
    interval_level: float = 0.95,
    n_estimators: int = 8,
    n_jobs: int = 1,
    save: bool = False,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Generate dense TabICL prediction frames, optionally saving them.

    Returns a mapping from task name to the frame that was generated.   
    When ``save=True``, each frame is also written to the standard path under
    ``results/{data_name}/models/{fixed_effect_name}``.
    """
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1.")

    args = SimpleNamespace(
        data_name=data_name,
        fixed_effect_name=fixed_effect_name,
        split_seed=split_seed,
        tasks=list(tasks),
        model_names=list(model_names),
        n_tasks=n_tasks,
        n_support=n_support,
        n_query=n_query,
        group_id_offset=group_id_offset,
        interval_level=interval_level,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        overwrite=overwrite,
    )

    requested_tasks = {IN_CONTEXT_TASK_NAME, FEW_SHOT_TASK_NAME} if "all" in args.tasks else set(args.tasks)
    frames = {}
    if IN_CONTEXT_TASK_NAME in requested_tasks:
        frame, path = prepare_in_context_predictions(args)
        if save:
            save_prediction_frame(frame, path, overwrite=overwrite)
        frames[IN_CONTEXT_TASK_NAME] = frame
    if FEW_SHOT_TASK_NAME in requested_tasks:
        frame, path = prepare_few_shot_predictions(args)
        if save:
            save_prediction_frame(frame, path, overwrite=overwrite)
        frames[FEW_SHOT_TASK_NAME] = frame
    return frames
