"""Diagnostic plots for fitted models and prediction objects.

These helpers are used by experiment runners for quick validation plots. Most
functions accept either flat or grouped predictions and use these predictions
to generate plots.

These plotting functions were generated for quick model diagnostics, 
and their style and layout may not be suitable for publication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import plotnine as pn
from src.evaluation.crps import crps_score, gaussian_crps
from src.evaluation.metrics import groupwise_metrics
import torch
from matplotlib.figure import Figure
from statsmodels.nonparametric.smoothers_lowess import lowess

from plotnine import (
    aes,
    coord_equal,
    facet_grid,
    facet_wrap,
    geom_abline,
    geom_boxplot,
    geom_density,
    geom_hline,
    geom_line,
    geom_text,
    geom_point,
    geom_ribbon,
    geom_smooth,
    geom_segment,
    ggplot,
    guide_legend,
    guides,
    coord_cartesian,
    coord_fixed,
    scale_shape_manual,
    stat_smooth,
    geom_col,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
    element_blank,
    element_line,
    element_rect,
    element_text,
    scale_fill_manual,
    scale_alpha_manual,
    scale_size_manual
)

from src.data.datasets import BaseDataset, DataBundle, ModelDataAdapter, NPBDataset
from src.evaluation.model_predictions import GroupedModelPrediction, ModelPrediction
from src.constants import FEW_SHOT_TASK_NAME, GROUPING_COLUMN_NAME, IN_CONTEXT_TASK_NAME
from src.task.base_task import BaseTask
from src.utils.helpers import isolated_torch_rng

from src.plot.theme import (
    NPBOOST_NAME,
    ANPBOOST_NAME,
    NP_NAME,
    ANP_NAME,
    GPLINEAR_NAME,
    LME_NAME,
    PLOT_MODEL_NAMES,
    CONTEXT_NAME,
    TARGET_NAME,
    PAPER_PALETTE,
    PAPER_THEME,
    apply_academic_style as _apply_style
)


def _display_model_name(model_name: str) -> str:
    return PLOT_MODEL_NAMES.get(str(model_name).lower(), str(model_name))


def _as_1d(values: Any) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def _distribution_device(distribution: torch.distributions.Distribution) -> torch.device | None:
    try:
        mean = distribution.mean
    except NotImplementedError:
        return None

    if isinstance(mean, torch.Tensor):
        return mean.device
    return None


def _sample_distribution(
    distribution: torch.distributions.Distribution,
    sample_shape: torch.Size,
    seed: int,
) -> torch.Tensor:
    """Sample a predictive distribution without changing the global Torch RNG."""
    with isolated_torch_rng(seed, device=_distribution_device(distribution)):
        return distribution.sample(sample_shape)


def _resolve_save_path(save_path: str | Path | None) -> Optional[Path]:
    """Create the parent folder for an optional plot path."""
    if not save_path:
        return None
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _group_points(data: Any, group_id: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts group-specific points.
    
    Works on BaseDataset and NPBDataset.
    """
    mask = (data.grouping_var == group_id).to_numpy()
    
    # Extract, squeeze to 1D, and ensure it is at least a 1D array
    x_values = np.atleast_1d(data.features[mask].to_numpy().squeeze())
    y_values = np.atleast_1d(data.response[mask].to_numpy().squeeze())
    original_indices = np.where(mask)[0]
    
    return x_values, y_values, original_indices


def _single_group_point_frame(data: Any, group_id: Any, label: str) -> pd.DataFrame:
    """Return plotting rows for one group's observed points."""
    x_values, y_values, _ = _group_points(data, group_id)
    return pd.DataFrame({"x": x_values, "y": y_values, "PointType": label})


# Used for direct model exploration
def plot_fixed_effect(
    features: pd.DataFrame,
    fixed_effect_pred: np.ndarray,
    true_fixed_effect: Optional[np.ndarray] = None,
    model_name: str = "npboost",
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> pn.ggplot:
    """
    Plot predicted fixed effects against the input features.

    For 1D features, the x-axis is the feature value and the y-axis is
    the predicted fixed effect, with an optional dashed true fixed-effect line. 
    For 2D features, points are placed at the two feature coordinates and colored by
    the predicted fixed effect. 
    
    Use this to see whether the fixed-effect component has learned some global structure.
    """
    model_display_name = _display_model_name(model_name)
    model_color = PAPER_PALETTE.get(model_display_name, PAPER_PALETTE["Validation Prediction"])
    y_pred = _as_1d(fixed_effect_pred)

    if len(features) != len(y_pred):
        raise ValueError("features and fixed_effect_pred must have the same length")

    n_features = features.shape[1]

    if n_features == 1:
        x_name = features.columns[0]
        x_values = features.iloc[:, 0].to_numpy()
        order = np.argsort(x_values)
        x_sorted = x_values[order]
        pred_sorted = y_pred[order]

        # Prepare a long-format DataFrame for geom_line to have automatic legend matching paper colors
        lines_df = pd.DataFrame({
            "x": x_sorted,
            "Value": pred_sorted,
            "Type": model_display_name
        })

        if true_fixed_effect is not None:
            true_sorted = _as_1d(true_fixed_effect)[order]
            true_df = pd.DataFrame({
                "x": x_sorted,
                "Value": true_sorted,
                "Type": "Truth"
            })
            lines_df = pd.concat([lines_df, true_df], ignore_index=True)

        p = (
            ggplot(lines_df, aes(x="x", y="Value", color="Type", linetype="Type"))
            + geom_line(size=1.1)
            + scale_color_manual(values=PAPER_PALETTE)
            + scale_linetype_manual(values={model_display_name: "solid", "Truth": "dashed"})
        )

        pts_df = pd.DataFrame({"x": x_values, "y": y_pred})
        p = p + geom_point(
            data=pts_df,
            mapping=aes(x="x", y="y"),
            color=model_color,
            size=1.5,
            alpha=0.35,
            inherit_aes=False,
            show_legend=False,
        )

        p = p + labs(
            title="Fixed Effect Diagnostic",
            x=x_name,
            y="Fixed effect",
            color="",
            linetype="",
        )

        p = p + PAPER_THEME

    elif n_features == 2:
        x_name, y_name = features.columns[:2]
        x_values = features.iloc[:, 0].to_numpy()
        y_values = features.iloc[:, 1].to_numpy()

        plot_df = pd.DataFrame({
            "x": x_values,
            "y": y_values,
            "prediction": y_pred,
        })

        p = (
            ggplot(plot_df, aes(x="x", y="y", color="prediction"))
            + geom_point(size=2.0, alpha=0.95)
            + pn.scale_color_cmap(cmap_name="viridis", name="Predicted\nfixed effect")
            + coord_fixed(ratio=1)
            + labs(
                title="2D Fixed Effect Distribution",
                x=x_name,
                y=y_name,
            )
            + PAPER_THEME
            + theme(
                legend_position="right",
                figure_size=(7.0, 6.0),
            )
        )

    else:
        raise ValueError(f"Expected 1D or 2D features, got {n_features}D")

    # Save to file if path is provided
    resolved_path = _resolve_save_path(save_path)
    if resolved_path:
        p.save(str(resolved_path), dpi=dpi, verbose=False)

    if show:
        p.draw(show=True)

    return p

# Used for NPBoost model exploration notebook
def plot_fixed_effect_history(
    npboost_model: Any,
    target_data: BaseDataset | NPBDataset,
    true_fixed_effect: Optional[np.ndarray] = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> Figure:
    """
    Plot NPBoost fixed-effect predictions over boosting rounds for 1D features.

    The x-axis is the target feature and each line is one entry of
    the model's fixed effect history.  
    Color intensity increases with boosting round, so the latest round is the darkest line.
    
    Use this to diagnose whether the fixed effect stabilizes.
    """
    if not hasattr(npboost_model, "f_pred_history") or not npboost_model.f_pred_history:
        raise ValueError("Model does not have 'f_pred_history'. Make sure NPBoost is fitted.")

    x_values = target_data.features.to_numpy().reshape(-1)
    sort_indices = np.argsort(x_values)
    x_sorted = x_values[sort_indices]

    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)

    n_rounds = len(npboost_model.f_pred_history)
    alphas = np.linspace(0.15, 0.85, n_rounds)

    for round_idx in range(n_rounds):
        f_pred = npboost_model.f_pred_history[round_idx]
        if len(f_pred) == len(target_data.features):
            f_pred_sorted = f_pred[sort_indices]
        else:
            f_pred_sorted = npboost_model.lgbm_booster_.predict(target_data.features.iloc[sort_indices])

        ax.plot(
            x_sorted,
            f_pred_sorted,
            color=PAPER_PALETTE[NPBOOST_NAME],
            alpha=alphas[round_idx],
            linewidth=1.2,
            label="Boosting Round" if round_idx == n_rounds - 1 else None,
        )

    if true_fixed_effect is not None:
        true_sorted = _as_1d(true_fixed_effect)[sort_indices]
        ax.plot(
            x_sorted,
            true_sorted,
            color=PAPER_PALETTE["Truth"],
            linestyle="--",
            linewidth=2.2,
            label="True Fixed Effect",
        )

    _apply_style(
        ax,
        title="Fixed Effect History Over Boosting Rounds",
        xlabel="Feature (x)",
        ylabel="Fixed Effect",
        has_legend=True,
    )

    resolved_path = _resolve_save_path(save_path)
    if resolved_path:
        fig.savefig(resolved_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig

# Used for NPBoost model exploration notebook
def plot_pred_vs_true_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> Figure:
    """
    Plot predicted response values against true response values.

    Each point is one row, with true response on the x-axis and predicted response on the y-axis.
    The dashed identity line marks perfect predictions.
    The annotation reports RMSE and R2. 
    
    Good predictions cluster tightly along the diagonal. 
    Curvature, spread, or systematic offsets indicate bias or underfit.
    """
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    model_display_name = _display_model_name(model_name)

    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)

    ax.scatter(
        y_true,
        y_pred,
        color=PAPER_PALETTE.get(model_display_name, PAPER_PALETTE["Validation Prediction"]),
        alpha=0.5,
        s=20,
        linewidths=0,
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    lims = [min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val)]
    ax.plot(lims, lims, color="#D1495B", linestyle="--", linewidth=1.5, label="Identity (Perfect)")

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)

    stats_text = f"RMSE: {rmse:.4f}\n$R^2$: {r2:.4f}"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#C7C7C7", alpha=0.9),
    )

    _apply_style(
        ax,
        title=f"Predicted vs. True Response ({model_display_name})",
        xlabel="True Value",
        ylabel="Predicted Value",
        has_legend=True,
    )

    resolved_path = _resolve_save_path(save_path)
    if resolved_path:
        fig.savefig(resolved_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _coverage_quantile_arrays_from_predictions(
    predictions: ModelPrediction | GroupedModelPrediction,
    target_data: BaseDataset | NPBDataset,
    nominal_levels: list[float],
    eval_samples: int = 200,
    is_gaussian_prediction: bool = False,
    seed: int = 2,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], np.ndarray]:
    """
    Build lower/upper interval arrays for each nominal level, aligned to the full
    target dataset order.

    Returns
    -------
    lower_by_level, upper_by_level, true_y
        Dicts keyed by nominal coverage level, each value an array of shape (n_points,).
    """
    if isinstance(predictions, GroupedModelPrediction):
        preds = predictions.reorder_to_original()
    elif isinstance(predictions, ModelPrediction):
        preds = predictions
    else:
        raise TypeError("predictions must be ModelPrediction or GroupedModelPrediction")

    nominal_levels = [float(level) for level in nominal_levels]
    alphas = [1.0 - level for level in nominal_levels]

    true_y = _as_1d(target_data.response)
    n_points = len(true_y)

    lower_by_level = {level: np.empty(n_points, dtype=float) for level in nominal_levels}
    upper_by_level = {level: np.empty(n_points, dtype=float) for level in nominal_levels}

    group_ids = np.asarray(preds.group_ids)
    unique_groups = np.unique(group_ids)

    for group in unique_groups:
        group_mask = group_ids == group

        if not np.any(group_mask):
            continue

        if is_gaussian_prediction:
            mean_vals = _as_1d(preds.mean[group_mask])
            std_vals = _as_1d(preds.std[group_mask])

            for level, alpha in zip(nominal_levels, alphas):
                z_score = float(stats.norm.ppf(1 - alpha / 2))
                lower_by_level[level][group_mask] = mean_vals - z_score * std_vals
                upper_by_level[level][group_mask] = mean_vals + z_score * std_vals
        else:
            # Handle the the predictive distribution sampling for non-Gaussian predictions
            dist = preds.predictive_distributions[group]
            sample = _sample_distribution(dist, torch.Size([eval_samples]), seed=seed)
            sample = (
                torch.flatten(sample, start_dim=0, end_dim=1)
                .squeeze(-1)
                .squeeze(1)
                .to("cpu")
                .numpy()
            )

            for level, alpha in zip(nominal_levels, alphas):
                lower_by_level[level][group_mask] = np.quantile(
                    sample, q=alpha / 2, axis=0
                )
                upper_by_level[level][group_mask] = np.quantile(
                    sample, q=1 - alpha / 2, axis=0
                )

    return lower_by_level, upper_by_level, true_y

# Used for experiment runner plots
def plot_single_model_coverage(
    model_name: str,
    predictions: ModelPrediction | GroupedModelPrediction,
    prediction_metrics: dict[str, float] | None,
    target_data: BaseDataset | NPBDataset,
    nominal_levels: list[float] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
    eval_samples: int = 20,
    is_gaussian_prediction: bool = False,
    seed: int = 2,
) -> ggplot:
    """
    Plot empirical versus nominal interval coverage for one model.

    For each nominal level, the y-value is the fraction of target responses inside the corresponding confidence interval. 
    Intervals use Gaussian mean +/- z * std when `is_gaussian_prediction` is true
    Otherwise they use quantiles of predictive-distribution samples. 
    
    The diagonal is ideal calibration.
    If the resulting line lies below the diagonal, the model is overconfident.
    If the resulting line lies above the diagonal, the model is underconfident.
    """
    if nominal_levels is None:
        nominal_levels = np.round(np.linspace(0.05, 0.95, 19), 3).tolist()

    nominal_levels = [float(level) for level in nominal_levels]

    if any(level <= 0 or level >= 1 for level in nominal_levels):
        raise ValueError("All nominal_levels must lie strictly between 0 and 1.")

    lower_by_level, upper_by_level, true_y = _coverage_quantile_arrays_from_predictions(
        predictions=predictions,
        target_data=target_data,
        nominal_levels=nominal_levels,
        eval_samples=eval_samples,
        is_gaussian_prediction=is_gaussian_prediction,
        seed=seed,
    )

    rows = []
    for level in nominal_levels:
        lower = lower_by_level[level]
        upper = upper_by_level[level]
        empirical = ((true_y >= lower) & (true_y <= upper)).mean()

        rows.append(
            {
                "nominal": level,
                "empirical": float(empirical),
                "Model": model_name,
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("nominal").reset_index(drop=True)

    # Compute mean absolute calibration error (MACE)
    mean_abs_calibration_error = float(
        np.mean(np.abs(plot_df["empirical"] - plot_df["nominal"]))
    )

    if prediction_metrics is not None:
        subtitle = (
            f"CRPS: {prediction_metrics['crps']:.3f} | RMSE: {prediction_metrics['rmse']:.3f}\n"
            f"Mean absolute calibration error: {mean_abs_calibration_error:.3f}\n"
        )
    else:
        subtitle = f"Mean absolute calibration error: {mean_abs_calibration_error:.3f}"

    model_display_name = _display_model_name(model_name)
    model_color = PAPER_PALETTE["Validation Prediction"]

    fig = (
        ggplot(plot_df, aes(x="nominal", y="empirical"))
        + geom_abline(
            slope=1,
            intercept=0,
            linetype="dashed",
            color="#6B7280",
            size=0.8,
        )
        + geom_line(color=model_color, size=1.2)
        + geom_point(color=model_color, size=2.5)
        + scale_x_continuous(
            limits=(0, 1),
            breaks=np.round(np.linspace(0, 1, 6), 2).tolist(),
        )
        + scale_y_continuous(
            limits=(0, 1),
            breaks=np.round(np.linspace(0, 1, 6), 2).tolist(),
        )
        + coord_equal()
        + labs(
            title=f"{model_display_name} Coverage Calibration",
            subtitle=subtitle,
            x="Nominal coverage",
            y="Empirical coverage",
        )
        + PAPER_THEME
        + theme(figure_size=(5.5, 5.5))
    )

    resolved_path = _resolve_save_path(save_path)
    if resolved_path:
        fig.save(str(resolved_path), dpi=dpi, verbose=False)
    if show:
        fig.draw(show=True)

    return fig

# Used for experiment runner plots
def plot_residuals_vs_predicted_single(
    model_name: str,
    predictions: ModelPrediction | GroupedModelPrediction,
    prediction_metrics: dict[str, float] | None,
    target_data: BaseDataset | NPBDataset,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> ggplot:
    """
    Plot residuals against predicted means for one model.

    Each point is a target observation, with predictive mean on the x-axis and residual on the y-axis.
    The red line is a LOWESS smooth of residuals versus predictions. 
    
    Good plots are centered around zero with a flat smooth and close to constant spread.
    """
    
    if isinstance(predictions, GroupedModelPrediction):
        preds = predictions.reorder_to_original()
    elif isinstance(predictions, ModelPrediction):
        preds = predictions
    else:
        raise TypeError("predictions must be ModelPrediction or GroupedModelPrediction")
    
    true_y = _as_1d(target_data.response)
    pred_mean = _as_1d(preds.mean)
    residuals = true_y - pred_mean
    
    residual_df = pd.DataFrame({
        "predicted": pred_mean,
        "residual": residuals,
    })
    
    subtitle = None
    if prediction_metrics is not None:
        subtitle = (
            f"CRPS: {prediction_metrics['crps']:.3f} | RMSE: {prediction_metrics['rmse']:.3f}"
        )
    
    lowess_smooth = lowess(residual_df['residual'], residual_df['predicted'], frac=0.3)
    lowess_df = pd.DataFrame(lowess_smooth, columns=["predicted", "residual"])

    model_display_name = _display_model_name(model_name)
    model_color = PAPER_PALETTE["Validation Prediction"]

    fig = (
        ggplot(residual_df, aes(x="predicted", y="residual"))
        + geom_hline(yintercept=0, linetype="dashed", color="#6B7280", size=0.8)
        + geom_point(alpha=0.25, size=0.8, color=model_color)
        + geom_line(lowess_df, aes(x="predicted", y="residual"), color=PAPER_PALETTE["Truth"], size=1.0)
        + labs(
            title=f"{model_display_name} Residual vs. Predicted",
            subtitle=subtitle,
            x="Predicted value",
            y="Residual (true - predicted)",
        )
        + PAPER_THEME
        + theme(figure_size=(7.0, 6.0), legend_position="none")
    )
    resolved_path = _resolve_save_path(save_path)
    if resolved_path:
        fig.save(str(resolved_path), dpi=dpi, verbose=False)
    if show:
        fig.draw(show=True)
    return fig

# Used for experiment runner plots
def plot_single_model_predictions(
    model_name: str,
    predictions: ModelPrediction | GroupedModelPrediction,
    prediction_metrics: dict[str, float] | None,
    target_data: BaseDataset | NPBDataset,
    context_data: BaseDataset | NPBDataset,
    task: BaseTask,
    groups: list | None = None,
    n_groups: int = 10,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
    seed: int = 2,
    eval_samples: int = 20,
    alpha: float = 0.05,
    is_gaussian_prediction: bool = False,
    add_group_metrics: bool = True,
    true_fixed_effect: np.ndarray | pd.Series | None = None,
    true_random_effect: np.ndarray | pd.Series | None = None,
    prediction_split_name: str = "Validation",
) -> ggplot:
    """
    Plot one model's predictive mean, (1-alpha) predictive interval, and observed points by group.

    Each facet is a selected group. 
    Points are the context and the target set.
    The line is the prediction on the target features, given by the mean prediction of the model
    The ribbon is a confidence interval from either predictive samples or Gaussian quantiles.
    Optional black lines show true fixed and fixed-plus-random effects. 
    
    Use this to inspect model fit, uncertainty width, and group-specific failures.
    """
    rng = np.random.default_rng(seed)

    if isinstance(predictions, GroupedModelPrediction):
        preds = predictions.reorder_to_original()
    elif isinstance(predictions, ModelPrediction):
        preds = predictions
    else:
        raise TypeError("predictions must be ModelPrediction or GroupedModelPrediction")

    model_display_name = _display_model_name(model_name)
    model_color = PAPER_PALETTE["Validation Prediction"]
    prediction_label = f"{prediction_split_name} Prediction"
    performance_label = f"{prediction_split_name} Performance"

    ci_upper = f"upper_{1 - alpha:.2f}"
    ci_lower = f"lower_{1 - alpha:.2f}"
    z_score = float(stats.norm.ppf(1 - alpha / 2))

    if task.name == IN_CONTEXT_TASK_NAME:
        context_label = "Train Point"
        target_label = f"{prediction_split_name} Point"
    else:
        context_label = "Support Point"
        target_label = "Target Point"

    subtitle = None
    if prediction_metrics is not None:
        subtitle = (
            f"{performance_label}\n"
            f"CRPS: {prediction_metrics['crps']:.3f} | RMSE: {prediction_metrics['rmse']:.3f}"
        )

    def _line_df_from_samples(group, group_mask, x_vals) -> pd.DataFrame:
        dist = preds.predictive_distributions[group]
        sample = _sample_distribution(dist, torch.Size([eval_samples]), seed=seed)
        sample = (
            torch.flatten(sample, start_dim=0, end_dim=1)
            .squeeze(-1)
            .squeeze(1)
            .to("cpu")
            .numpy()
        )
        mean_vals = _as_1d(preds.mean[group_mask])
        return pd.DataFrame(
            {
                "Group": group,
                "x": x_vals,
                "mean": mean_vals,
                ci_upper: np.quantile(sample, q=1 - alpha / 2, axis=0),
                ci_lower: np.quantile(sample, q=alpha / 2, axis=0),
            }
        )

    def _line_df_from_std(group, group_mask, x_vals) -> pd.DataFrame:
        mean_vals = _as_1d(preds.mean[group_mask])
        std_vals = _as_1d(preds.std[group_mask])
        return pd.DataFrame(
            {
                "Group": group,
                "x": x_vals,
                "mean": mean_vals,
                ci_upper: mean_vals + z_score * std_vals,
                ci_lower: mean_vals - z_score * std_vals,
            }
        )

    def _single_group_truth_frame(
        dataset: BaseDataset | NPBDataset,
        truth_values: np.ndarray,
        group,
        line_type: str,
    ) -> pd.DataFrame:
        group_mask = np.asarray(dataset.grouping_var) == group
        if not np.any(group_mask):
            return pd.DataFrame(columns=["Group", "x", "y", "LineType"])

        x_vals = dataset.features.iloc[group_mask].to_numpy().reshape(-1)
        y_vals = np.asarray(truth_values)[group_mask].reshape(-1)

        return pd.DataFrame(
            {
                "Group": group,
                "x": x_vals,
                "y": y_vals,
                "LineType": line_type,
            }
        ).sort_values("x")

    def _build_point_df(group) -> pd.DataFrame:
        context_df = (
            _single_group_point_frame(context_data, group, context_label)
            .assign(Group=group, Split="Context", PointType=context_label)
        )
        target_df = (
            _single_group_point_frame(target_data, group, target_label)
            .assign(Group=group, Split="Target", PointType=target_label)
        )
        return pd.concat([context_df, target_df], ignore_index=True)

    if groups is None:
        all_groups = np.unique(target_data.grouping_var)
        groups = rng.choice(
            all_groups,
            size=min(n_groups, len(all_groups)),
            replace=False,
        ).tolist()

    def _compute_group_metric_df() -> pd.DataFrame:
        with isolated_torch_rng(seed):
            metric_df = groupwise_metrics(
                predictions=preds,
                target_data=target_data,
                group_ids=np.asarray(preds.group_ids),
            )
        return metric_df[metric_df[GROUPING_COLUMN_NAME].isin(groups)].copy()

    build_line_df = _line_df_from_std if is_gaussian_prediction else _line_df_from_samples

    point_dfs = []
    line_dfs = []
    true_line_dfs = []
    selected_groups = []

    tf = np.asarray(true_fixed_effect).reshape(-1) if true_fixed_effect is not None else None
    tr = np.asarray(true_random_effect).reshape(-1) if true_random_effect is not None else None

    n_context = len(context_data.grouping_var)
    n_target = len(target_data.grouping_var)

    tf_context = tf[:n_context] if tf is not None else None
    tf_target = tf[n_context:n_context + n_target] if tf is not None else None

    tr_context = tr[:n_context] if tr is not None else None
    tr_target = tr[n_context:n_context + n_target] if tr is not None else None

    for group in groups:
        group_mask = preds.group_ids == group

        if not np.any(group_mask):
            continue

        x_vals = target_data.features.iloc[group_mask].to_numpy().reshape(-1)

        point_dfs.append(_build_point_df(group))
        line_dfs.append(build_line_df(group, group_mask, x_vals).sort_values("x"))

        if tf is not None:
            true_line_dfs.append(
                _single_group_truth_frame(
                    context_data,
                    tf_context,
                    group,
                    "True fixed effect",
                )
            )
            true_line_dfs.append(
                _single_group_truth_frame(
                    target_data,
                    tf_target,
                    group,
                    "True fixed effect",
                )
            )

        if tf is not None and tr is not None:
            true_line_dfs.append(
                _single_group_truth_frame(
                    context_data,
                    tf_context + tr_context,
                    group,
                    "True fixed + random effect",
                )
            )
            true_line_dfs.append(
                _single_group_truth_frame(
                    target_data,
                    tf_target + tr_target,
                    group,
                    "True fixed + random effect",
                )
            )

        selected_groups.append(group)

    if not line_dfs:
        raise ValueError(f"No prediction data found for selected groups: {groups}")

    point_df = pd.concat(point_dfs, ignore_index=True)
    line_df = pd.concat(line_dfs, ignore_index=True)
    true_line_df = (
        pd.concat([df for df in true_line_dfs if not df.empty], ignore_index=True)
        if true_line_dfs
        else None
    )

    n_panels = len(line_df["Group"].unique())
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    point_color_map = {
        context_label: PAPER_PALETTE[CONTEXT_NAME],
        target_label: PAPER_PALETTE[TARGET_NAME],
    }

    point_shape_map = {
        context_label: "o",
        target_label: "^",
    }

    point_alpha_map = {
        context_label: 0.30,
        target_label: 0.9,
    }

    point_size_map = {
        context_label: 1.2,
        target_label: 2.2,
    }

    line_type_map = {
        prediction_label: "solid",
        "True fixed effect": "dotted",
        "True fixed + random effect": "dashed",
    }

    fig = (
        ggplot(line_df, aes(x="x"))
        + geom_ribbon(
            aes(ymin=ci_lower, ymax=ci_upper),
            fill=model_color,
            alpha=0.15,
            color=None,
        )
        + geom_line(
            aes(y="mean", linetype=f'"{prediction_label}"'),
            color=model_color,
            size=1.1,
        )
        + geom_point(
            data=point_df,
            mapping=aes(
                x="x",
                y="y",
                color="PointType",
                shape="PointType",
                alpha="PointType",
                size="PointType",
            ),
        )
        + facet_wrap("~Group", nrow=n_rows, ncol=n_cols)
        + labs(
            title=f"{model_display_name} Predictions and Uncertainty ({task.name.replace('_', ' ').title()})",
            subtitle=subtitle,
            x="Feature (x)",
            y="Response (y)",
            color="",
            shape="",
            alpha="",
            size="",
            linetype="",
        )
        + scale_color_manual(values=point_color_map)
        + scale_shape_manual(values=point_shape_map)
        + scale_alpha_manual(values=point_alpha_map)
        + scale_size_manual(values=point_size_map)
        + scale_linetype_manual(values=line_type_map)
        + PAPER_THEME
        + theme(
            legend_position="top",
            figure_size=(max(4.0 * n_cols, 8), max(3.8 * n_rows + 1.6, 6.5)),
            plot_title_position="plot",
        )
    )

    if true_line_df is not None:
        fig = (
            fig
            + geom_line(
                data=true_line_df,
                mapping=aes(x="x", y="y", linetype="LineType"),
                color="black",
                inherit_aes=False,
                size=0.6,
                alpha=0.95,
            )
            + guides(
                color=guide_legend(order=1),
                shape=guide_legend(order=1),
                alpha=guide_legend(order=1),
                size=guide_legend(order=1),
                linetype=guide_legend(
                    title=None,
                    order=2,
                    override_aes={
                        "color": [
                            model_color,
                            "black",
                            "black",
                        ]
                    },
                ),
            )
        )
    else:
        fig = fig + guides(
            color=guide_legend(order=1),
            shape=guide_legend(order=1),
            alpha=guide_legend(order=1),
            size=guide_legend(order=1),
            linetype=guide_legend(title=None, order=2),
        )

    if add_group_metrics:
        metric_df = _compute_group_metric_df()

        pos_rows = []
        for group in selected_groups:
            gdf = line_df[line_df["Group"] == group]
            x_min, x_max = gdf["x"].min(), gdf["x"].max()

            y_min = min(
                gdf[ci_lower].min(),
                point_df.loc[point_df["Group"] == group, "y"].min(),
            )
            y_max = max(
                gdf[ci_upper].max(),
                point_df.loc[point_df["Group"] == group, "y"].max(),
            )

            if true_line_df is not None:
                tgdf = true_line_df[true_line_df["Group"] == group]
                if not tgdf.empty:
                    y_min = min(y_min, tgdf["y"].min())
                    y_max = max(y_max, tgdf["y"].max())

            row = metric_df.loc[metric_df[GROUPING_COLUMN_NAME] == group].iloc[0]
            pos_rows.append(
                {
                    "Group": group,
                    "x": x_max - 0.03 * (x_max - x_min if x_max > x_min else 1.0),
                    "y": y_max - 0.05 * (y_max - y_min if y_max > y_min else 1.0),
                    "label": f"CRPS={row['crps']:.3f}\nRMSE={row['rmse']:.3f}",
                }
            )

        ann_df = pd.DataFrame(pos_rows)

        fig = fig + geom_text(
            data=ann_df,
            mapping=aes(x="x", y="y", label="label"),
            inherit_aes=False,
            ha="right",
            va="top",
            size=7,
        )

    resolved_path = _resolve_save_path(save_path)
    if resolved_path:
        fig.save(str(resolved_path), dpi=dpi, verbose=False)
    if show:
        fig.draw(show=True)

    return fig

# Used for experiment runner plots
def plot_variance_components(
    model_name: str,
    predictions: ModelPrediction | GroupedModelPrediction,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> ggplot:
    """
    Plot the variance of predicted fixed and random components across rows.

    The bars are the variance of the fixed effect predictions and
    the variance of the random effect predictions.
    
    A near-zero fixed-effect bar means the fixed-effect learner predicts an
    almost constant value across the observed feature range, so essentially all
    variation in the model's predictions stems from the random-effect learner.
    """
    # Get standardized model display name
    display_name = _display_model_name(model_name)
    
    # Calculate fixed effect component
    fe = predictions.fixed_effect
    if fe is None:
        raise ValueError("Fixed effect predictions are required for variance decomposition.")
        
    # Calculate random effect component
    re = predictions.random_effect
    if re is None:
        raise ValueError("Random effect predictions are required for variance decomposition.")
        
    var_f = float(np.var(fe))
    var_g = float(np.var(re))
    
    df = pd.DataFrame([
        {"Model": display_name, "Component": "Fixed Effects (f)", "Variance": var_f},
        {"Model": display_name, "Component": "Random Effects (g)", "Variance": var_g}
    ])
    
    # Custom colors matching the aesthetic style
    fill_colors = {
        "Fixed Effects (f)": "#4C72B0",  # Muted Blue
        "Random Effects (g)": "#DD8452"   # Muted Orange
    }
    
    fig = (
        ggplot(df, aes(x="Model", y="Variance", fill="Component"))
        + geom_col(position="stack", width=0.4, alpha=0.9)
        + scale_fill_manual(values=fill_colors)
        + labs(
            title=f"{display_name} Variance Decomposition",
            subtitle="Contribution of Fixed (f) vs. Random (g) effects",
            x="",
            y="Variance of Predicted Values",
            fill=""
        )
        + PAPER_THEME
        + theme(
            figure_size=(6.0, 5.0),
            legend_position="top"
        )
    )
    
    resolved_path = _resolve_save_path(save_path)
    if resolved_path:
        fig.save(str(resolved_path), dpi=dpi, verbose=False)
    if show:
        fig.draw(show=True)
        
    return fig

# Used for experiment runner plots
def plot_fixed_effect_density(
    model_name: str,
    predictions: ModelPrediction | GroupedModelPrediction,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> ggplot:
    """
    Plot the density of predicted fixed-effect values across target rows.

    The x-axis is each row's fixed effect prediction and the y-axis is Kernel Density Estimate (KDE). 
    
    If fixed effects are absent or almost constant, the plot shows a message instead. 
    Use this to detect a collapsed fixed effect or very concentrated fixed-effect predictions.
    """
    
    display_name = _display_model_name(model_name)
    fe = predictions.fixed_effect
    
    if fe is None or np.std(_as_1d(fe)) < 1e-5:
        plot_df = pd.DataFrame({"x": [0], "y": [0], "text": ["No non-constant fixed effects predicted"]})
        fig = (
            ggplot(plot_df, aes(x="x", y="y"))
            + geom_text(aes(label="text"), size=12, color="gray")
            + labs(
                title=f"{display_name} Density of Predicted Fixed Effects",
                subtitle="Constant or zero fixed effects",
                x="",
                y=""
            )
            + PAPER_THEME
            + theme(
                figure_size=(7.0, 5.0),
                panel_grid_major=element_blank(),
                panel_grid_minor=element_blank(),
                axis_text_x=element_blank(),
                axis_text_y=element_blank()
            )
        )
    else:
        fe = _as_1d(fe)
        plot_df = pd.DataFrame({
            "FixedEffect": fe,
            "Model": display_name
        })
        
        # Build palette, incorporating GPLinear/LME
        palette_colors = {
            NPBOOST_NAME: "#4E79A7",
            ANPBOOST_NAME: "#4E79A7",
            NP_NAME: "#F28E2B",
            ANP_NAME: "#F28E2B",
            "GPLinear": "#76B7B2",
            "LME": "#B07AA1",
            "Truth": "#D1495B"
        }
        model_color = PAPER_PALETTE["Validation Prediction"]
        
        fig = (
            ggplot(plot_df, aes(x="FixedEffect"))
            + geom_density(color=model_color, fill=model_color, alpha=0.15, size=1.0)
            + labs(
                title=f"{display_name} Density of Predicted Fixed Effects",
                x="Predicted Fixed Effect Value",
                y="Density"
            )
            + PAPER_THEME
            + theme(
                figure_size=(7.0, 5.0),
                legend_position="none"
            )
        )
        
    resolved_path = _resolve_save_path(save_path)
    if resolved_path:
        fig.save(str(resolved_path), dpi=dpi, verbose=False)
    if show:
        fig.draw(show=True)
        
    return fig

# Used for experiment runner plots
def plot_fixed_effect_vs_response(
    val_preds: ModelPrediction | GroupedModelPrediction,
    val_target: BaseDataset | NPBDataset,
    test_preds: ModelPrediction | GroupedModelPrediction,
    test_target: BaseDataset | NPBDataset,
    model_name: str,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> ggplot:
    """
    Plot predicted fixed effects against observed responses for validation and test.

    Each point is one row, with fixed effect predictions on the x-axis and the
    true response on the y-axis. 
    Facets separate validation and test.
    The dashed line is a linear smooth within each split. 
    
    Strong alignment means that the fixed effect explains the response quite well.
    """
    
    # Reorder grouped predictions if necessary
    if hasattr(val_preds, "reorder_to_original"):
        val_preds = val_preds.reorder_to_original()
    if hasattr(test_preds, "reorder_to_original"):
        test_preds = test_preds.reorder_to_original()
        
    if val_preds.fixed_effect is None or test_preds.fixed_effect is None:
        raise ValueError("Predictions must contain fixed_effect values to plot fixed effect vs. response.")
        
    val_fe = _as_1d(val_preds.fixed_effect)
    val_y = _as_1d(val_target.response)
    test_fe = _as_1d(test_preds.fixed_effect)
    test_y = _as_1d(test_target.response)
    
    df_val = pd.DataFrame({
        "PredictedFixedEffect": val_fe,
        "TrueResponse": val_y,
        "Split": "Validation"
    })
    df_test = pd.DataFrame({
        "PredictedFixedEffect": test_fe,
        "TrueResponse": test_y,
        "Split": "Test"
    })
    df = pd.concat([df_val, df_test], ignore_index=True)
    
    model_display_name = _display_model_name(model_name)
    model_color = PAPER_PALETTE["Validation Prediction"]
    
    fig = (
        ggplot(df, aes(x="PredictedFixedEffect", y="TrueResponse"))
        + geom_point(alpha=0.3, color=model_color, size=1.0)
        + geom_smooth(method="lm", color="#D1495B", linetype="dashed", size=1.0)
        + facet_wrap("~Split", nrow=1)
        + labs(
            title=f"Predicted Fixed Effect vs. True Response ({model_display_name})",
            x="Predicted Fixed Effect (f(x))",
            y="True Response (y)"
        )
        + PAPER_THEME
        + theme(
            figure_size=(9.0, 5.0),
            legend_position="none"
        )
    )
    
    resolved_path = _resolve_save_path(save_path)
    if resolved_path:
        fig.save(str(resolved_path), dpi=dpi, verbose=False)
    if show:
        fig.draw(show=True)
        
    return fig
