"""Utilities for collecting runs and producing experiment result tables.

Reporting pipeline
------------------
1. ``fetch_runs_api`` or ``raw_results.csv`` provides one row per W&B run.
2. ``calculate_grouped_performance`` selects the best validation run per model,
   result group, and seed using final validation RMSE, then averages the
   selected test metrics.
3. ``generate_performance_report`` returns that aggregated frame as
   ``final_results.csv`` and renders it independently as notebook HTML and
   single-experiment LaTeX.
4. ``write_main_text_performance_tables_from_local_results`` builds the paper's
   compact tables from saved ``*/final_results.csv`` files.

The central rule is validation selection: within each model, result group, and
data split seed, the run with the best final validation RMSE is selected before
test metrics are summarized. LaTeX export never refetches W&B runs; it formats
an already aggregated result table.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from IPython.display import display

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import (
    FEW_SHOT_TASK_NAME,
    IN_CONTEXT_TASK_NAME,
    SELECTED_MODEL_PARAMS_PER_MODEL,
)


MODELS = [
    "npboost",
    "anpboost",
    "np",
    "anp",
    "gbm_group_cat",
    "gbm_no_group",
    "gplinear",
    "lme",
    "tabicl_taskwise",
    "tabicl_pooled",
]
TASKS = [IN_CONTEXT_TASK_NAME, FEW_SHOT_TASK_NAME]

# Type alias for digits input to formatting functions. It can be an int or a mapping from labels to ints (if some experiments should get a different number of digits).
DigitsSpec = int | Mapping[str, int]

SEED_COL_NAME = "data_split_seed"
TASK_COL_NAME = "task_name"
VALIDATION_PERFORMANCE_COL_NAME = "validation_performance"

SYNTHETIC_GROUPING_COLS = ["fixed_effect_type", "feature_dimension", TASK_COL_NAME]
REAL_GROUPING_COLS = [TASK_COL_NAME]
FIXED_EFFECT_TYPES = ["zero", "steps"]
MODELS_ALLOWING_MISSING_CRPS = {"gbm_group_cat", "gbm_no_group"}
STANDARD_REPORT_METRICS = ("rmse", "crps")
PAPER_MODEL_NAMES = {
    "npboost": "NPBoost",
    "anpboost": "ANPBoost",
    "np": "NP",
    "anp": "ANP",
    "gbm_group_cat": "Task-ID GBT",
    "gbm_no_group": "GBT",
    "gplinear": "GPLinear",
    "lme": "LME",
    "tabicl_taskwise": "TabICL_T",
    "tabicl_pooled": "TabICL_P",
}
PAPER_TASK_NAMES = {
    IN_CONTEXT_TASK_NAME: "within-task",
    "in-context": "within-task",
    FEW_SHOT_TASK_NAME: "few-shot",
    "few-shot": "few-shot",
}
MAIN_TEXT_SYNTHETIC_MODELS = [
    "npboost",
    "anpboost",
    "np",
    "anp",
    "gbm_group_cat",
    # "gbm_no_group",
    "gplinear",
    "lme",
    "tabicl_taskwise",
    "tabicl_pooled",
]
MAIN_TEXT_REAL_MODELS = [
    "npboost",
    "anpboost",
    "np",
    "anp",
    "gbm_group_cat",
    # "gbm_no_group",
    "gplinear",
    "lme",
    "tabicl_taskwise",
    "tabicl_pooled",
]
MAIN_TEXT_SYNTHETIC_TABLE_CAPTION = (
    r"\caption{\textbf{Predictive performance across synthetic datasets.} "
    r"Each entry reports the mean RMSE / mean CRPS over five seeds, with "
    r"standard errors shown in parentheses below in the same RMSE / CRPS order; "
    r"lower values indicate better performance. The best result in each row for "
    r"each metric is shown in bold, and \textemdash{} indicates an unavailable metric.}"
)
MAIN_TEXT_REAL_TABLE_CAPTION = (
    r"\caption{\textbf{Predictive performance across real datasets.} "
    r"Each entry reports the mean RMSE / mean CRPS over five seeds, with "
    r"standard errors shown in parentheses below in the same RMSE / CRPS order; "
    r"lower values indicate better performance. The best result in each row for "
    r"each metric is shown in bold, and \textemdash{} indicates an unavailable metric.}"
)

logger = logging.getLogger(__name__)


def get_nested_keys(data_dict: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Fetch a value from a nested dict using a dot-notation path."""
    value = data_dict
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _warn_duplicate_run_ids(df: pd.DataFrame) -> None:
    """Log duplicated W&B run IDs and whether their extracted rows match."""
    if df.empty or "run_id" not in df.columns:
        return

    duplicate_mask = df["run_id"].duplicated(keep=False)
    if not duplicate_mask.any():
        return

    for run_id, duplicate_rows in df.loc[duplicate_mask].groupby("run_id", dropna=False):
        duplicate_rows = duplicate_rows.reset_index(drop=True)
        first_row = duplicate_rows.iloc[[0]]
        rows_identical = all(
            duplicate_rows.iloc[[idx]].reset_index(drop=True).equals(first_row)
            for idx in range(1, len(duplicate_rows))
        )
        logger.warning(
            f"Fetched W&B run_id {run_id} {len(duplicate_rows)} times; "
            f"extracted rows identical={rows_identical}"
        )


def fetch_runs_api(
    project: str,
    wandb_api: Any | None = None,
    extra_config_fields: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch finished W&B runs for a project and return one row per run.

    ``extra_config_fields`` maps output column names to dot-notation config
    paths, which is useful for variants such as context-size experiments.
    """
    if wandb_api is None:
        print("Initializing W&B API")
        import wandb

        # Set a higher timeout (90s) to prevent GraphQL dropouts
        wandb_api = wandb.Api(timeout=90)
    print("Fetching runs from W&B. This may take up to 30 minutes for large W&B projects.")
    # Fetch all runs so unfinished/crashed runs can be reported and skipped
    # before metric extraction.
    runs = wandb_api.runs(project, per_page=100)
    finished_runs = []
    for run in runs:
        run_id = run.id
        run_name = run.name
        run_state = run.state
        if run_state == "finished":
            finished_runs.append(run)
        else:
            logger.warning(
                f"Skipping unfinished W&B run {run_id} ({run_name}): state={run_state}"
            )

    if not finished_runs:
        return pd.DataFrame()

    experiment_type = finished_runs[0].config["data"]["type"]

    # 3. Helper worker to process single run inside threads
    def _fetch_single_run(run):
        try:
            config = run.config
            summary = run.summary._json_dict if hasattr(run.summary, "_json_dict") else dict(run.summary)

            model_name = config["model"]["name"]
            validation_summary = summary.get("VALIDATION", {})
            test_summary = summary.get("TEST", {})
            allows_missing_crps = model_name in MODELS_ALLOWING_MISSING_CRPS
            # Run names are configured such that the second hyphen-separated part is the task name.
            task_name = run.name.split("-")[1]

            row = {
                "name": run.name,
                "run_id": run.id,
                "created_at": run.created_at,
                "model_name": model_name,
                "task_name": task_name,
                "data_split_seed": config["data"]["experiment"]["split_data_seed"],
                "validation_metric": config["model"]["validation_metric"],
                "validation_crps": (
                    validation_summary.get("crps", np.nan)
                    if allows_missing_crps
                    else validation_summary["crps"]
                ),
                "validation_rmse": validation_summary["rmse"],
                "test_crps": (
                    test_summary.get("crps", np.nan)
                    if allows_missing_crps
                    else test_summary["crps"]
                ),
                "test_rmse": test_summary["rmse"],
                "fitting_time_s": summary.get("fitting_time_s", np.nan),
                "validation_prediction_time_s": summary.get("validation_prediction_time_s", np.nan),
                "test_prediction_time_s": summary.get("test_prediction_time_s", np.nan),
                "validation_fixed_effect_residual_rmse": validation_summary.get(
                    "fixed_effect_residual_rmse",
                    summary.get("val_fixed_effect_residual_rmse", np.nan),
                ),
            }

            if experiment_type == "synthetic":
                fe_name = config["data"]["synthetic"]["fixed_effect"]["name"]
                fe_parts = fe_name.split("_")
                row["fixed_effect_type"] = "_".join(fe_parts[:-1])
                row["feature_dimension"] = int(fe_parts[-1].rstrip("D"))

            if extra_config_fields is not None:
                for column, path in extra_config_fields.items():
                    row[column] = get_nested_keys(config, path)

            for param in SELECTED_MODEL_PARAMS_PER_MODEL.get(model_name, []):
                value = get_nested_keys(config, param)
                row[param] = value if value is not None else get_nested_keys(summary, param)

            return row
        except KeyError as exc:
            print(f"Skipping {run.name}: missing key {exc}")
            return None

    # 4. Stream tasks directly from run_generator() into threads.
    rows = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_fetch_single_run, run) for run in finished_runs]
        for future in tqdm(as_completed(futures), desc="Processing W&B Runs", unit="run"):
            res = future.result()
            if res is not None:
                rows.append(res)

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    _warn_duplicate_run_ids(results)
    return _prepare_results_frame(results)


def calculate_grouped_performance(
    model_results: pd.DataFrame,
    model_name: str,
    best_hyperparams_save_dir: str | Path | None = None,
    show_best_hyperparams: bool = True,
    grouping_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return validation-selected test summaries for one model.

    For each result group and seed, the best run is selected by final validation
    RMSE. Selection by CRPS or fixed-effect residual RMSE is not supported here
    at the moment.

    Returns:
        One row per result group with test mean, standard deviation, standard
        error, and sample count.
    """
    grouping = _grouping_levels(model_results, grouping_cols)
    grouping_with_seed = grouping + [SEED_COL_NAME]
    validation_col = "validation_rmse"
    if validation_col not in model_results.columns or model_results[validation_col].dropna().empty:
        raise ValueError(
            f"No usable validation metrics found for model '{model_name}'. "
            "Selection currently requires a non-missing validation_rmse column."
        )

    best_rows = _select_best_validation_rmse_rows(model_results, grouping_with_seed)

    agg = (
        best_rows.groupby(grouping, observed=True)
        .agg(
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_rmse_n=("test_rmse", "count"),
            test_crps_mean=("test_crps", "mean"),
            test_crps_std=("test_crps", "std"),
            test_crps_n=("test_crps", "count"),
        )
        .reset_index()
    )
    agg["test_rmse_se"] = agg["test_rmse_std"] / np.sqrt(agg["test_rmse_n"])
    agg["test_crps_se"] = agg["test_crps_std"] / np.sqrt(agg["test_crps_n"])

    if best_hyperparams_save_dir or show_best_hyperparams:
        # Save best hyperparameters to disk and/or display them in the notebook.
        _handle_best_hyperparameters(
            best_rows=best_rows,
            model_name=model_name,
            grouping=grouping,
            best_hyperparams_save_dir=best_hyperparams_save_dir,
            show_best_hyperparams=show_best_hyperparams,
        )

    return agg


def generate_performance_report(
    wandb_project_name: str,
    use_local: bool = False,
    save_dir: str | Path | None = None,
    show_best_hyperparams: bool = True,
    grouping_cols: Sequence[str] | None = None,
    extra_config_fields: Mapping[str, str] | None = None,
    exclude_values: Mapping[str, Sequence[Any]] | None = None,
    digits: int = 4,
) -> pd.DataFrame:
    """Generate and optionally save performance report files for one project.

    Args:
        wandb_project_name: W&B project name used when ``use_local`` is false.
        use_local: Load ``raw_results.csv`` from ``save_dir`` instead of
            fetching runs from W&B.
        save_dir: Directory for raw, aggregated, HTML, and LaTeX files.
            If omitted, files are not written.
        show_best_hyperparams: Display selected seed-level hyperparameters.
        grouping_cols: Aggregation columns. Defaults to synthetic grouping
            ``fixed_effect_type, feature_dimension, task_name`` when those
            columns exist, otherwise ``task_name``.
        extra_config_fields: Extra W&B config paths to extract into raw rows.
        exclude_values: Values to filter out before model aggregation, for example
            ``{"model_name": ["npboost"]}`` drops all rows where the model name is "npboost".
        digits: Decimal places (used only by display/LaTeX renderers).

    The raw rows are selected by validation metric inside
    ``calculate_grouped_performance`` before test metrics are averaged.
    ``final_results_table.html`` and ``final_results_table.tex`` are both
    rendered from the same ``all_stats`` DataFrame returned by this function;
    the LaTeX path does not recalculate metrics or reuse the styled HTML object.

    Saves:
      - raw_results.csv: one row per run
      - final_results.csv: aggregated mean/std/se per configured grouping
      - final_results_table.html: styled version of the aggregated table
      - final_results_table.tex: publication-ready LaTeX table
    """
    local_folder = Path(save_dir) if save_dir is not None else None
    raw_path = local_folder / "raw_results.csv" if local_folder is not None else None

    if local_folder is not None:
        local_folder.mkdir(parents=True, exist_ok=True)

    if use_local:
        if raw_path is None:
            raise ValueError("save_dir must be provided when use_local is True.")
        if not raw_path.exists():
            raise FileNotFoundError(f"Local results not found at {raw_path}.")
        results_df = _prepare_results_frame(pd.read_csv(raw_path))
    else:
        results_df = fetch_runs_api(
            wandb_project_name,
            extra_config_fields=extra_config_fields,
        )
        if raw_path is not None:
            results_df.to_csv(raw_path, index=False)

    results_df = _exclude_values(results_df, exclude_values)

    stats_per_model = []
    for model_name, group in results_df.groupby("model_name", observed=True):
        model_stats = calculate_grouped_performance(
            group,
            model_name,
            best_hyperparams_save_dir=local_folder,
            show_best_hyperparams=show_best_hyperparams,
            grouping_cols=grouping_cols,
        )
        model_stats.insert(0, "model_name", model_name)
        stats_per_model.append(model_stats)

    if not stats_per_model:
        raise ValueError("No model results available after loading and filtering.")

    all_stats = pd.concat(stats_per_model, ignore_index=True)
    styled = display_performance_report(
        all_stats,
        split_by_model=False,
        grouping_cols=grouping_cols,
        digits=digits,
    )

    if local_folder is not None:
        all_stats.to_csv(local_folder / "final_results.csv", index=False)
        styled.to_html(
            local_folder / "final_results_table.html",
            table_uuid=f"final_results-{local_folder.name}",
        )
        write_performance_report_latex(
            all_stats,
            local_folder / "final_results_table.tex",
            grouping_cols=grouping_cols,
            digits=digits,
        )

    return all_stats


def display_performance_report(
    combined_stats: pd.DataFrame,
    split_by_model: bool = False,
    grouping_cols: Sequence[str] | None = None,
    digits: int = 4,
):
    """Display a combined styled table, or one plain table per model."""
    if split_by_model:
        models_present = combined_stats["model_name"].unique()
        for model in (m for m in MODELS if m in models_present):
            print(f"\n--- Performance: {model} ---")
            display(
                format_model_results_table(
                    combined_stats[combined_stats["model_name"] == model],
                    grouping_cols=grouping_cols,
                    digits=digits,
                )
            )
        return None

    styled = format_all_models_results_for_display(
        combined_stats,
        grouping_cols=grouping_cols,
        digits=digits,
    )
    display(styled)
    return styled


def format_model_results_table(
    combined_results: pd.DataFrame,
    grouping_cols: Sequence[str] | None = None,
    digits: int = 3,
) -> pd.DataFrame:
    """Pivot one model into the compact row-group/statistic layout."""
    digits = _validate_digits(digits)
    df = _ensure_standard_errors(combined_results)
    grouping = _grouping_levels(df, grouping_cols)
    id_vars = [col for col in grouping if col in df.columns]

    melted = df.melt(
        id_vars=id_vars,
        value_vars=["test_rmse_mean", "test_rmse_se", "test_crps_mean", "test_crps_se"],
        var_name="metric_raw",
        value_name="Value",
    )
    melted["Metric"] = melted["metric_raw"].str.extract(r"(rmse|crps)")
    melted["Statistic"] = melted["metric_raw"].str.extract(r"(mean|se)")
    melted["Row_Group"] = melted[TASK_COL_NAME].astype(str) + "_" + melted["Metric"]

    non_task_grouping = [col for col in grouping if col != TASK_COL_NAME]
    pivot_index = non_task_grouping + ["Row_Group", "Statistic"]
    table = melted.pivot_table(index=pivot_index, values="Value", observed=True)

    row_order = ["in_context_rmse", "in_context_crps", "few_shot_rmse", "few_shot_crps"]
    table = (
        table
        .reindex(row_order, level="Row_Group")
        .reindex(["mean", "se"], level="Statistic")
    )

    return table.round(digits)


def highlight_best(row: pd.Series) -> list[str]:
    """Bold and color the minimum value in a display row."""
    def parse_mean(val: Any) -> float:
        match = re.match(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))", str(val))
        return float(match.group(1)) if match else np.nan

    values = row.map(parse_mean)
    finite_values = values[np.isfinite(values)]
    if finite_values.empty:
        return [""] * len(values)

    min_val = finite_values.min()
    return ["font-weight: bold; color: green" if value == min_val else "" for value in values]


def format_all_models_results_for_display(
    combined_results: pd.DataFrame,
    grouping_cols: Sequence[str] | None = None,
    digits: int = 4,
):
    """Format all models side-by-side and highlight each row's best value."""
    digits = _validate_digits(digits)
    df = _ensure_standard_errors(combined_results)

    df["RMSE"] = df.apply(
        lambda row: _format_mean_se(row.test_rmse_mean, row.test_rmse_se, digits),
        axis=1,
    )
    df["CRPS"] = df.apply(
        lambda row: _format_mean_se(row.test_crps_mean, row.test_crps_se, digits),
        axis=1,
    )

    grouping = _grouping_levels(df, grouping_cols)

    melted = df.melt(
        id_vars=grouping + ["model_name"],
        value_vars=["RMSE", "CRPS"],
        var_name="Metric_Label",
        value_name="Display_Value",
        ignore_index=False,
    )

    pivot_index = grouping + ["Metric_Label"]
    table = melted.pivot(index=pivot_index, columns="model_name", values="Display_Value")

    present_order = [metric for metric in ["RMSE", "CRPS"] if metric in melted["Metric_Label"].unique()]
    table = table.reindex(present_order, level="Metric_Label")

    models_ordered = [model for model in MODELS if model in table.columns]
    table = table.reindex(columns=models_ordered)

    return table.style.apply(highlight_best, axis=1, subset=table.columns)


def write_performance_report_latex(
    combined_results: pd.DataFrame,
    output_path: str | Path,
    *,
    grouping_cols: Sequence[str] | None = None,
    digits: int = 4,
) -> Path:
    """Write one experiment's aggregated report as a LaTeX table.

    The input is expected to be the ``all_stats``/``final_results.csv`` schema.
    This is a formatting step only: it may derive missing standard-error columns
    from ``std`` and ``n``, but it does not reselect runs or recompute means.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    latex = format_performance_report_as_latex(
        combined_results,
        grouping_cols=grouping_cols,
        digits=digits,
    )
    output.write_text(latex, encoding="utf-8")
    return output


def format_performance_report_as_latex(
    combined_results: pd.DataFrame,
    *,
    grouping_cols: Sequence[str] | None = None,
    digits: int = 4,
) -> str:
    """Return one experiment's aggregated report as a full LaTeX table string."""
    digits = _validate_digits(digits)
    df = _ensure_standard_errors(combined_results)
    if "fixed_effect_type" in df.columns:
        df["fixed_effect_type"] = pd.Categorical(
            df["fixed_effect_type"].astype(str),
            categories=FIXED_EFFECT_TYPES,
            ordered=True,
        )
    grouping = _grouping_levels(df, grouping_cols)
    is_synthetic_report = {"fixed_effect_type", "feature_dimension", TASK_COL_NAME}.issubset(grouping)

    model_order = [model for model in MODELS if model in set(df["model_name"].astype(str))]
    if not model_order:
        raise ValueError("No model columns available for LaTeX export.")

    if is_synthetic_report:
        lines = _format_synthetic_performance_latex(df, grouping, model_order, digits)
    else:
        lines = _format_real_performance_latex(df, grouping, model_order, digits)

    return "\n".join(lines) + "\n"


def write_main_text_performance_tables_from_local_results(
    result_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    synthetic_experiments: Mapping[str, str] | None = None,
    real_datasets: Mapping[str, str] | None = None,
    digits: int = 3,
) -> dict[str, Path]:
    """Write the main-text RMSE/CRPS tables from local ``final_results.csv`` files.

    This is the second LaTeX export path. It combines already aggregated
    per-experiment result files and renders compact paper tables. It never reads
    ``final_results_table.tex`` files and never returns to raw W&B runs.

    ``synthetic_experiments`` and ``real_datasets`` map result-directory names
    to LaTeX labels. When omitted, directories are detected by whether their
    ``final_results.csv`` contains synthetic fixed-effect columns. ``digits``
    can be one integer or a mapping from displayed dataset/experiment labels
    to digits, with optional ``"default"`` or ``"*"`` fallback keys. The output
    assumes the paper defines ``\\dataset`` and loads standard table packages
    such as booktabs, makecell, and nicematrix.
    """
    root = Path(result_root)
    out_dir = Path(output_dir) if output_dir is not None else root
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_frames, real_frames = _load_main_text_result_frames(
        root,
        synthetic_experiments=synthetic_experiments,
        real_datasets=real_datasets,
    )

    written = {}
    if synthetic_frames:
        synthetic_results = pd.concat(synthetic_frames, ignore_index=True)
        synthetic_path = out_dir / "main_text_synthetic_results.tex"
        synthetic_path.write_text(
            format_main_text_synthetic_performance_latex(synthetic_results, digits=digits),
            encoding="utf-8",
        )
        written["synthetic"] = synthetic_path

    if real_frames:
        real_results = pd.concat(real_frames, ignore_index=True)
        real_path = out_dir / "main_text_real_results.tex"
        real_path.write_text(
            format_main_text_real_performance_latex(real_results, digits=digits),
            encoding="utf-8",
        )
        written["real"] = real_path

    return written


def format_main_text_synthetic_performance_latex(
    combined_results: pd.DataFrame,
    *,
    model_order: Sequence[str] = MAIN_TEXT_SYNTHETIC_MODELS,
    digits: DigitsSpec = 3,
) -> str:
    """Return a compact synthetic paper table with ``RMSE / CRPS`` cells."""
    digits = _validate_digits_spec(digits)
    df = _prepare_main_text_results_frame(combined_results, label_col="experiment_label")
    models = [model for model in model_order if model in set(df["model_name"].astype(str))]
    col_spec = f"lll *{{{len(models)}}}{{c}}"
    header = " & ".join(
        [
            r"\textbf{Dataset}",
            r"\textbf{\(d\)}",
            r"\textbf{Scenario}",
            *[
                rf"\textbf{{{_latex_escape(PAPER_MODEL_NAMES.get(model, model))}}}"
                for model in models
            ],
        ]
    )

    lines = [
        r"\begin{table*}[h!]",
        r"\centering",
        MAIN_TEXT_SYNTHETIC_TABLE_CAPTION,
        r"\label{tab:synthetic_results_main}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{NiceTabular}}{{{col_spec}}}",
        r"\toprule",
        f"{header} \\\\",
        r"\midrule",
    ]

    experiment_values = _ordered_unique(df["experiment_label"])
    for experiment_index, experiment_label in enumerate(experiment_values):
        experiment_digits = _digits_for_label(digits, experiment_label)
        experiment_df = df[df["experiment_label"].astype(str) == str(experiment_label)]
        dimensions = _ordered_numeric_unique(experiment_df["feature_dimension"])
        n_experiment_rows = 2 * len(dimensions)

        for dimension_index, dimension in enumerate(dimensions):
            dimension_df = experiment_df[experiment_df["feature_dimension"].astype(float) == float(dimension)]
            for task_index, task_name in enumerate([IN_CONTEXT_TASK_NAME, FEW_SHOT_TASK_NAME]):
                row_df = dimension_df[dimension_df[TASK_COL_NAME].astype(str) == task_name]
                displayed_row_df = row_df[row_df["model_name"].astype(str).isin(models)]
                cells = [
                    rf"\Block{{{n_experiment_rows}-1}}{{\dataset{{{_latex_dataset_label(experiment_label)}}}}}"
                    if dimension_index == 0 and task_index == 0
                    else "",
                    rf"\Block{{2-1}}{{{int(float(dimension))}}}" if task_index == 0 else "",
                    _latex_escape(_paper_task_name(task_name)),
                    *[
                        _latex_compact_metric_cell(displayed_row_df, model, experiment_digits)
                        for model in models
                    ],
                ]
                lines.append(f"{' & '.join(cells)} \\\\")
            if dimension_index < len(dimensions) - 1:
                lines.append(f"\\cmidrule(lr){{2-{3 + len(models)}}}")
        if experiment_index < len(experiment_values) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{NiceTabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def format_main_text_real_performance_latex(
    combined_results: pd.DataFrame,
    *,
    model_order: Sequence[str] = MAIN_TEXT_REAL_MODELS,
    digits: int = 3,
) -> str:
    """Return a compact real-data paper table with ``RMSE / CRPS`` cells."""
    digits = _validate_digits_spec(digits)
    df = _prepare_main_text_results_frame(combined_results, label_col="dataset_label")
    models = [model for model in model_order if model in set(df["model_name"].astype(str))]
    col_spec = f"ll *{{{len(models)}}}{{c}}"
    header = " & ".join(
        [
            r"\textbf{Dataset}",
            r"\textbf{Scenario}",
            *[
                rf"\textbf{{{_latex_escape(PAPER_MODEL_NAMES.get(model, model))}}}"
                for model in models
            ],
        ]
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        MAIN_TEXT_REAL_TABLE_CAPTION,
        r"\label{tab:real_results_main}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{NiceTabular}}{{{col_spec}}}",
        r"\toprule",
        f"{header} \\\\",
        r"\midrule",
    ]

    dataset_values = _ordered_unique(df["dataset_label"])
    for dataset_index, dataset_label in enumerate(dataset_values):
        dataset_digits = _digits_for_label(digits, dataset_label)
        dataset_df = df[df["dataset_label"].astype(str) == str(dataset_label)]
        for task_index, task_name in enumerate([IN_CONTEXT_TASK_NAME, FEW_SHOT_TASK_NAME]):
            row_df = dataset_df[dataset_df[TASK_COL_NAME].astype(str) == task_name]
            displayed_row_df = row_df[row_df["model_name"].astype(str).isin(models)]
            cells = [
                rf"\Block{{2-1}}{{\dataset{{{_latex_dataset_label(dataset_label)}}}}}"
                if task_index == 0
                else "",
                _latex_escape(_paper_task_name(task_name)),
                *[
                    _latex_compact_metric_cell(displayed_row_df, model, dataset_digits)
                    for model in models
                ],
            ]
            lines.append(f"{' & '.join(cells)} \\\\")
        if dataset_index < len(dataset_values) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{NiceTabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_synthetic_performance_latex(
    df: pd.DataFrame,
    grouping: Sequence[str],
    model_order: Sequence[str],
    digits: int,
) -> list[str]:
    fixed_groups = (
        df[["fixed_effect_type", "feature_dimension"]]
        .drop_duplicates()
        .sort_values(["fixed_effect_type", "feature_dimension"])
        .itertuples(index=False, name=None)
    )
    n_cols = 3 + len(model_order)
    col_spec = "lll" + "c" * len(model_order)
    header = " & ".join(
        ["Fixed Effect", "Scenario", "Metric", *[_latex_escape(PAPER_MODEL_NAMES.get(model, model)) for model in model_order]]
    )

    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3pt}",
        "",
        r"    \resizebox{\textwidth}{!}{%",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        r"        \toprule",
        f"        {header} \\\\",
        r"        \midrule",
    ]

    fixed_groups = list(fixed_groups)
    for group_index, (fixed_effect_type, feature_dimension) in enumerate(fixed_groups):
        fixed_label = _latex_fixed_effect_label(fixed_effect_type, feature_dimension)
        for row_index, (task_name, metric) in enumerate(_performance_row_order()):
            row_df = df[
                (df["fixed_effect_type"].astype(str) == str(fixed_effect_type))
                & (df["feature_dimension"].astype(float) == float(feature_dimension))
                & (df[TASK_COL_NAME].astype(str) == task_name)
            ]
            cells = [
                f"\\multirow{{4}}{{*}}{{{fixed_label}}}" if row_index == 0 else "",
                _latex_escape(_paper_task_name(task_name)),
                metric.upper(),
                *[_latex_metric_cell(row_df, model, metric, digits) for model in model_order],
            ]
            lines.append(f"        {' & '.join(cells)} \\\\")
        if group_index < len(fixed_groups) - 1:
            lines.append(f"        \\cmidrule(lr){{1-{n_cols}}}")

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table*}",
        ]
    )
    return lines


def _format_real_performance_latex(
    df: pd.DataFrame,
    grouping: Sequence[str],
    model_order: Sequence[str],
    digits: int,
) -> list[str]:
    extra_grouping = [col for col in grouping if col != TASK_COL_NAME]
    col_spec = "l" * (1 + len(extra_grouping)) + "c" * (1 + len(model_order))
    first_headers = [_latex_header_for_grouping_col(col) for col in extra_grouping]
    header = " & ".join(
        [
            *first_headers,
            "Scenario",
            "Metric",
            *[_latex_escape(PAPER_MODEL_NAMES.get(model, model)) for model in model_order],
        ]
    )

    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3pt}",
        "",
        r"    \resizebox{\textwidth}{!}{%",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        r"        \toprule",
        f"        {header} \\\\",
        r"        \midrule",
    ]

    group_values = _latex_group_values(df, extra_grouping)
    for value_index, values in enumerate(group_values):
        group_mask = pd.Series(True, index=df.index)
        for col, value in zip(extra_grouping, values):
            group_mask &= df[col].astype(str) == str(value)

        for task_name, metric in _performance_row_order():
            row_df = df[group_mask & (df[TASK_COL_NAME].astype(str) == task_name)]
            group_cells = [_latex_escape(str(value)) for value in values]
            cells = [
                *group_cells,
                _latex_escape(_paper_task_name(task_name)),
                metric.upper(),
                *[_latex_metric_cell(row_df, model, metric, digits) for model in model_order],
            ]
            lines.append(f"        {' & '.join(cells)} \\\\")
        if extra_grouping and value_index < len(group_values) - 1:
            lines.append(f"        \\cmidrule(lr){{1-{2 + len(extra_grouping) + len(model_order)}}}")

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table*}",
        ]
    )
    return lines


def _performance_row_order() -> list[tuple[str, str]]:
    return [
        (IN_CONTEXT_TASK_NAME, "rmse"),
        (IN_CONTEXT_TASK_NAME, "crps"),
        (FEW_SHOT_TASK_NAME, "rmse"),
        (FEW_SHOT_TASK_NAME, "crps"),
    ]


def _latex_metric_cell(
    row_df: pd.DataFrame,
    model_name: str,
    metric: str,
    digits: int,
) -> str:
    model_rows = row_df[row_df["model_name"].astype(str) == model_name]
    if model_rows.empty:
        return "--"

    row = model_rows.iloc[0]
    mean = row.get(f"test_{metric}_mean", np.nan)
    se = row.get(f"test_{metric}_se", np.nan)
    if pd.isna(mean):
        return "--"

    mean_text = f"{mean:.{digits}f}"
    if _is_displayed_best(mean, row_df[f"test_{metric}_mean"], digits):
        mean_text = f"\\textbf{{{mean_text}}}"

    se_text = "nan" if pd.isna(se) else f"{se:.{digits}f}"
    return f"{mean_text}$\\pm${se_text}"


def _load_main_text_result_frames(
    root: Path,
    *,
    synthetic_experiments: Mapping[str, str] | None,
    real_datasets: Mapping[str, str] | None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    if synthetic_experiments is None and real_datasets is None:
        synthetic_frames = []
        real_frames = []
        for final_results_path in sorted(root.glob("*/final_results.csv")):
            frame = pd.read_csv(final_results_path)
            if {"fixed_effect_type", "feature_dimension"}.issubset(frame.columns):
                frame.insert(0, "experiment_label", final_results_path.parent.name)
                synthetic_frames.append(frame)
            else:
                frame.insert(0, "dataset_label", final_results_path.parent.name)
                real_frames.append(frame)
        return synthetic_frames, real_frames

    synthetic_frames = [
        _load_labeled_main_text_result(root, folder_name, "experiment_label", label)
        for folder_name, label in (synthetic_experiments or {}).items()
    ]
    real_frames = [
        _load_labeled_main_text_result(root, folder_name, "dataset_label", label)
        for folder_name, label in (real_datasets or {}).items()
    ]
    return synthetic_frames, real_frames


def _load_labeled_main_text_result(
    root: Path,
    folder_name: str,
    label_col: str,
    label: str,
) -> pd.DataFrame:
    path = root / folder_name / "final_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Local final results not found at {path}.")
    frame = pd.read_csv(path)
    frame.insert(0, label_col, label)
    return frame


def _prepare_main_text_results_frame(df: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    out = _ensure_standard_errors(df)
    required_cols = {
        label_col,
        "model_name",
        TASK_COL_NAME,
        "test_rmse_mean",
        "test_crps_mean",
    }
    missing = sorted(required_cols - set(out.columns))
    if missing:
        raise ValueError(f"Missing columns for compact LaTeX table: {missing}")

    out["model_name"] = pd.Categorical(out["model_name"].astype(str), categories=MODELS, ordered=True)
    out[TASK_COL_NAME] = pd.Categorical(
        out[TASK_COL_NAME].astype(str),
        categories=[IN_CONTEXT_TASK_NAME, FEW_SHOT_TASK_NAME],
        ordered=True,
    )
    return out


def _latex_compact_metric_cell(row_df: pd.DataFrame, model_name: str, digits: int) -> str:
    model_rows = row_df[row_df["model_name"].astype(str) == model_name]
    if model_rows.empty:
        return _latex_compact_metric_pair_cell(
            r"\textemdash{} / \textemdash{}",
            r"\textemdash{} / \textemdash{}",
        )

    row = model_rows.iloc[0]
    rmse = row.get("test_rmse_mean", np.nan)
    crps = row.get("test_crps_mean", np.nan)
    rmse_se = row.get("test_rmse_se", np.nan)
    crps_se = row.get("test_crps_se", np.nan)
    mean_line = (
        f"{_latex_compact_metric_value(rmse, row_df['test_rmse_mean'], digits)} / "
        f"{_latex_compact_metric_value(crps, row_df['test_crps_mean'], digits)}"
    )
    se_line = (
        f"{_latex_compact_se_value(rmse_se, digits)} / "
        f"{_latex_compact_se_value(crps_se, digits)}"
    )

    return _latex_compact_metric_pair_cell(mean_line, se_line)


def _latex_compact_metric_pair_cell(mean_line: str, se_line: str) -> str:
    """Keep RMSE/CRPS and their SEs inside one model cell without widening."""
    return rf"\makecell[cc]{{{mean_line}\\{{\scriptsize ({se_line})}}}}"


def _latex_compact_se_value(value: Any, digits: int) -> str:
    if pd.isna(value):
        return r"\textemdash{}"
    return f"{value:.{digits}f}"


def _latex_compact_metric_value(value: Any, comparison_values: pd.Series, digits: int) -> str:
    if pd.isna(value):
        return r"\textemdash{}"
    value_text = f"{value:.{digits}f}"
    if _is_displayed_best(value, comparison_values, digits):
        return f"\\textbf{{{value_text}}}"
    return value_text


def _is_displayed_best(value: Any, comparison_values: pd.Series, digits: int) -> bool:
    """Return whether ``value`` is tied for the displayed minimum."""
    value_display = _displayed_numeric_value(value, digits)
    if value_display is None:
        return False

    displayed_values = [
        displayed
        for displayed in (
            _displayed_numeric_value(comparison_value, digits)
            for comparison_value in comparison_values
        )
        if displayed is not None
    ]
    if not displayed_values:
        return False

    return value_display == min(displayed_values)


def _displayed_numeric_value(value: Any, digits: int) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric_value):
        return None

    return float(f"{numeric_value:.{digits}f}")


def _ordered_unique(series: pd.Series) -> list[Any]:
    return list(dict.fromkeys(series.astype(str).tolist()))


def _ordered_numeric_unique(series: pd.Series) -> list[Any]:
    values = pd.to_numeric(series, errors="coerce").dropna().drop_duplicates()
    return sorted(values.tolist())


def _latex_dataset_label(label: Any) -> str:
    # \dataset{...} is a paper macro, so keep spaces/dashes readable but do not
    # escape underscores as LaTeX commands inside the macro argument.
    return str(label).replace("_", "-")


def _latex_fixed_effect_label(fixed_effect_type: Any, feature_dimension: Any) -> str:
    fixed_effect = str(fixed_effect_type).replace("_", " ").title()
    dimension = int(float(feature_dimension))
    return f"{_latex_escape(fixed_effect)} ${dimension}$D"


def _paper_task_name(task_name: Any) -> str:
    return PAPER_TASK_NAMES.get(str(task_name), str(task_name).replace("_", "-"))


def _latex_header_for_grouping_col(col: str) -> str:
    return _latex_escape(col.replace("_", " ").title())


def _latex_group_values(df: pd.DataFrame, grouping: Sequence[str]) -> list[tuple[Any, ...]]:
    if not grouping:
        return [()]
    return list(
        df[list(grouping)]
        .drop_duplicates()
        .sort_values(list(grouping))
        .itertuples(index=False, name=None)
    )


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _prepare_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared categorical ordering and recover legacy fixed-effect fields."""
    if df.empty:
        return df

    df = df.copy()
    df["model_name"] = pd.Categorical(df["model_name"], categories=MODELS, ordered=True)
    df["task_name"] = pd.Categorical(df["task_name"], categories=TASKS, ordered=True)

    if "fixed_effect_type" in df.columns:
        df["fixed_effect_type"] = df["fixed_effect_type"].astype("object")

        if {"name", "feature_dimension"}.issubset(df.columns):
            parsed_from_name = df["name"].str.extract(
                r"-(?P<fixed_effect_type>[^-]+)_(?P<feature_dimension>\d+)D-"
            )
            missing_fixed_effect = df["fixed_effect_type"].isna()
            df.loc[missing_fixed_effect, "fixed_effect_type"] = parsed_from_name.loc[
                missing_fixed_effect,
                "fixed_effect_type",
            ]
            df.loc[missing_fixed_effect, "feature_dimension"] = parsed_from_name.loc[
                missing_fixed_effect,
                "feature_dimension",
            ].astype(float)

        df["fixed_effect_type"] = pd.Categorical(
            df["fixed_effect_type"],
            categories=FIXED_EFFECT_TYPES,
            ordered=True,
        )

    return _sort_results_frame(df)


def _sort_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return run-level results in a deterministic order."""
    sort_cols = [
        col
        for col in [
            "model_name",
            "fixed_effect_type",
            "feature_dimension",
            TASK_COL_NAME,
            SEED_COL_NAME,
            "validation_metric",
            "run_id",
            "name",
        ]
        if col in df.columns
    ]
    if not sort_cols:
        return df
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _is_synthetic(df: pd.DataFrame) -> bool:
    return "fixed_effect_type" in df.columns


def _grouping_levels(
    df: pd.DataFrame,
    grouping_cols: Sequence[str] | None = None,
) -> list[str]:
    """Resolve the aggregation columns for standard and custom reports."""
    if grouping_cols is not None:
        return list(grouping_cols)
    if _is_synthetic(df):
        return SYNTHETIC_GROUPING_COLS
    return REAL_GROUPING_COLS


def _select_best_validation_rmse_rows(
    model_results: pd.DataFrame,
    grouping_with_seed: Sequence[str],
) -> pd.DataFrame:
    """Select the lowest final-validation-RMSE run per grouping and seed."""
    validation_col = "validation_rmse"
    candidates = model_results.dropna(subset=[validation_col])
    if candidates.empty:
        return model_results.iloc[0:0].copy()

    grouping = list(grouping_with_seed)
    tie_breakers = [col for col in ("run_id", "name") if col in candidates.columns]
    sort_df = candidates.copy()
    sort_cols = grouping + [validation_col]
    ascending = [True] * len(sort_cols)

    for col in tie_breakers:
        sort_col = f"__tie_break_{col}"
        sort_df[sort_col] = sort_df[col].astype(str)
        sort_cols.append(sort_col)
        ascending.append(True)

    # Sorting by group/seed first and validation score second makes the first
    # duplicate within each group/seed the best validation-RMSE run.
    return (
        sort_df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        .drop_duplicates(grouping, keep="first")
        .drop(columns=[f"__tie_break_{col}" for col in tie_breakers])
        .copy()
    )


def _handle_best_hyperparameters(
    *,
    best_rows: pd.DataFrame,
    model_name: str,
    grouping: Sequence[str],
    best_hyperparams_save_dir: str | Path | None,
    show_best_hyperparams: bool,
) -> None:
    """Display and/or save selected model params plus fit metadata."""
    params = SELECTED_MODEL_PARAMS_PER_MODEL[model_name]

    if best_rows.empty:
        return

    df = best_rows.copy()
    for param in params:
        if param not in df.columns:
            df[param] = np.nan

    cols = list(grouping) + [
        SEED_COL_NAME,
        "validation_rmse",
        "run_id",
        "test_rmse",
        "test_crps",
    ] + params
    df = df.loc[:, cols].rename(columns={"validation_rmse": "best_validation_score"})
    sort_cols = list(grouping) + [SEED_COL_NAME]
    df = df.sort_values(sort_cols)

    display_cols = ["best_validation_score", "run_id", "test_rmse", "test_crps"] + params
    table = df.set_index(sort_cols)[display_cols]

    if best_hyperparams_save_dir is not None:
        save_dir = Path(best_hyperparams_save_dir)
        table.to_csv(save_dir / f"{model_name}_params.csv")

    if show_best_hyperparams:
        print(f"\n--- Best model parameters for {model_name} ---")
        display(table)


def _exclude_values(
    df: pd.DataFrame,
    exclude_values: Mapping[str, Sequence[Any]] | None,
) -> pd.DataFrame:
    if exclude_values is None or df.empty:
        return df

    filtered = df.copy()
    for column, values in exclude_values.items():
        if column not in filtered.columns:
            continue
        filtered = filtered[~filtered[column].isin(values)]
    return filtered


def select_best_seed_runs(
    run_results: pd.DataFrame,
    grouping_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select one validation-RMSE-best run per model, result group, and seed.

    Selection currently always uses final validation RMSE.
    """
    df = _prepare_results_frame(run_results)
    grouping = _grouping_levels(df, grouping_cols)
    selected = []

    for _, model_results in df.groupby("model_name", observed=True):
        if "validation_rmse" not in model_results.columns or model_results["validation_rmse"].dropna().empty:
            continue

        best_rows = _select_best_validation_rmse_rows(model_results, grouping + [SEED_COL_NAME])
        best_rows = best_rows.copy()
        selected.append(best_rows)

    if not selected:
        return df.iloc[0:0].copy()

    return pd.concat(selected, ignore_index=True)


def _normalize_metrics(metrics: Sequence[str]) -> list[str]:
    normalized = [metric.lower() for metric in metrics]
    invalid = [metric for metric in normalized if metric not in {"rmse", "crps"}]
    if invalid:
        raise ValueError(f"Unsupported metrics: {invalid}. Expected 'rmse' and/or 'crps'.")
    return normalized


def _ensure_standard_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure display formatters have standard-error columns available."""
    out = df.copy()
    for metric in ("rmse", "crps"):
        se_col = f"test_{metric}_se"
        std_col = f"test_{metric}_std"
        n_col = f"test_{metric}_n"
        # If standard error is missing but std and n are present, compute standard error.
        if se_col not in out.columns and {std_col, n_col}.issubset(out.columns):
            out[se_col] = out[std_col] / np.sqrt(out[n_col])
    return out


def _format_mean_se(mean: Any, se: Any, digits: int) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(se):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def _validate_digits(digits: int) -> int:
    if digits < 0:
        raise ValueError("digits must be non-negative.")
    return digits


def _validate_digits_spec(digits: DigitsSpec) -> DigitsSpec:
    if isinstance(digits, Mapping):
        return {str(label): _validate_digits(label_digits) for label, label_digits in digits.items()}
    return _validate_digits(digits)


def _digits_for_label(digits: DigitsSpec, label: Any) -> int:
    if not isinstance(digits, Mapping):
        return digits

    label_key = str(label)
    if label_key in digits:
        return digits[label_key]
    for configured_label, configured_digits in digits.items():
        if configured_label.lower() == label_key.lower():
            return configured_digits
    if "default" in digits:
        return digits["default"]
    if "*" in digits:
        return digits["*"]
    return 3
