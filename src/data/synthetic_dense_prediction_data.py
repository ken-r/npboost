"""Helpers for dense synthetic prediction data used in visualization.

The utilities in this module build plot-oriented synthetic split frames with a
dense 1D query grid. They are mainly used to generate smooth model prediction
curves and intervals for figures, rather than to create ordinary training or
evaluation data.

For in-context plots, the dense target rows should belong to the same realized
task functions as the saved training observations. To do that, the current
implementation recovers each stored group's random-effect realization from the
raw data before evaluating it on the dense grid. That recovery is intentionally
GP-specific at the moment: it expects an ``RBFGPRandomEffect`` exposing the
provider kernel via ``_create_kernel()`` and ``alpha``. The few-shot helper
samples fresh plot-only tasks from the provider and is also restricted here to
1D plotting grids.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

from src.constants import (
    CONTEXT_ROLE_COLUMN_NAME,
    FIXED_EFFECT_PART_NAME,
    GROUPING_COLUMN_NAME,
    NOISE_PART_NAME,
    RANDOM_EFFECT_PART_NAME,
    RESPONSE_COLUMN_NAME,
    SPLIT_COLUMN_NAME,
    SUPPORT_ROLE_NAME,
    SYNTHETIC_FEATURE_COLUMN_PREFIX,
    TARGET_ROLE_NAME,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
)


PLOT_GENERATION_SEED = 10_000_001
IN_CONTEXT_DENSE_NOISE_SEED = 10_000_101
FEW_SHOT_RANDOM_EFFECT_SEED = 10_000_201
FEW_SHOT_NOISE_SEED = 10_000_202
FEW_SHOT_SUPPORT_SEED = 10_000_203
FEW_SHOT_IRRELEVANT_SEED = 10_000_204
GROUP_ID_OFFSET = 100000


def dense_1d_query_grid_avoiding(
    lower: float,
    upper: float,
    *,
    n_query: int,
    avoid_values: np.ndarray,
    atol: float = 1e-12,
) -> np.ndarray:
    """Return a dense 1D grid that does not coincide with saved locations."""
    avoid_values = np.asarray(avoid_values, dtype=float).reshape(-1)
    n_candidates = max(4 * n_query, n_query + 2 * len(avoid_values) + 10)
    candidates = np.linspace(lower, upper, n_candidates)
    if avoid_values.size:
        keep = ~np.isclose(candidates[:, None], avoid_values[None, :], rtol=0.0, atol=atol).any(axis=1)
        candidates = candidates[keep]
    if len(candidates) < n_query:
        raise ValueError("Could not create enough dense query points after removing saved locations.")

    selected = candidates[np.linspace(0, len(candidates) - 1, n_query, dtype=int)]
    if avoid_values.size:
        assert not np.isclose(selected[:, None], avoid_values[None, :], rtol=0.0, atol=atol).any()
    return selected[:, None]


def relevant_feature_columns(provider) -> list[str]:
    return [f"{SYNTHETIC_FEATURE_COLUMN_PREFIX}{i}" for i in range(provider.fixed_effect.feature_dimension)]


def fit_stored_realization_gp(
    provider,
    conditioning_features: np.ndarray,
    conditioning_values: np.ndarray,
    *,
    jitter_factors=(1.0, 10.0, 100.0, 1e3, 1e4),
):
    """Condition the provider's GP kernel on saved group random-effect values."""
    random_effect = provider.random_effect
    if type(random_effect).__name__ != "RBFGPRandomEffect":
        raise TypeError(
            "Dense recovery currently expects RBFGPRandomEffect. "
            f"Got {type(random_effect).__name__}."
        )
    if not hasattr(random_effect, "_create_kernel") or not hasattr(random_effect, "alpha"):
        raise AttributeError("The random effect must expose _create_kernel() and alpha.")

    x_obs = np.asarray(conditioning_features, dtype=float)
    y_obs = np.asarray(conditioning_values, dtype=float).reshape(-1)
    last_error = None
    for factor in jitter_factors:
        effective_alpha = float(random_effect.alpha) * factor
        gpr = GaussianProcessRegressor(
            kernel=random_effect._create_kernel(),
            alpha=effective_alpha,
            optimizer=None,
            normalize_y=False,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpr.fit(x_obs, y_obs)
            fitted_at_obs = gpr.predict(x_obs)
            max_abs_deviation = float(np.max(np.abs(fitted_at_obs - y_obs)))
            return gpr, effective_alpha, max_abs_deviation
        except (np.linalg.LinAlgError, ValueError) as error:
            last_error = error
    raise RuntimeError(
        f"Could not condition GP up to alpha={float(random_effect.alpha) * jitter_factors[-1]:.2e}."
    ) from last_error


def recover_stored_group_random_effect(
    provider,
    stored_group_df: pd.DataFrame,
    target_features: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    gpr, effective_alpha, max_abs_deviation = fit_stored_realization_gp(
        provider,
        stored_group_df[feature_cols].to_numpy(dtype=float),
        stored_group_df[RANDOM_EFFECT_PART_NAME].to_numpy(dtype=float),
    )
    recovered = gpr.predict(np.asarray(target_features, dtype=float)).reshape(-1)
    return recovered, {
        "effective_alpha": effective_alpha,
        "max_abs_deviation": max_abs_deviation,
    }


def dense_in_context_plot_split_df(
    provider,
    base_split_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    groups: list[int],
    seed: int,
    noise_seed: int,
    n_query: int,
    feature: str,
    split_name: str = TEST_SPLIT_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover saved in-context tasks and extend them with dense noisy target rows."""
    lower, upper = map(float, provider.features_domain)
    feature_cols = relevant_feature_columns(provider)
    selected_df = base_split_df[base_split_df[GROUPING_COLUMN_NAME].isin(groups)].copy()
    context_df = selected_df[selected_df[SPLIT_COLUMN_NAME] == TRAIN_SPLIT_NAME].copy()
    context_df["true_signal"] = context_df[FIXED_EFFECT_PART_NAME] + context_df[RANDOM_EFFECT_PART_NAME]
    context_df["plot_generation_seed"] = seed
    context_df["plot_only"] = False

    irrelevant_rng = np.random.default_rng(seed + 1)
    irrelevant_columns = [column for column in base_split_df.columns if column.startswith("irrelevant_feature_")]

    target_frames = []
    recovery_rows = []
    for group_index, group_id in enumerate(groups):
        group_saved = raw_df[raw_df[GROUPING_COLUMN_NAME] == group_id].sort_values(feature)
        if group_saved.empty:
            raise ValueError(f"Group {group_id} is missing from the raw data.")
        saved_x = group_saved[feature].to_numpy(dtype=float)
        x_target = dense_1d_query_grid_avoiding(
            lower,
            upper,
            n_query=n_query,
            avoid_values=saved_x,
        )

        fixed = np.asarray(provider.fixed_effect.generate(x_target), dtype=float).reshape(-1)
        random, diagnostics = recover_stored_group_random_effect(
            provider,
            group_saved,
            x_target,
            feature_cols,
        )
        noise = np.asarray(provider.noise.generate(x_target, seed=noise_seed + group_index), dtype=float).reshape(-1)
        true_signal = fixed + random
        response = true_signal + noise

        target_frame = pd.DataFrame(
            {
                GROUPING_COLUMN_NAME: group_id,
                RESPONSE_COLUMN_NAME: response,
                FIXED_EFFECT_PART_NAME: fixed,
                RANDOM_EFFECT_PART_NAME: random,
                NOISE_PART_NAME: noise,
                feature: x_target[:, 0],
                SPLIT_COLUMN_NAME: split_name,
                "true_signal": true_signal,
                "plot_generation_seed": seed,
                "noise_generation_seed": noise_seed + group_index,
                "plot_only": True,
            }
        )
        for column in irrelevant_columns:
            target_frame[column] = irrelevant_rng.uniform(lower, upper, size=len(target_frame))
        target_frames.append(target_frame)
        recovery_rows.append(
            {
                GROUPING_COLUMN_NAME: group_id,
                "effective_alpha": diagnostics["effective_alpha"],
                "max_abs_deviation": diagnostics["max_abs_deviation"],
                "random_effect_scale": float(group_saved[RANDOM_EFFECT_PART_NAME].std()),
            }
        )

    dense_df = pd.concat([context_df, *target_frames], ignore_index=True, sort=False)
    recovery_df = pd.DataFrame(recovery_rows)
    return dense_df, recovery_df


def dense_few_shot_plot_split_df(
    provider,
    *,
    random_effect_seed: int,
    noise_seed: int,
    support_seed: int,
    irrelevant_seed: int,
    n_tasks: int,
    n_support: int,
    n_query: int,
    group_id_offset: int,
    split_name: str = TEST_SPLIT_NAME,
) -> pd.DataFrame:
    """Create fresh dense few-shot tasks and randomly label support rows."""
    fixed_effect = provider.fixed_effect
    if fixed_effect.feature_dimension != 1:
        raise ValueError("This helper intentionally supports only 1D plotting grids.")

    lower, upper = map(float, provider.features_domain)
    support_rng = np.random.default_rng(support_seed)
    irrelevant_rng = np.random.default_rng(irrelevant_seed)

    features = []
    groups = []
    roles = []
    n_total = n_support + n_query
    for task_index in range(n_tasks):
        group_id = group_id_offset + task_index
        x = np.linspace(lower, upper, n_total)
        support_indices = set(support_rng.choice(n_total, size=n_support, replace=False).tolist())
        group_roles = [SUPPORT_ROLE_NAME if i in support_indices else TARGET_ROLE_NAME for i in range(n_total)]

        features.append(x[:, None])
        groups.extend([group_id] * n_total)
        roles.extend(group_roles)

    relevant_features = np.vstack(features)
    groups = np.asarray(groups)
    roles = np.asarray(roles)

    fixed = np.asarray(fixed_effect.generate(relevant_features), dtype=float).reshape(-1)
    random = np.asarray(
        provider.random_effect.generate(
            relevant_features,
            groups,
            seed=random_effect_seed,
        ),
        dtype=float,
    ).reshape(-1)
    noise = np.asarray(provider.noise.generate(relevant_features, seed=noise_seed), dtype=float).reshape(-1)
    true_signal = fixed + random
    response = true_signal + noise

    df = pd.DataFrame(
        {
            GROUPING_COLUMN_NAME: groups,
            RESPONSE_COLUMN_NAME: response,
            FIXED_EFFECT_PART_NAME: fixed,
            RANDOM_EFFECT_PART_NAME: random,
            NOISE_PART_NAME: noise,
            f"{SYNTHETIC_FEATURE_COLUMN_PREFIX}0": relevant_features[:, 0],
            SPLIT_COLUMN_NAME: split_name,
            CONTEXT_ROLE_COLUMN_NAME: roles,
            "true_signal": true_signal,
            "plot_generation_seed": PLOT_GENERATION_SEED,
            "random_effect_generation_seed": random_effect_seed,
            "noise_generation_seed": noise_seed,
            "support_generation_seed": support_seed,
            "plot_only": True,
        }
    )

    for i in range(getattr(provider, "n_irrelevant_features", 0)):
        df[f"irrelevant_feature_{i}"] = irrelevant_rng.uniform(lower, upper, size=len(df))

    return df
