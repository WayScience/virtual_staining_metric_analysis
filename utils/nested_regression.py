"""
Nested regression utilities for bootstrapping and summarizing effect sizes
    of one restricted term and one additional full term at a time with linear
    regression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap nested regression."""

    n_boot: int = 300  # how many bootstrap repeats
    sample_frac: float = 0.5  # fraction of group rows to sample for each bootstrap
    replace: bool = True  # whether to sample with replacement, usually True for bootstrap
    standardize: bool = True  # whether to standardize x1 and x2 columns before fitting
    random_state: int | None = 42
    use_tqdm: bool = True
    drop_na: bool = True
    min_group_size: int = 25  # minimum number of rows in a group to perform bootstrap
    max_per_group: int | None = None
    robust_cov: str | None = (
        None  # covariance type for robust standard errors, e.g., "HC3" or "HC0"
    )


@dataclass
class ColumnSpec:
    """Column names and modeling options for a 2-step nested regression."""

    group_cols: tuple[str, ...]
    y: str
    x1: str
    x2: str
    x2_categorical: bool = False
    x2_ordinal: bool = False
    ordinal_order: tuple[Any, ...] | None = None
    standardize_cols: tuple[str, ...] | None = None


def _std_cols(colspec: ColumnSpec) -> tuple[str, ...]:
    if colspec.standardize_cols is not None:
        return tuple(colspec.standardize_cols)
    if colspec.x2_categorical:
        return (colspec.x1,)
    return (colspec.x1, colspec.x2)


def _x2_term(colspec: ColumnSpec) -> str:
    return f"C({colspec.x2})" if colspec.x2_categorical else colspec.x2


def _fit_ols_formula(
    df: pd.DataFrame,
    formula: str,
    robust_cov: str | None = None,
) -> smf.RegressionResultsWrapper:
    """
    Helper function to fit an OLS model using a formula
        and optionally compute robust covariance estimates.

    :param df: DataFrame containing the data for regression.
    :param formula: A string representing the regression formula.
    :param robust_cov: Optional string specifying the type of robust covariance
        to compute (e.g., "HC3", "HC0"). If None, standard covariance is used.
    :return: A fitted OLS regression results object.
    """
    model = smf.ols(formula, data=df)
    res = model.fit()
    if robust_cov:
        res = res.get_robustcov_results(cov_type=robust_cov)
    return res


def _compute_effect_sizes(res_re, res_fu) -> dict[str, float]:
    """
    Compute effect sizes and R-squared statistics from restricted and
        full regression results.
    """

    # R-squared values
    r2_re = float(getattr(res_re, "rsquared", np.nan))
    r2_fu = float(getattr(res_fu, "rsquared", np.nan))

    # sum of squares
    ssr_re = float(getattr(res_re, "ssr", np.nan))
    ssr_fu = float(getattr(res_fu, "ssr", np.nan))

    # delta R-squared, measures the increase in explained variance when
    # adding x2 to the model
    delta_r2 = r2_fu - r2_re if np.isfinite(r2_re) and np.isfinite(r2_fu) else np.nan

    if np.isfinite(r2_re) and np.isfinite(r2_fu) and (1 - r2_re) > 0:
        partial_r2_x2 = (r2_fu - r2_re) / (1 - r2_re)
    elif np.isfinite(ssr_re) and np.isfinite(ssr_fu) and ssr_re > 0:
        partial_r2_x2 = (ssr_re - ssr_fu) / ssr_re
    else:
        partial_r2_x2 = np.nan

    if np.isfinite(r2_fu) and (1 - r2_fu) > 0:
        f2_x2 = (r2_fu - r2_re) / (1 - r2_fu)
    else:
        f2_x2 = np.nan

    return {
        "r2_restricted": r2_re,
        "r2_full": r2_fu,
        "delta_r2": delta_r2,
        "partial_r2_x2": partial_r2_x2,
        "cohen_f2_x2": f2_x2,
    }


def _prepare_bootstrap_frame(
    boot: pd.DataFrame,
    colspec: ColumnSpec,
) -> pd.DataFrame:
    """
    Prepare the bootstrap DataFrame for regression by converting x2 to ordinal
    if specified in the ColumnSpec.
    This function modifies the DataFrame in place and returns it.
    If x2 is not ordinal or is numeric, the DataFrame is returned unchanged.
    If x2 is non-numeric and ordinal_order is not provided, a ValueError is raised.
    """
    if not colspec.x2_ordinal:
        return boot

    if pd.api.types.is_numeric_dtype(boot[colspec.x2]):
        return boot

    if colspec.ordinal_order is None:
        raise ValueError(
            f"x2 '{colspec.x2}' is non-numeric and x2_ordinal=True, "
            "but no ordinal_order was provided."
        )

    order_map = {value: idx for idx, value in enumerate(colspec.ordinal_order)}

    boot = boot.copy()
    boot[colspec.x2] = boot[colspec.x2].map(order_map)
    return boot


def _one_bootstrap(
    df_group: pd.DataFrame,
    cfg: BootstrapConfig,
    colspec: ColumnSpec,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Perform one bootstrap iteration for a single group of data, fitting both
    the restricted and full regression models, and computing effect sizes.

    :param df_group: DataFrame containing the data for a single group.
    :param cfg: BootstrapConfig object with configuration parameters.
    :param colspec: ColumnSpec object specifying the columns for regression.
    :param rng: NumPy random number generator for reproducibility.
    :return: A dictionary containing regression coefficients, effect sizes, and
        the number of rows used in the bootstrap sample.
    """
    n_group = len(df_group)
    bsize = max(2, round(cfg.sample_frac * n_group)) if cfg.sample_frac else n_group
    idx = rng.choice(df_group.index.to_numpy(), size=bsize, replace=cfg.replace)
    boot = df_group.loc[idx].copy()

    boot = _prepare_bootstrap_frame(boot, colspec)

    if cfg.standardize:
        cols = list(_std_cols(colspec))
        scaler = StandardScaler()
        boot.loc[:, cols] = scaler.fit_transform(boot.loc[:, cols])

    y, x1, x2 = colspec.y, colspec.x1, colspec.x2
    x2_term = _x2_term(colspec)
    formula_re = f"{y} ~ {x1}"
    formula_fu = f"{y} ~ {x1} + {x2_term}"

    res_re = _fit_ols_formula(boot, formula_re, cfg.robust_cov)
    res_fu = _fit_ols_formula(boot, formula_fu, cfg.robust_cov)

    beta_x1_re = res_re.params.get(x1, np.nan)
    beta_x1_fu = res_fu.params.get(x1, np.nan)
    beta_x2 = np.nan if colspec.x2_categorical else res_fu.params.get(x2, np.nan)
    effects = _compute_effect_sizes(res_re, res_fu)

    return {
        "beta_x1_restricted": beta_x1_re,
        "beta_x1_full": beta_x1_fu,
        "beta_x2": beta_x2,
        **effects,
        "n_boot_rows": bsize,
        "n_group_rows": n_group,
    }


def bootstrap_nested_regression(
    df: pd.DataFrame,
    colspec: ColumnSpec,
    cfg: BootstrapConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Run grouped bootstrap nested regression for one x1 + one x2 specification.

    :param df: Input DataFrame containing the data.
    :param colspec: ColumnSpec object specifying the columns for regression.
    :param cfg: BootstrapConfig object with configuration parameters.
    :param rng: NumPy random number generator for reproducibility.
    :return: DataFrame containing bootstrap results for all groups and iterations.
    """
    if cfg is None:
        cfg = BootstrapConfig()

    if colspec.x2_categorical and colspec.x2_ordinal:
        raise ValueError("x2 cannot be both categorical and ordinal.")

    required_cols = {
        *colspec.group_cols,
        colspec.y,
        colspec.x1,
        colspec.x2,
    }
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Input data missing required columns: {sorted(missing_cols)}")

    work = df.loc[:, list(required_cols)].copy()

    if cfg.drop_na:
        work = work.dropna(subset=list(required_cols)).copy()

    if rng is None:
        rng = np.random.default_rng(cfg.random_state)

    if colspec.group_cols:
        grouped = work.groupby(list(colspec.group_cols), sort=False)
    else:
        grouped = [((), work)]

    if cfg.use_tqdm:
        try:
            from tqdm.auto import tqdm

            grouped_iter = tqdm(grouped, desc="Bootstrap groups")
        except ImportError:
            grouped_iter = grouped
    else:
        grouped_iter = grouped

    records: list[dict[str, Any]] = []

    for group_key, gdf in grouped_iter:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        n_group = len(gdf)
        if n_group < cfg.min_group_size:
            continue

        if cfg.max_per_group is not None and n_group > cfg.max_per_group:
            keep_idx = rng.choice(
                gdf.index.to_numpy(),
                size=cfg.max_per_group,
                replace=False,
            )
            gdf = gdf.loc[keep_idx]

        for b in range(cfg.n_boot):
            try:
                res = _one_bootstrap(gdf, cfg, colspec, rng)
            except (np.linalg.LinAlgError, ValueError):
                # Keep the bootstrap run resilient to occasional singular fits.
                logger.debug(
                    "Skipping failed bootstrap fit for group %s at iteration %d.",
                    group_key,
                    b,
                    exc_info=True,
                )
                continue

            row = {col: val for col, val in zip(colspec.group_cols, group_key)}
            row["boot_idx"] = b
            row.update(res)
            records.append(row)

    return pd.DataFrame.from_records(records)


def summarize_r2_scatter_bootstrap(
    boot_df: pd.DataFrame,
    output_file: str | Path | None = None,
    group_cols: tuple[str, ...] = ("metric_name", "transform_name"),
    restricted_col: str = "r2_restricted",
    partial_col: str = "partial_r2_x2",
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Summarize bootstrap results for restricted and partial R-squared values.
    Takes the concatenated bootstrap DataFrame of multiple single-group
        bootstrap results and computes the mean and (empirical) confidence
        intervals for restricted and partial R-squared values for each group.

    :param boot_df: DataFrame containing bootstrap results with required columns.
    :param output_file: Optional path to save the summary DataFrame as a Parquet file.
    :param group_cols: Tuple of column names to group by for summarization.
    :param restricted_col: Column name for restricted R-squared values.
    :param partial_col: Column name for partial R-squared values.
    :param ci: Confidence interval level (between 0 and 1) for quantile calculations.
    :return: DataFrame containing summarized R-squared statistics for each group.
    """

    required = set(group_cols) | {"boot_idx", restricted_col, partial_col}
    missing = sorted(required - set(boot_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q

    df = boot_df.copy()
    df[restricted_col] = pd.to_numeric(df[restricted_col], errors="coerce")
    df[partial_col] = pd.to_numeric(df[partial_col], errors="coerce")

    summary = (
        df.groupby(list(group_cols), dropna=False)
        .agg(
            n_boot=("boot_idx", "nunique"),
            restricted_r2_mean=(restricted_col, "mean"),
            restricted_r2_lower=(restricted_col, lambda x: x.quantile(lower_q)),
            restricted_r2_upper=(restricted_col, lambda x: x.quantile(upper_q)),
            partial_r2_mean=(partial_col, "mean"),
            partial_r2_lower=(partial_col, lambda x: x.quantile(lower_q)),
            partial_r2_upper=(partial_col, lambda x: x.quantile(upper_q)),
        )
        .reset_index()
        .sort_values(list(group_cols))
        .reset_index(drop=True)
    )

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(output_file, index=False)

    return summary
