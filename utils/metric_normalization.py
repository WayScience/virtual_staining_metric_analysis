"""Utilities for normalizing image-quality metric results."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

DOCUMENTED_METRIC_BOUNDS = {
    "ssim": (0.0, 1.0),
    "psnr": (0.0, 50.0),
    "mae": (0.0, 1.0),
    "lpips": (0.0, 1.0),
    "dists": (0.0, 1.0),
    "foreground_ssim": (0.0, 1.0),
    "foreground_psnr": (0.0, 50.0),
}

HIGHER_IS_BETTER = {
    "ssim": True,
    "psnr": True,
    "mae": False,
    "lpips": False,
    "dists": False,
    "foreground_ssim": True,
    "foreground_psnr": True,
}


def normalize_metric_results(
    results: pd.DataFrame,
    *,
    bounds: Mapping[str, tuple[float, float]] = DOCUMENTED_METRIC_BOUNDS,
    higher_is_better: Mapping[str, bool] = HIGHER_IS_BETTER,
) -> pd.DataFrame:
    """Clip metrics to documented bounds and orient normalized scores toward quality."""
    metric_names = set(results["metric_name"])
    if metric_names != set(bounds):
        raise ValueError("Documented bounds must define every result metric exactly once.")
    if metric_names != set(higher_is_better):
        raise ValueError("Metric directions must define every result metric exactly once.")

    normalized = results.copy()
    normalized["lower_bound"] = normalized["metric_name"].map(
        {name: limits[0] for name, limits in bounds.items()}
    )
    normalized["upper_bound"] = normalized["metric_name"].map(
        {name: limits[1] for name, limits in bounds.items()}
    )
    bound_width = normalized["upper_bound"] - normalized["lower_bound"]
    if (bound_width <= 0).any():
        raise ValueError("Every documented upper bound must exceed its lower bound.")

    normalized["clipped_metric_value"] = normalized["metric_value"].clip(
        lower=normalized["lower_bound"],
        upper=normalized["upper_bound"],
    )
    normalized["normalized_metric_value"] = (
        normalized["clipped_metric_value"] - normalized["lower_bound"]
    ) / bound_width
    lower_is_better = ~normalized["metric_name"].map(higher_is_better)
    normalized.loc[lower_is_better, "normalized_metric_value"] = (
        1.0 - normalized.loc[lower_is_better, "normalized_metric_value"]
    )

    values = normalized["normalized_metric_value"]
    if not np.isfinite(values).all() or not values.between(0.0, 1.0).all():
        raise ValueError("Normalized metric values must be finite and within [0, 1].")
    return normalized
