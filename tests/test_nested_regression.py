import numpy as np
import pandas as pd

from utils.nested_regression import (
    BootstrapConfig,
    ColumnSpec,
    bootstrap_nested_regression,
)


def test_bootstrap_nested_regression_accepts_ungrouped_frame_and_shared_rng():
    frame = pd.DataFrame(
        {
            "metric_value": np.arange(30, dtype=float),
            "parameter_value": np.arange(30, dtype=float),
            "seeding_density": np.tile([1.0, 2.0, 3.0], 10),
        }
    )
    colspec = ColumnSpec(
        group_cols=(),
        y="metric_value",
        x1="parameter_value",
        x2="seeding_density",
    )
    config = BootstrapConfig(n_boot=2, min_group_size=25, use_tqdm=False)
    rng = np.random.default_rng(7)

    first = bootstrap_nested_regression(frame, colspec, config, rng=rng)
    second = bootstrap_nested_regression(frame, colspec, config, rng=rng)

    assert len(first) == 2
    assert len(second) == 2
    assert "boot_idx" in first.columns
    assert not first.equals(second)


def test_bootstrap_nested_regression_extracts_coefficients_with_robust_covariance():
    frame = pd.DataFrame(
        {
            "metric_value": np.arange(30, dtype=float) + np.tile([0.0, 0.5, -0.25], 10),
            "parameter_value": np.arange(30, dtype=float),
            "seeding_density": np.tile([1.0, 2.0, 3.0], 10),
        }
    )
    colspec = ColumnSpec(
        group_cols=(),
        y="metric_value",
        x1="parameter_value",
        x2="seeding_density",
    )
    config = BootstrapConfig(
        n_boot=1,
        min_group_size=25,
        robust_cov="HC3",
        use_tqdm=False,
    )

    result = bootstrap_nested_regression(frame, colspec, config)

    assert len(result) == 1
    coefficient_cols = ["beta_x1_restricted", "beta_x1_full", "beta_x2"]
    assert np.isfinite(result.loc[0, coefficient_cols]).all()
