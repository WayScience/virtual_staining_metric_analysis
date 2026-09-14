from unittest.mock import patch

import numpy as np
import pandas as pd

from utils import nested_regression
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

    initial_rng_state = rng.bit_generator.state
    first = bootstrap_nested_regression(frame, colspec, config, rng=rng)
    state_after_first = rng.bit_generator.state
    second = bootstrap_nested_regression(frame, colspec, config, rng=rng)
    state_after_second = rng.bit_generator.state

    assert len(first) == 2
    assert len(second) == 2
    assert "boot_idx" in first.columns
    assert initial_rng_state != state_after_first
    assert state_after_first != state_after_second


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

    with patch.object(
        nested_regression,
        "_fit_ols_formula",
        wraps=nested_regression._fit_ols_formula,
    ) as fit_ols:
        result = bootstrap_nested_regression(frame, colspec, config)

    assert len(result) == 1
    assert fit_ols.call_count == 2
    assert [fit_call.args[2] for fit_call in fit_ols.call_args_list] == ["HC3", "HC3"]
    coefficient_cols = ["beta_x1_restricted", "beta_x1_full", "beta_x2"]
    assert np.isfinite(result.loc[0, coefficient_cols]).all()
