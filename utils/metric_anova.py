from collections.abc import Mapping
from itertools import chain
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

VariableKind = Literal["continuous", "categorical"]


class VariableSpec(TypedDict):
    kind: VariableKind


class AnovaSpec(TypedDict):
    dependent: str
    variables: dict[str, VariableSpec]
    terms: tuple[tuple[str, ...], ...]
    anova_type: Literal[1, 2, 3]


def _formula_variable(
    name: str,
    spec: VariableSpec,
) -> str:
    """Render one variable for a Patsy formula."""
    quoted = f"Q({name!r})"
    return f"C({quoted})" if spec["kind"] == "categorical" else quoted


def _estimable_terms(
    frame: pd.DataFrame,
    spec: AnovaSpec,
) -> tuple[tuple[str, ...], ...]:
    """Return configured terms whose variables are present and nonconstant."""
    variables = spec["variables"]

    estimable = {
        name
        for name in variables
        if (name in frame.columns and frame[name].nunique(dropna=True) > 1)
    }

    return tuple(term for term in spec["terms"] if all(name in estimable for name in term))


def fit_anova(
    frame: pd.DataFrame,
    spec: AnovaSpec,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame | None:
    """
    Fit a configured ANOVA model after complete-case filtering.

    :param frame: Data frame containing the dependent variable and predictors.
    :param spec: Specification of the ANOVA model.
    :param metadata: Optional metadata to include in the result frame.
    :return: ANOVA table with eta2 and partial_eta2, or None if the model could not be fit.
    """

    dependent = spec["dependent"]
    variables = spec["variables"]

    if dependent not in frame.columns:
        raise ValueError(f"Dependent variable {dependent!r} is absent.")

    # Validate the specification itself.
    unknown = set(chain.from_iterable(spec["terms"])) - variables.keys()
    if unknown:
        raise ValueError(f"ANOVA terms contain undefined variables: {sorted(unknown)}")

    available_variables = [name for name in variables if name in frame.columns]

    data = frame.loc[
        :,
        [dependent, *available_variables],
    ].copy()

    # OLS response and continuous predictors must be numeric.
    numeric_columns = [
        dependent,
        *[name for name in available_variables if variables[name]["kind"] == "continuous"],
    ]

    for column in numeric_columns:
        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )
        data[column] = values.where(np.isfinite(values))

    # Response is always required.
    data = data.dropna(subset=[dependent])

    # First pass determines which predictors can contribute terms.
    terms = _estimable_terms(data, spec)

    required_variables = sorted(set(chain.from_iterable(terms)))

    # Complete-case filtering only uses variables actually needed by
    # currently estimable terms.
    data = data.dropna(subset=required_variables).copy()

    # Filtering may collapse a categorical factor or continuous predictor
    # to a single observed value.
    terms = _estimable_terms(data, spec)

    if len(data) < 2 or data[dependent].nunique(dropna=True) < 2:
        return None

    metadata = dict(metadata or {})

    if not terms:
        residual_ss = np.square(data[dependent] - data[dependent].mean()).sum()

        return pd.DataFrame(
            [
                {
                    "term": "Residual",
                    "sum_sq": residual_ss,
                    "df": len(data) - 1,
                    "F": np.nan,
                    "PR(>F)": np.nan,
                    "eta2": 1.0,
                    "partial_eta2": np.nan,
                    "n_obs": len(data),
                    **metadata,
                    "formula": f"Q({dependent!r}) ~ 1",
                }
            ]
        )

    formula_terms = {
        ":".join(_formula_variable(name, variables[name]) for name in term): ":".join(term)
        for term in terms
    }

    formula = f"Q({dependent!r}) ~ " + " + ".join(formula_terms)

    try:
        model = smf.ols(
            formula,
            data=data,
        ).fit()

        table = anova_lm(
            model,
            typ=spec["anova_type"],
        )
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        print(
            "Skipping ANOVA"
            + (f" ({', '.join(f'{k}={v}' for k, v in metadata.items())})" if metadata else "")
            + f": {exc}"
        )
        return None

    if "Residual" not in table.index:
        return None

    # Corrected total sum squares of the dependent variable.
    total_ss = np.square(data[dependent] - data[dependent].mean()).sum()

    if not np.isfinite(total_ss) or total_ss <= 0:
        return None

    residual_ss = table.loc["Residual", "sum_sq"]

    # compute eta2 for each term, which is the proportion of total variance
    # explained by that term
    table["eta2"] = table["sum_sq"] / total_ss

    denominator = table["sum_sq"] + residual_ss
    table["partial_eta2"] = np.where(
        table.index == "Residual",
        np.nan,
        np.divide(
            table["sum_sq"],
            denominator,
            out=np.full(len(table), np.nan),
            where=denominator.to_numpy() > 0,
        ),
    )

    return (
        table.rename(index=formula_terms)
        .reset_index(names="term")
        .assign(
            n_obs=len(data),
            **metadata,
            formula=formula,
        )
    )
