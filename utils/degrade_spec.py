"""
degrade_spec.py

Defines the DegradationSpec dataclass and the preset degradations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DegradationSpec:
    """One page in the degradation stack."""

    page_index: int
    transform_name: str
    transform_level: int
    parameter_name: str
    parameter_value: float | int
    parameters: dict[str, Any]
    albumentations_function: str
    opencv_function: str


def build_degradation_specs() -> list[DegradationSpec]:
    """Return the immutable 36-page degradation catalog in page order."""
    specs: list[DegradationSpec] = []

    def add_family(
        transform_name: str,
        parameter_name: str,
        values: Sequence[float | int],
        parameters_for_value: Any,
        albumentations_function: str,
        opencv_function: str,
    ) -> None:
        for level, value in enumerate(values, start=1):
            specs.append(
                DegradationSpec(
                    page_index=len(specs),
                    transform_name=transform_name,
                    transform_level=level,
                    parameter_name=parameter_name,
                    parameter_value=value,
                    parameters=parameters_for_value(value),
                    albumentations_function=albumentations_function,
                    opencv_function=opencv_function,
                )
            )

    magnitude_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    add_family(
        "grid_distortion",
        "distort_limit",
        magnitude_values,
        lambda value: {
            "num_steps": 5,
            "distort_limit": [-value, value],
            "normalized": True,
            "interpolation": int(cv2.INTER_LINEAR),
        },
        "albumentations.GridDistortion",
        "cv2.remap",
    )

    noise_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    add_family(
        "gauss_noise",
        "std_range",
        noise_values,
        lambda value: {
            "std_range": [value, value],
            "mean_range": [0.0, 0.0],
            "per_channel": True,
            "noise_scale_factor": 1.0,
        },
        "albumentations.GaussNoise",
        "",
    )

    blur_iterations = [10, 20, 30, 40, 50, 60]
    blur_sigmas = [0.8 * np.sqrt(iterations) for iterations in blur_iterations]
    add_family(
        "gaussian_blur",
        "sigma_limit",
        blur_sigmas,
        lambda value: {"blur_limit": 0, "sigma_limit": [value, value]},
        "albumentations.GaussianBlur",
        "cv2.sepFilter2D",
    )

    gamma_values = np.geomspace(1.0, 3.0, 6) * 100.0
    add_family(
        "random_gamma",
        "gamma_limit",
        gamma_values,
        lambda value: {"gamma_limit": [value, value]},
        "albumentations.RandomGamma",
        "",
    )
    add_family(
        "dilate",
        "iterations",
        [1, 2, 3, 4, 5, 6],
        lambda value: {"kernel_size": 3, "iterations": value},
        "utils.custom_transforms.Dilate",
        "cv2.dilate",
    )
    add_family(
        "erode",
        "iterations",
        [1, 2, 3, 4, 5, 6],
        lambda value: {"kernel_size": 3, "iterations": value},
        "utils.custom_transforms.Erode",
        "cv2.erode",
    )
    return specs


def specs_to_frame(specs):
    """
    Convert a list of DegradationSpec objects to a pandas DataFrame for display.
    """
    records = []
    for spec in specs:
        record = asdict(spec)
        parameters = record.pop("parameters")
        record["parameters_json"] = json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
        )
        record["albumentations_version"] = A.__version__
        record["opencv_version"] = cv2.__version__
        records.append(record)

    return pd.DataFrame.from_records(records)


def _build_transform(spec: DegradationSpec) -> A.Compose:
    """
    Helper function to build an Albumentations transform from a DegradationSpec.
    Hard-coded behavior for each transform type to restrict parameter behavior.
    """

    parameters = spec.parameters
    if spec.transform_name == "grid_distortion":
        transform = A.GridDistortion(
            num_steps=int(parameters["num_steps"]),
            distort_limit=tuple(parameters["distort_limit"]),
            normalized=bool(parameters["normalized"]),
            interpolation=int(parameters["interpolation"]),
            p=1.0,
        )
    elif spec.transform_name == "gauss_noise":
        transform = A.GaussNoise(
            std_range=tuple(parameters["std_range"]),
            mean_range=tuple(parameters["mean_range"]),
            per_channel=bool(parameters["per_channel"]),
            noise_scale_factor=float(parameters["noise_scale_factor"]),
            p=1.0,
        )
    elif spec.transform_name == "gaussian_blur":
        transform = A.GaussianBlur(
            blur_limit=int(parameters["blur_limit"]),
            sigma_limit=tuple(parameters["sigma_limit"]),
            p=1.0,
        )
    elif spec.transform_name == "random_gamma":
        transform = A.RandomGamma(
            gamma_limit=tuple(parameters["gamma_limit"]),
            p=1.0,
        )
    elif spec.transform_name == "dilate":
        from .custom_transforms import Dilate

        transform = Dilate(
            kernel_size=int(parameters["kernel_size"]),
            iterations=int(parameters["iterations"]),
            p=1.0,
        )
    elif spec.transform_name == "erode":
        from .custom_transforms import Erode

        transform = Erode(
            kernel_size=int(parameters["kernel_size"]),
            iterations=int(parameters["iterations"]),
            p=1.0,
        )
    else:
        raise ValueError(f"Unsupported degradation transform: {spec.transform_name}")
    return A.Compose([transform])
