"""
Utilities for iterative metric absolute response analysis.
Contains a preset for smaller effect degradations meant to be applied repeatedly,
    plus orchestrators for the actual iterative application.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from torch.nn.functional import l1_loss
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from .custom_metrics import ForegroundPSNR, ForegroundSSIM, ReusableDISTS, ReusableLPIPS
from .metric_spec import MetricSpec


@dataclass(frozen=True)
class AbsoluteResponseConfig:
    """Parameters controlling the cumulative degradation experiment."""

    iterations: int = 500
    snapshot_interval: int = 10
    noise_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive.")
        if self.noise_fraction < 0.0:
            raise ValueError("noise_fraction must be non-negative.")


def build_absolute_response_metric_specs() -> dict[str, MetricSpec]:
    """Build the metrics used to measure response to cumulative degradation."""
    return {
        "ssim": MetricSpec(
            name="ssim",
            metric=structural_similarity_index_measure,
            kwargs={"data_range": 1.0, "reduction": "none"},
        ),
        "psnr": MetricSpec(
            name="psnr",
            metric=peak_signal_noise_ratio,
            kwargs={"data_range": 1.0, "reduction": "none", "dim": (1, 2, 3)},
        ),
        "mae": MetricSpec(
            name="mae",
            metric=l1_loss,
            kwargs={"reduction": "none"},
        ),
        "lpips": MetricSpec(
            name="lpips",
            metric=ReusableLPIPS(normalize=True),
            kwargs={"reduction": "none"},
            input_channels=3,
        ),
        "dists": MetricSpec(
            name="dists",
            metric=ReusableDISTS(),
            kwargs={"reduction": "none"},
            input_channels=3,
        ),
        "foreground_ssim": MetricSpec(
            name="foreground_ssim",
            metric=ForegroundSSIM(data_range=(0.0, 1.0)),
            kwargs={"reduction": "none"},
        ),
        "foreground_psnr": MetricSpec(
            name="foreground_psnr",
            metric=ForegroundPSNR(data_range=(0.0, 1.0)),
            kwargs={"reduction": "none"},
        ),
    }


def validate_reference_image(image: np.ndarray) -> np.ndarray:
    """Return a finite 2D float32 image with intensities in ``[0, 1]``."""
    reference = np.asarray(image, dtype=np.float32).squeeze().copy()
    if reference.ndim != 2:
        raise ValueError(f"Expected a 2D image, received shape {reference.shape}.")
    if not np.isfinite(reference).all():
        raise ValueError("The reference image contains non-finite values.")
    if reference.min() < 0.0 or reference.max() > 1.0:
        raise ValueError(
            "Metrics and degradations expect image intensities in [0, 1]; "
            f"received [{reference.min()}, {reference.max()}]."
        )
    return reference


def build_iterative_degradations(
    reference_image: np.ndarray,
    config: AbsoluteResponseConfig,
) -> tuple[dict[str, A.Compose], dict[str, float | int | str]]:
    """Build seeded single-step transforms and their serializable configuration."""
    reference = validate_reference_image(reference_image)
    reference_contrast = float(reference.max() - reference.min())
    noise_std = config.noise_fraction * reference_contrast
    morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def erode_once(image: np.ndarray, **_: object) -> np.ndarray:
        return cv2.erode(image, morphology_kernel, iterations=1)

    def dilate_once(image: np.ndarray, **_: object) -> np.ndarray:
        return cv2.dilate(image, morphology_kernel, iterations=1)

    transforms = {
        "gauss_noise": A.Compose(
            [
                A.GaussNoise(
                    std_range=(noise_std, noise_std),
                    mean_range=(0.0, 0.0),
                    per_channel=True,
                    p=1.0,
                )
            ],
            seed=42,
        ),
        "gaussian_blur": A.Compose(
            [A.GaussianBlur(blur_limit=0, sigma_limit=(0.8, 0.8), p=1.0)],
            seed=43,
        ),
        "grid_distortion": A.Compose(
            [
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=(-0.01, 0.01),
                    normalized=True,
                    p=1.0,
                )
            ],
            seed=44,
        ),
        "random_gamma": A.Compose(
            [A.RandomGamma(gamma_limit=(102.5, 102.5), p=1.0)],
            seed=45,
        ),
        "erode": A.Compose([A.Lambda(image=erode_once, p=1.0)]),
        "dilate": A.Compose([A.Lambda(image=dilate_once, p=1.0)]),
    }
    settings: dict[str, float | int | str] = {
        **asdict(config),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "reference_contrast": reference_contrast,
        "gaussian_noise_std": noise_std,
        "morphology_kernel": "3x3 ellipse",
        "morphology_iterations_per_step": 1,
    }
    return transforms, settings


class MetricEvaluator:
    """Evaluate a collection of metric specifications on one image pair."""

    def __init__(
        self,
        metric_specs: Mapping[str, MetricSpec],
        device: torch.device | str | None = None,
    ) -> None:
        self.metric_specs = dict(metric_specs)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        for spec in self.metric_specs.values():
            if isinstance(spec.metric, torch.nn.Module):
                spec.metric.to(device=self.device, dtype=torch.float32)
                spec.metric.eval()

    def _to_tensor(self, image: np.ndarray, input_channels: int) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(
            device=self.device,
            dtype=torch.float32,
        )
        if input_channels > 1:
            tensor = tensor.expand(-1, input_channels, -1, -1)
        return tensor

    @torch.inference_mode()
    def __call__(
        self,
        degraded_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> dict[str, float]:
        scores = {}
        for metric_name, spec in self.metric_specs.items():
            degraded_tensor = self._to_tensor(degraded_image, spec.input_channels)
            reference_tensor = self._to_tensor(reference_image, spec.input_channels)
            values = spec.metric(degraded_tensor, reference_tensor, **dict(spec.kwargs))
            values = spec.aggregate_samples(values)
            if values.numel() != 1:
                raise ValueError(
                    f"Metric {metric_name!r} returned {values.numel()} values for one image."
                )
            scores[metric_name] = float(values.item())
        return scores


def run_iterative_degradation_analysis(
    reference_image: np.ndarray,
    reference_channel: str,
    output_dir: str | Path,
    *,
    config: AbsoluteResponseConfig | None = None,
    metric_specs: Mapping[str, MetricSpec] | None = None,
    device: torch.device | str | None = None,
) -> pd.DataFrame:
    """Run cumulative degradations and persist metrics, snapshots, and configuration."""
    config = config or AbsoluteResponseConfig()
    reference = validate_reference_image(reference_image)
    transforms, settings = build_iterative_degradations(reference, config)
    evaluator = MetricEvaluator(
        metric_specs or build_absolute_response_metric_specs(),
        device=device,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings["reference_channel"] = reference_channel
    with (output_dir / "configuration.json").open("w") as config_file:
        json.dump(settings, config_file, indent=2)
    tiff.imwrite(output_dir / "reference_iteration_000.tiff", reference)

    records = []
    for transform_name, transform in transforms.items():
        degraded_image = reference.copy()
        transform_output_dir = output_dir / transform_name
        transform_output_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(1, config.iterations + 1):
            degraded_image = np.asarray(
                transform(image=degraded_image)["image"],
                dtype=np.float32,
            )
            degraded_image = np.clip(degraded_image, 0.0, 1.0)
            scores = evaluator(degraded_image, reference)
            records.extend(
                {
                    "reference_channel": reference_channel,
                    "transform_name": transform_name,
                    "iteration": iteration,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                }
                for metric_name, metric_value in scores.items()
            )

            if iteration % config.snapshot_interval == 0:
                tiff.imwrite(
                    transform_output_dir / f"iteration_{iteration:03d}.tiff",
                    degraded_image,
                )

        print(f"Completed {transform_name}: {config.iterations} iterations")

    results = pd.DataFrame.from_records(records)
    results.to_parquet(output_dir / "metric_results.parquet", index=False, compression="zstd")
    return results
