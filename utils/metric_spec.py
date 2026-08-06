from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

MetricCallable = Callable[..., torch.Tensor]
SampleAggregator = Callable[[torch.Tensor], torch.Tensor]


def mean_non_batch_dimensions(values: torch.Tensor) -> torch.Tensor:
    """Reduce all non-batch dimensions to one value per sample."""
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    return values.mean(dim=tuple(range(1, values.ndim)))


@dataclass(frozen=True)
class MetricSpec:
    name: str
    metric: MetricCallable | nn.Module
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    input_channels: int = 1
    page_batch_size: int = 36
    aggregate_samples: SampleAggregator = mean_non_batch_dimensions

    def __post_init__(self) -> None:
        if not self.name or any(character in self.name for character in "/\\"):
            raise ValueError(f"Invalid metric name: {self.name!r}.")
        if self.input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3.")
        if self.page_batch_size <= 0:
            raise ValueError("page_batch_size must be positive.")
