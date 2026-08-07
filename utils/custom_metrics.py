"""
Persistent metric implementations wrapping the functional metrics.
For some reason the torchmetric non-functional versions of these metrics
    do not support per-sample accumulation or non accumulation while
    the functional versions do. Here the functional metrics are wrapped
    under a torch.nn.Module interface to allow functional accumulation
    behavior as well as avoid needing to constantly reload networks
    behind the metrics.
"""

import torch
from torch import nn


class ReusableLPIPS(nn.Module):
    """LPIPS metric with one network instance and no accumulated metric state."""

    def __init__(self, net_type: str = "alex", normalize: bool = False) -> None:
        super().__init__()
        from torchmetrics.functional.image.lpips import _NoTrainLpips

        self.network = _NoTrainLpips(net=net_type)
        self.normalize = normalize

    def forward(
        self,
        degraded: torch.Tensor,
        reference: torch.Tensor,
        *,
        reduction: str | None = "none",
    ) -> torch.Tensor:
        from torchmetrics.functional.image.lpips import _lpips_compute, _lpips_update

        loss = _lpips_update(
            degraded,
            reference,
            net=self.network,
            normalize=self.normalize,
        )
        return _lpips_compute(loss, reduction)


class ReusableDISTS(nn.Module):
    """DISTS metric with one network instance and per-sample output support."""

    def __init__(self) -> None:
        super().__init__()
        from torchmetrics.functional.image.dists import DISTSNetwork

        self.network = DISTSNetwork()

    def forward(
        self,
        degraded: torch.Tensor,
        reference: torch.Tensor,
        *,
        reduction: str | None = "none",
    ) -> torch.Tensor:
        scores = self.network(degraded, reference, require_grad=False)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        if reduction == "sum":
            return scores.sum()
        if reduction == "mean":
            return scores.mean()
        if reduction is None or reduction == "none":
            return scores
        raise ValueError(
            f"reduction must be one of 'sum', 'mean', 'none', or None; received {reduction!r}."
        )
