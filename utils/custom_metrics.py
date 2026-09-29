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
from torchmetrics.functional.image import structural_similarity_index_measure

from .custom_metric_utils import (
    _center_crop,
    _check_masking_settings,
    _check_paired_inputs,
    _foreground_mask,
    _masked_mean,
    _reduce,
)


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
        return _reduce(scores, reduction)


class ForegroundPSNR(nn.Module):
    """
    PSNR restricted to the foreground pixels of the reference image.

    Only the mean squared error is masked; the peak stays the full data range so
        that scores remain on the same scale as unmasked PSNR.
    """

    def __init__(
        self,
        data_range: tuple[float, float] = (0.0, 1.0),
        num_bins: int = 256,
    ) -> None:
        super().__init__()
        _check_masking_settings(data_range, num_bins)
        self.data_range = data_range
        self.num_bins = num_bins
        self._peak = data_range[1] - data_range[0]

    def forward(
        self,
        degraded: torch.Tensor,
        reference: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        reduction: str | None = "none",
    ) -> torch.Tensor:
        """
        Compute foreground PSNR for a batch of paired images.

        :param degraded: Degraded image batch of shape (N, 1, H, W).
        :param reference: Reference image batch of shape (N, 1, H, W).
        :param mask: Optional explicit mask; an Otsu mask of the reference is used when omitted.
        :param reduction: One of 'sum', 'mean', 'none', or None.
        :return: Per-sample PSNR of shape (N,) when no reduction is applied.
        """
        _check_paired_inputs(degraded, reference)
        mask = _foreground_mask(reference, mask, self.data_range, self.num_bins)

        squared_error = (degraded - reference) ** 2
        mean_squared_error = _masked_mean(squared_error, mask)

        peak = torch.as_tensor(
            self._peak,
            device=degraded.device,
            dtype=mean_squared_error.dtype,
        )
        scores = 20.0 * torch.log10(peak) - 10.0 * torch.log10(mean_squared_error)

        return _reduce(scores, reduction)


class ForegroundSSIM(nn.Module):
    """
    SSIM restricted to the foreground pixels of the reference image.

    The functional SSIM is used so that no metric state is accumulated between
        calls, and its full similarity map is averaged over the mask instead of
        over every pixel.
    """

    def __init__(
        self,
        data_range: tuple[float, float] = (0.0, 1.0),
        num_bins: int = 256,
        gaussian_kernel: bool = True,
        sigma: float = 1.5,
        kernel_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        super().__init__()
        _check_masking_settings(data_range, num_bins)
        self.data_range = data_range
        self.num_bins = num_bins
        self._peak = data_range[1] - data_range[0]
        self._ssim_kwargs = {
            "gaussian_kernel": gaussian_kernel,
            "sigma": sigma,
            "kernel_size": kernel_size,
            "k1": k1,
            "k2": k2,
        }

        # torchmetrics reflect-pads the inputs before convolving, then discards that
        # border when reducing the similarity map; mirror the same footprint so the
        # foreground score stays comparable with unmasked SSIM
        footprint = int(3.5 * sigma + 0.5) * 2 + 1 if gaussian_kernel else kernel_size
        self._border = (footprint - 1) // 2

    def forward(
        self,
        degraded: torch.Tensor,
        reference: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        reduction: str | None = "none",
    ) -> torch.Tensor:
        """
        Compute foreground SSIM for a batch of paired images.

        :param degraded: Degraded image batch of shape (N, 1, H, W).
        :param reference: Reference image batch of shape (N, 1, H, W).
        :param mask: Optional explicit mask; an Otsu mask of the reference is used when omitted.
        :param reduction: One of 'sum', 'mean', 'none', or None.
        :return: Per-sample SSIM of shape (N,) when no reduction is applied.
        """
        _check_paired_inputs(degraded, reference)
        mask = _foreground_mask(reference, mask, self.data_range, self.num_bins)

        _, similarity_map = structural_similarity_index_measure(
            degraded,
            reference,
            data_range=self._peak,
            reduction="none",
            return_full_image=True,
            **self._ssim_kwargs,
        )

        # some torchmetrics releases return the uncropped map while reducing only its
        # interior, so trim the border here and align the mask to whatever remains
        if self._border > 0 and similarity_map.shape[-2:] == mask.shape[-2:]:
            border = self._border
            similarity_map = similarity_map[..., border:-border, border:-border]
        mask = _center_crop(mask, similarity_map.shape[-2:])

        return _reduce(_masked_mean(similarity_map, mask), reduction)
