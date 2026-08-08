"""
Persistent metric implementations wrapping the functional metrics.
For some reason the torchmetric non-functional versions of these metrics
    do not support per-sample accumulation or non accumulation while
    the functional versions do. Here the functional metrics are wrapped
    under a torch.nn.Module interface to allow functional accumulation
    behavior as well as avoid needing to constantly reload networks
    behind the metrics.
This module also defines foreground-masked variants of SSIM and PSNR, which
    restrict the metric to the pixels selected by a per-image Otsu threshold
    of the reference image.
"""

import torch
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure


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


def _reduce(scores: torch.Tensor, reduction: str | None) -> torch.Tensor:
    """
    Apply a batch reduction to per-sample metric scores.

    :param scores: Per-sample scores of shape (N,).
    :param reduction: One of 'sum', 'mean', 'none', or None.
    :return: The reduced scores, or the unmodified scores when no reduction applies.
    """
    if reduction is None or reduction == "none":
        return scores
    if reduction == "sum":
        return scores.sum()
    if reduction == "mean":
        return scores.mean()
    raise ValueError(
        f"reduction must be one of 'sum', 'mean', 'none', or None; received {reduction!r}."
    )


def _check_paired_inputs(degraded: torch.Tensor, reference: torch.Tensor) -> None:
    """
    Validate that a degraded/reference pair is a batch of single channel images.

    :param degraded: Degraded image batch.
    :param reference: Reference image batch.
    """
    if degraded.shape != reference.shape:
        raise ValueError(
            f"degraded shape {tuple(degraded.shape)} must match "
            f"reference shape {tuple(reference.shape)}."
        )
    if degraded.device != reference.device:
        raise ValueError("degraded and reference must be on the same device.")
    if degraded.ndim != 4 or degraded.shape[1] != 1:
        raise ValueError(
            "Foreground metrics expect single-channel batches shaped (N, 1, H, W); "
            f"received {tuple(degraded.shape)}."
        )


def _otsu_threshold(
    images: torch.Tensor,
    data_range: tuple[float, float],
    num_bins: int,
) -> torch.Tensor:
    """
    Compute a per-image Otsu threshold over histogram bins spanning a fixed data range.

    Bin edges are fixed by `data_range` rather than derived from the batch, so the
        threshold of an image is independent of the other images it is batched with.
        This matters here because the reference image is broadcast across every
        degraded variant and then evaluated in arbitrarily sized chunks.

    :param images: Image batch of shape (N, 1, H, W).
    :param data_range: The (minimum, maximum) intensity spanned by the histogram.
    :param num_bins: Number of histogram bins.
    :return: Per-image thresholds of shape (N,); infinite where no valid split exists.
    """
    low, high = data_range
    bin_width = (high - low) / num_bins

    batch_size = images.shape[0]
    flat = images.reshape(batch_size, -1).clamp(min=low, max=high)
    bin_index = ((flat - low) / bin_width).floor().to(torch.long).clamp_(0, num_bins - 1)

    histogram = torch.zeros(
        (batch_size, num_bins),
        device=images.device,
        dtype=torch.float32,
    )
    histogram.scatter_add_(1, bin_index, torch.ones_like(bin_index, dtype=torch.float32))
    histogram = histogram / histogram.sum(dim=1, keepdim=True)

    bin_values = torch.arange(num_bins, device=images.device, dtype=torch.float32)
    weighted = histogram * bin_values

    # split candidate i puts bins [0, i] in the background and (i, num_bins) in the
    # foreground, hence dropping the final cumulative entry which leaves no foreground
    weight_background = histogram.cumsum(dim=1)[:, :-1]
    weight_foreground = 1.0 - weight_background
    sum_background = weighted.cumsum(dim=1)[:, :-1]
    sum_foreground = weighted.sum(dim=1, keepdim=True) - sum_background

    valid = (weight_background > 0) & (weight_foreground > 0)
    mean_background = sum_background / weight_background.clamp_min(torch.finfo(torch.float32).tiny)
    mean_foreground = sum_foreground / weight_foreground.clamp_min(torch.finfo(torch.float32).tiny)

    inter_class_variance = torch.where(
        valid,
        weight_background * weight_foreground * (mean_background - mean_foreground) ** 2,
        torch.full_like(weight_background, -1.0),
    )

    best_bin = inter_class_variance.argmax(dim=1)
    best_variance = inter_class_variance.gather(1, best_bin[:, None]).squeeze(1)

    # the threshold is the upper edge of the last background bin
    threshold = low + (best_bin.to(torch.float32) + 1.0) * bin_width

    # a constant (or otherwise unsplittable) image has no foreground, which the
    # downstream masked mean turns into NaN rather than a fabricated score
    return torch.where(best_variance > 0, threshold, torch.full_like(threshold, torch.inf))


def _foreground_mask(
    reference: torch.Tensor,
    mask: torch.Tensor | None,
    data_range: tuple[float, float],
    num_bins: int,
) -> torch.Tensor:
    """
    Resolve the foreground mask, deriving it from the reference image when absent.

    :param reference: Reference image batch of shape (N, 1, H, W).
    :param mask: Optional explicit mask of shape (N, 1, H, W).
    :param data_range: The (minimum, maximum) intensity spanned by the Otsu histogram.
    :param num_bins: Number of Otsu histogram bins.
    :return: Mask of shape (N, 1, H, W) matching the reference device and dtype.
    """
    if mask is None:
        threshold = _otsu_threshold(reference, data_range, num_bins)
        mask = reference > threshold.reshape(-1, 1, 1, 1)
    elif mask.shape != reference.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match reference shape {tuple(reference.shape)}."
        )
    return mask.to(device=reference.device, dtype=reference.dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Average a per-pixel map over the masked pixels of each sample.

    :param values: Per-pixel values of shape (N, 1, H, W).
    :param mask: Mask of shape (N, 1, H, W).
    :return: Per-sample means of shape (N,); NaN where a sample has an empty mask.
    """
    reduced_dims = tuple(range(1, values.ndim))
    numerator = (values * mask).sum(dim=reduced_dims)
    denominator = mask.sum(dim=reduced_dims)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny),
        torch.full_like(numerator, torch.nan),
    )


def _check_masking_settings(data_range: tuple[float, float], num_bins: int) -> None:
    """
    Validate the shared foreground masking configuration.

    :param data_range: The (minimum, maximum) intensity spanned by the Otsu histogram.
    :param num_bins: Number of Otsu histogram bins.
    """
    if len(data_range) != 2 or data_range[1] <= data_range[0]:
        raise ValueError(f"data_range must be an increasing (low, high) pair; got {data_range!r}.")
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2; got {num_bins}.")


def _center_crop(tensor: torch.Tensor, spatial_shape: torch.Size) -> torch.Tensor:
    """
    Center crop the trailing spatial dimensions of a tensor.

    :param tensor: Tensor of shape (N, C, H, W).
    :param spatial_shape: Target (height, width).
    :return: The tensor cropped to the target spatial shape.
    """
    height, width = tensor.shape[-2:]
    target_height, target_width = spatial_shape
    if (target_height, target_width) == (height, width):
        return tensor
    if target_height > height or target_width > width:
        raise ValueError(
            f"Cannot crop spatial shape {(height, width)} up to {(target_height, target_width)}."
        )
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return tensor[..., top : top + target_height, left : left + target_width]


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
