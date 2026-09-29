"""
Utility functions for custom image quality metrics.
"""

import torch


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
    Average a per-pixel map over select pixels defined by mask,
        with extra small value/empty mask handling to avoid division by zero and NaNs.

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
    Needed to catch version specific behavior differences of torchmetrics
        `structural_similarity_index_measure` in `return_full_image=True` mode.
        Some versions of torchmetrics may return a larger spatial map of SSIM
        due to padding not being removed. The scalar return is correct in these
        cases because an padding aware mask is applied to reduce the spatial map,
        however, the full spatial map being returned is technically incorrect
        and contain extra trailing rows and columns.
    This helper validates the spatial shape of the returned full ssim map
        against the expected spatial shape (derived from input images) and
        drops the padding in the event torchmetrics doesn't drop them.

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
