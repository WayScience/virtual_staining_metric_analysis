"""Matched reference and degradation montage plots."""

from numbers import Integral
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from utils.fov_ingestion.collection import MicroscopyCollection
from utils.fov_ingestion.writer import digest
from utils.plot_utils import add_scalebar, reference_norm


def save_degradation_montages(
    reference: MicroscopyCollection,
    variants: list[tuple[str, MicroscopyCollection]],
    fov_id: str,
    output_dir: Path,
    *,
    channels: list[str] | None = None,
    crop_size: int | None = 256,
) -> list[Path]:
    """Save one family × severity grid per intensity channel, in raw units.

    Catalog identities are checked before rendering. Pixels are read one channel
    and variant at a time; only cropped/strided display arrays remain in the figure.
    All panels share the reference crop's P1–P99.5, gamma=0.7 normalization.
    Smaller catalogs are supported, provided every family has the same levels.
    """
    if crop_size is not None and (
        isinstance(crop_size, bool) or not isinstance(crop_size, Integral) or crop_size < 1
    ):
        raise ValueError("crop_size must be a positive integer or None")

    def completed(collection):
        marker = collection.root / "_SUCCESS"
        if (not marker.is_file() or marker.read_text().strip() != collection.fingerprint
                or collection.fingerprint != digest(collection.spec)):
            raise ValueError("Montages require completed collections")

    completed(reference)
    if reference.spec.get("kind") == "degradation":
        raise ValueError("Supply the original reference collection")
    metadata = reference.metadata().loc[lambda frame: frame.fov_id.eq(fov_id)]
    if len(metadata) != 1:
        raise ValueError(f"Unknown or duplicate reference FOV: {fov_id}")
    info = metadata.iloc[0]
    pixel_size = float(info.pixel_size_um)
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("Scale bar requires positive pixel_size_um")
    registry = reference.images.loc[
        reference.images.fov_id.eq(fov_id) & reference.images.role.eq("intensity")
    ]
    channels = list(registry.channel) if channels is None else list(channels)
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("Select distinct, nonempty intensity channels")

    def record(collection, channel):
        rows = collection.images.loc[
            collection.images.fov_id.eq(fov_id)
            & collection.images.role.eq("intensity")
            & collection.images.channel.eq(channel)
        ]
        if len(rows) != 1:
            raise ValueError(f"{fov_id}/{channel}: expected exactly one intensity row")
        row = rows.iloc[0]
        divisor = float(row.normalization_divisor)
        if row.intensity_scale != "raw" or not np.isfinite(divisor) or divisor <= 0:
            raise ValueError(f"{fov_id}/{channel}: expected raw intensity scale and positive divisor")
        return row

    reference_rows = {channel: record(reference, channel) for channel in channels}
    families = {}
    for level, collection in variants:
        completed(collection)
        spec = collection.spec
        if spec.get("kind") != "degradation":
            raise ValueError("Variants must be completed degradation collections")
        if (spec.get("reference_fingerprint") != reference.fingerprint
                or not spec.get("reference_collection_uri")
                or (collection.root / spec["reference_collection_uri"]).resolve()
                != reference.root.resolve()):
            raise ValueError("Variant does not link to the supplied reference collection")
        family = spec.get("transform", {}).get("name")
        if not isinstance(family, str) or not family or not isinstance(level, str) or not level:
            raise ValueError("Variants require nonempty family and level names")
        group = families.setdefault(family, {})
        if level in group:
            raise ValueError(f"Duplicate family/level: {family}/{level}")
        group[level] = collection
        for channel, source in reference_rows.items():
            row = record(collection, channel)
            if (row.reference_image_id != source.image_id
                    or (row.height, row.width) != (source.height, source.width)
                    or row.normalization_divisor != source.normalization_divisor):
                raise ValueError(f"{fov_id}/{channel}: variant row differs from reference")
    if not families:
        raise ValueError("Supply at least one degradation variant")
    levels = list(next(iter(families.values())))
    if any(set(group) != set(levels) for group in families.values()):
        raise ValueError("All families must provide the same severity levels")

    def read_plane(collection, channel, shape):
        plane = collection.read_fov(
            fov_id, channels=[channel], include_masks=False, normalize=False
        )[("intensity", channel)]
        if plane.ndim != 2 or not plane.size or plane.shape != shape:
            raise ValueError(f"{fov_id}/{channel}: expected matching nonempty 2D shapes")
        if not np.isfinite(plane).all():
            raise ValueError(f"{fov_id}/{channel}: nonfinite pixels")
        return plane

    family_labels = {
        "elastic_grid": "Elastic deformation", "gauss_noise": "Gaussian noise",
        "gaussian_blur": "Gaussian blur", "random_gamma": "Gamma",
        "dilate": "Dilation", "erode": "Erosion",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for channel, source in reference_rows.items():
        shape = (int(source.height), int(source.width))
        plane = read_plane(reference, channel, shape)
        height, width = shape if crop_size is None else tuple(min(crop_size, n) for n in shape)
        y, x = (shape[0] - height) // 2, (shape[1] - width) // 2
        crop = plane[y:y + height, x:x + width]
        norm = reference_norm(crop)
        stride = max(1, (max(height, width) + 1023) // 1024)
        reference_view = crop[::stride, ::stride].copy()
        del crop, plane
        fig = Figure(
            figsize=(2.6 * (len(levels) + 1), 2.7 * len(families) + 1.2),
            layout="constrained", facecolor="black",
        )
        try:
            axes = fig.subplots(len(families), len(levels) + 1, squeeze=False)
            for i, (family, group) in enumerate(families.items()):
                for j, level in enumerate([None, *levels]):
                    ax = axes[i, j]
                    if level is None:
                        view = reference_view
                    else:
                        plane = read_plane(group[level], channel, shape)
                        view = plane[y:y + height:stride, x:x + width:stride].copy()
                        del plane
                    ax.imshow(view, cmap="gray", norm=norm, interpolation="nearest")
                    ax.axis("off")
                    if i == 0:
                        ax.set_title("Reference" if level is None else level.title(), color="white")
                    if j == 0:
                        ax.text(
                            -0.06, 0.5, family_labels.get(family, family.replace("_", " ").title()),
                            transform=ax.transAxes, ha="right", va="center", rotation=90,
                            color="white",
                        )
                if i == 0:
                    add_scalebar(axes[i, 0], width, pixel_size, stride)
            biology = info.get("cell_line")
            if pd.isna(biology):
                biology = info.get("cell_type", "Unknown biology")
            fig.suptitle(
                f"{biology} | seeding density {info.seeding_density}\n"
                f"{info.plate} / {info.well} / site {info.site} | {fov_id} | {channel}\n"
                f"Crop x=[{x}, {x + width}), y=[{y}, {y + height})", color="white",
            )
            fig.supxlabel(
                "Reference P1–P99.5, γ=0.7 shared across all panels; display ≤1024 px/side",
                color="white",
            )
            path = output_dir / (
                f"montage_{quote(str(fov_id), safe='')}_{quote(str(channel), safe='')}.png"
            )
            FigureCanvasAgg(fig).print_figure(path, dpi=150, facecolor="black")
            paths.append(path)
        finally:
            fig.clear()
    return paths
