"""Data ingestion summary plots (FOV counts, QC and montages)"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap, NoNorm, PowerNorm
from matplotlib.figure import Figure
from microfilm.microplot import Microimage

from utils.pedcp_data_integration.recipe import CORRECTED_CHANNELS, ILLUMINATION_METHODS


def plot_counts(counts: pd.DataFrame, path: Path, title: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(11, 5), layout="constrained")
    ax = fig.subplots()
    for plate, values in counts.iterrows():
        ax.plot(counts.columns, values, "o-", label=str(plate))
    ax.set(title=title, ylabel="FOV count", ylim=(0, None))
    ax.legend(title=counts.index.name or "Plate", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    FigureCanvasAgg(fig).print_figure(path, dpi=150)
    fig.clear()
    return path


def save_montages(collection, count: int, seed: int, output_dir: Path) -> list[Path]:
    """Sample sorted IDs reproducibly; validate full planes, stride only the display."""
    metadata = collection.metadata().sort_values("fov_id")
    selected = metadata.sample(n=min(count, len(metadata)), random_state=seed)
    expected = [("intensity", channel) for channel in CORRECTED_CHANNELS.values()]
    expected += [("labels", "cells"), ("labels", "nuclei")]
    colors = ListedColormap(["black", *colormaps["tab20"].colors])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in selected.itertuples(index=False):
        registry = collection.images.loc[collection.images.fov_id.eq(row.fov_id)]
        # Older corrected collections can still be previewed, without relabeling raw data.
        legacy = registry.role.eq("intensity") & registry.channel.isin(CORRECTED_CHANNELS)
        if legacy.any():
            if not registry.loc[legacy, "illumination_method"].eq(
                registry.loc[legacy, "channel"].map(ILLUMINATION_METHODS)
            ).all():
                raise ValueError(f"{row.fov_id}: cannot label uncorrected planes as Illum*")
            registry = registry.copy()
            registry.loc[legacy, "channel"] = registry.loc[legacy, "channel"].map(CORRECTED_CHANNELS)
        registry = registry.set_index(["role", "channel"])
        planes = collection.read_fov(row.fov_id, normalize=False)
        if legacy.any():
            planes = {
                (role, CORRECTED_CHANNELS.get(channel, channel) if role == "intensity" else channel): plane
                for (role, channel), plane in planes.items()
            }
        if len(registry) != 8 or not set(registry.index) == set(planes) == set(expected):
            raise ValueError(f"{row.fov_id}: expected six channels, cells and nuclei")
        shape = planes[expected[0]].shape
        fig = Figure(figsize=(14, 8), layout="constrained", facecolor="black")
        for ax, key in zip(fig.subplots(2, 4).ravel(), expected):
            plane, record = planes[key], registry.loc[key]
            if plane.ndim != 2 or not plane.size or plane.shape != shape:
                raise ValueError(f"{row.fov_id}/{key}: expected common nonempty 2D shape")
            if shape != (record.height, record.width) or plane.dtype != np.dtype(record["dtype"]):
                raise ValueError(f"{row.fov_id}/{key}: shape or dtype differs from registry")
            stride = max(1, (max(shape) + 1023) // 1024)
            view = plane[::stride, ::stride]
            if key[0] == "labels":
                if plane.dtype.kind not in "iu" or plane.min() < 0:
                    raise ValueError(f"{row.fov_id}/{key}: labels must be nonnegative integers")
                display = (view % 20 + 1).astype(np.uint8)
                display[view == 0] = 0
                ax.imshow(display, cmap=colors, norm=NoNorm(), interpolation="nearest")
            else:
                low, high = np.percentile(view, (1, 99.5))
                norm = PowerNorm(0.7, low, max(high, low + 1e-6), clip=True)
                ax.imshow(view, cmap="gray", norm=norm, interpolation="nearest")
                if key == expected[0]:
                    pixel_size = float(row.pixel_size_um)
                    if not np.isfinite(pixel_size) or pixel_size <= 0:
                        raise ValueError(f"{row.fov_id}: scale bar requires positive pixel_size_um")
                    # A readable 1/2/5 length, about one fifth of the physical FOV width.
                    target = shape[1] * pixel_size / 5
                    magnitude = 10 ** np.floor(np.log10(target))
                    length = max(value for value in (1, 2, 5) if value <= target / magnitude)
                    Microimage(images=None, ax=ax).add_scalebar(
                        unit="um",
                        scalebar_size_in_units=float(length * magnitude),
                        scalebar_unit_per_pix=pixel_size * stride,
                        scalebar_font_size=10,
                    )
            ax.set_title(key[1], color="white")
            ax.axis("off")
        biology = f"{row.cell_line} | seeding density {row.seeding_density}"
        identity = f"{row.plate} / {row.well} / site {row.site}\n{row.fov_id}"
        fig.suptitle(f"{biology}\n{identity}", color="white")
        fig.supxlabel("Sampled FOV; display ≤1024 px/side, P1–P99.5, γ=0.7", color="white")
        path = output_dir / f"montage_{quote(str(row.fov_id), safe='')}.png"
        FigureCanvasAgg(fig).print_figure(path, dpi=150)
        fig.clear()
        paths.append(path)
    return paths
