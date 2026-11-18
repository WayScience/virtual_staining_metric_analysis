"""
Build the QC-filtered, annotated PedCP image file index from downloaded loaddata
Fixing image pathing in relation to local image data.
"""

from collections.abc import Mapping
from functools import cache
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from . import plots
from . import recipe as r
from .artifacts import Artifacts, atomic_write, selected_artifacts


def _validate_index_keys(frame, keys):
    """Check that the specified keys are unique and non-null."""
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Keys must be unique and nonnull: {keys}")


def file_index_qc_mask(qc: pd.DataFrame) -> pd.Series:
    """Return passing FOV flags from pooled per-channel QC; constants score zero."""
    _validate_index_keys(qc, r.FIELD_KEYS)
    passing = pd.Series(True, index=qc.index)
    for metric, threshold in (
        (r.BLUR_METRIC, r.BLUR_THRESHOLD),
        (r.SATURATION_METRIC, r.SATURATION_THRESHOLD),
    ):
        for channel in r.CHANNELS:
            values = pd.to_numeric(qc[f"{metric}_{channel}"], errors="raise")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite QC metric: {metric}_{channel}")
            scores = 0 if values.nunique() <= 1 else (values - values.mean()) / values.std(ddof=0)
            passing &= ~(np.abs(scores) > threshold)
    return passing


def validate_file_index_paths(frame: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    """Remap image paths and retain FOVs with all channel TIFFs; this does not apply QC."""
    _validate_index_keys(frame, r.FIELD_KEYS)
    root = Path(image_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    @cache
    def resolved_parent(path):
        return path.resolve()

    @cache
    def directory(source):
        relative = PurePosixPath(source).relative_to(r.SOURCE_PATH_PREFIX)
        if ".." in relative.parts:
            raise ValueError("Image directory escapes root")
        return (root / relative).resolve()

    def remap(source, filename):
        try:
            local, name = directory(source), Path(filename)
            if name.is_absolute() or ".." in name.parts:
                return None
            target = resolved_parent((local / name).parent) / name.name
            if target.is_symlink():
                target = target.resolve()
            if not target.is_relative_to(root) or target.suffix.lower() not in (".tif", ".tiff"):
                return None
            return str(local) if target.is_file() else None
        except (TypeError, ValueError, OSError, RuntimeError):
            return None

    result = frame.copy()
    paths = [f"PathName_{channel}" for channel in r.CHANNELS]
    for channel, path in zip(r.CHANNELS, paths):
        pairs = frame[[path, f"FileName_{channel}"]]
        result[path] = [remap(*pair) for pair in pairs.itertuples(index=False, name=None)]
    keep = result[paths].notna().all(axis=1)
    print(f"TIFF completeness: dropped {int((~keep).sum())} of {len(frame)} fields")
    return result.loc[keep].copy()


def build_well_annotations(
    barcode: pd.DataFrame, platemaps: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Expand barcode annotations over platemaps, preserving original columns."""
    _validate_index_keys(barcode, ["barcode"])
    if barcode[["time_point", "platemap_file"]].isna().any().any() or barcode.empty:
        raise ValueError("Barcode time points and platemap names are required")
    expanded = []
    for row in barcode.to_dict("records"):
        frame = platemaps[row["platemap_file"]]
        _validate_index_keys(frame, ["well"])
        if set(frame.columns) & (set(row) | set(r.FIELD_KEYS[:2])):
            raise ValueError("Annotation column overlap")
        expanded.append(frame.assign(Metadata_Plate=row["barcode"], time_point=row["time_point"]))
    result = pd.concat(expanded, ignore_index=True).rename(columns={"well": "Metadata_Well"})
    _validate_index_keys(result, r.FIELD_KEYS[:2])
    return result


def prepare_file_index(config, *, cohort="all") -> dict[str, Path]:
    """Apply shared six-plate QC, then write the selected reference/training indexes."""
    outputs = selected_artifacts(config, cohort)
    artifacts = Artifacts(config)
    if config.image_root is None:
        raise ValueError("Preparation requires image_root")
    columns = r.FIELD_KEYS + [
        f"{metric}_{channel}"
        for metric in (r.BLUR_METRIC, r.SATURATION_METRIC)
        for channel in r.CHANNELS
    ]
    sources = []
    for plate in r.PLATES:
        quality = pd.read_csv(artifacts.raw / f"whole_img_qc_output/{plate}.csv")[columns]
        if set(quality.Metadata_Plate) != {plate}:
            raise ValueError(f"Unexpected or empty QC plate: {plate}")
        sources.append(quality)
    qc = pd.concat(sources, ignore_index=True)
    passing = file_index_qc_mask(qc)
    qc_keys = pd.MultiIndex.from_frame(qc[r.FIELD_KEYS])
    barcode = pd.read_csv(artifacts.raw / "Barcode_platemap_pilot_data.csv")
    files = [name for name in r.SOURCE_FILES if name.startswith("platemaps/")]
    maps = {Path(name).stem: pd.read_csv(artifacts.raw / name) for name in files}
    curated = build_well_annotations(barcode, maps)
    if set(barcode.barcode) != set(r.PLATES) or set(curated.Metadata_Plate) != set(r.PLATES):
        raise ValueError("Barcode and annotations must cover exactly the six recipe plates")
    reference = curated.loc[curated.time_point.eq(r.TIME_POINT)]
    annotations = {}
    for name, output in outputs.items():
        if output.density is None:
            annotations[name] = reference
        else:
            training = curated.assign(
                platemap_file=curated.Metadata_Plate.map(barcode.set_index("barcode").platemap_file)
            )
            annotations[name] = training.loc[
                training.platemap_file.eq(r.TRAIN_PLATEMAP)
                & training.cell_line.eq(r.TRAIN_CELL_LINE)
                & training.time_point.isin(r.TRAIN_TIME_POINTS)
                & training.seeding_density.eq(output.density)
            ]
    stages, selected = [], {name: [] for name in outputs}
    for plate in r.PLATES:
        raw = pd.read_csv(artifacts.raw / f"loaddatas/{plate}_concatenated.csv")
        if set(raw.Metadata_Plate) != {plate}:
            raise ValueError(f"Unexpected or empty LoadData plate: {plate}")
        valid = validate_file_index_paths(raw, config.image_root)
        keys = pd.MultiIndex.from_frame(valid[r.FIELD_KEYS])
        if not keys.isin(qc_keys).all():
            raise ValueError(f"Missing QC coverage: {plate}")
        good = valid.loc[keys.isin(qc_keys[passing])]
        for name, annotation in annotations.items():
            timed = good.loc[good.Metadata_Plate.isin(annotation.Metadata_Plate)]
            annotated = timed.merge(
                annotation, on=r.FIELD_KEYS[:2], validate="many_to_one", suffixes=(False, False)
            )
            selected[name].append(annotated)
            if name == "reference":
                stages.append((len(raw), len(valid), len(good), len(timed), len(annotated)))
    # Validate every requested selection before replacing any prepared index.
    frames = {name: pd.concat(parts, ignore_index=True) for name, parts in selected.items()}
    for name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"{name}: no annotated QC-passing fields")
        if outputs[name].density is not None:
            missing = set(r.TRAIN_TIME_POINTS) - set(frame.time_point)
            if missing:
                raise ValueError(f"{name}: no QC-passing fields at timepoints {sorted(missing)}")
            print(f"{name}: prepared coverage\n{training_coverage(frame).to_string()}")
    for name, frame in frames.items():
        atomic_write(
            outputs[name].export_parquet,
            lambda p, frame=frame: frame.to_parquet(p, compression="zstd", index=False),
        )
        if outputs[name].density is not None:
            plots.plot_counts(
                training_coverage(frame)[["FOVs"]],
                outputs[name].figures / "preparation_retention.png",
                f"{name}: QC-passing annotated training fields by timepoint",
            )
    qc_counts = qc.assign(passing=passing).groupby("Metadata_Plate").passing.agg(["size", "sum"])
    qc_counts.columns = ["Before QC", "After QC"]
    for output in outputs.values():
        plots.plot_counts(
            qc_counts, output.figures / "qc_retention.png", "QC: six plates before TIFF/cohort"
        )
    if "reference" in outputs:
        timepoints = barcode.set_index("barcode").time_point
        stagecounts = pd.DataFrame(
            stages,
            index=pd.Index(
                [f"{plate} ({timepoints[plate]:g} h)" for plate in r.PLATES],
                name="Acquisition plate (imaging timepoint)",
            ),
            columns=[
                "Raw", "TIFF complete", "QC passed",
                f"Select imaging timepoint\n{r.TIME_POINT:g} h only (cohort filter)",
                "Annotated",
            ],
        )
        plots.plot_counts(
            stagecounts,
            artifacts.figures / "preparation_retention.png",
            f"Reference preparation: QC followed by imaging-timepoint selection ({r.TIME_POINT:g} h)",
        )
    return {name: output.export_parquet for name, output in outputs.items()}


def training_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Count FOVs and distinct acquisition wells, including absent timepoints."""
    fields = frame.groupby("time_point").size().rename("FOVs")
    wells = (
        frame.drop_duplicates(["Metadata_Plate", "Metadata_Well"])
        .groupby("time_point")
        .size()
        .rename("Wells")
    )
    return pd.concat([fields, wells], axis=1).reindex(r.TRAIN_TIME_POINTS, fill_value=0)
