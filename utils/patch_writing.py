"""
Helper utilities for interacting with virtual_stain_flow.datasets.crop_dataset
(https://github.com/WayScience/virtual_stain_flow.git) to retrieve metadata
and write normalized patches to disk while keeping file index.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tifffile import imwrite
from tqdm.auto import tqdm


def _path_component(value):
    """Return a short filesystem-safe representation of a metadata value."""
    text = str(value)
    return "".join(
        character if character.isalnum() or character in "-_" else "-" for character in text
    )


def _site_label(value):
    """Format integer-like site identifiers consistently (for example, 2 -> 02)."""
    if pd.notna(value) and float(value).is_integer():
        return f"{int(value):02d}"
    return _path_component(value)


def get_crop_file_index(
    cropped_dataset: Any,
    file_index_to_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Interacts with virtual_stain_flow.datasets.crop_dataset
        to retrieve internal cropping and source file information and arrange
        as a dataframe with matching length and index order to the dataset,
        with each row containing:
            1) the crop bounding box definition
            2) the corresponding source image file paths
        and merges with the provided file index to metadata mapping to enrich
        the crop index by source image level metadata.

    :param cropped_dataset: The cropped dataset containing the manifest.
    :param file_index_to_metadata: A DataFrame mapping source image file indices to metadata.
    :return: A DataFrame with one row per crop, including a "crop_dataset_index" column.
    """

    ## Access cropped dataset internal structures
    manifest = getattr(cropped_dataset, "manifest", None)
    if not manifest:
        raise ValueError("Cropped dataset does not have a manifest.")
    # internal list of dataclasses defining bounding boxes for each crop
    # and their corresponding source image file indices
    crops: list[dataclass] | None = getattr(manifest, "crops", None)
    if not crops:
        raise ValueError("Cropped dataset manifest does not have crops.")
    # internal list of source image file indices corresponding
    source_index = getattr(cropped_dataset, "file_index", None)
    if source_index is None:
        raise ValueError("Cropped dataset does not have a file_index.")

    ## 1. flatten internal crop definitions to dataframe
    # dataframe contains x, y, width, height etc.
    crop_manifest = pd.DataFrame(
        crop.to_dict()
        for crop in crops
        # source_image_row corresponds to rows in source_index df
    ).rename(columns={"manifest_idx": "source_image_row"})
    # crop_dataset_index corresponds to the dataset __get_item__ index
    crop_manifest.insert(0, "crop_dataset_index", np.arange(len(crop_manifest)))

    ## 2. expand source image rows to match the crop manifest and concat
    # dataframe contains channel1 filepath, channel2 filepath etc.
    source_rows = crop_manifest["source_image_row"].to_numpy()
    source_index_expanded = (
        source_index.copy().iloc[source_rows].reset_index(names="source_image_index")
    )

    ## 3. map every crop to expanded source image file paths
    # this dataframe now contains x, y, width, height + channel1 path, channel2 path ...
    crop_index = pd.concat(
        [crop_manifest.reset_index(drop=True), source_index_expanded],
        axis="columns",
    )

    if file_index_to_metadata is None:
        return crop_index

    ## 4. merge with more image level metadata if provided
    crop_index_metadata = pd.merge(
        crop_index,
        file_index_to_metadata,
        how="left",
        on=list(source_index.columns),
        validate="m:1",
    )
    if len(crop_index) != len(crop_index_metadata):
        raise ValueError("Mismatch between crop index and merged metadata.")

    return crop_index_metadata


def write_normalized_crops(
    dataset: Any,
    crop_metadata: pd.DataFrame,
    output_dir: str,
    crop_indices: list[int] | None = None,
    overwrite: bool = False,
    index_filename: str = "index.csv",
) -> pd.DataFrame:
    """
    Write one float32 TIFF per crop channel and return file index

    :params dataset: The PyTorch dataset containing the crops.
    :param crop_metadata: A DataFrame containing metadata for each crop.
    :param output_dir: The directory where the TIFF files will be written.
    :param crop_indices: A list of crop indices to write. If None, all crops are written.
    :param overwrite: Whether to overwrite existing TIFF files.
    :param index_filename: The name of the CSV file to write the file index to.
    :return: A DataFrame containing the file index for the written crops.
    """
    output_dir = Path(output_dir)
    tiff_root = output_dir / "tiffs"
    tiff_root.mkdir(parents=True, exist_ok=True)

    if crop_indices is None:
        crop_indices = range(len(dataset))
    crop_indices = list(crop_indices)

    input_channels = list(dataset.input_channel_keys)
    target_channels = list(dataset.target_channel_keys)
    records = []

    for crop_dataset_index in tqdm(crop_indices, desc="Writing normalized crops"):
        metadata = crop_metadata.iloc[crop_dataset_index]
        if metadata["crop_dataset_index"] != crop_dataset_index:
            raise ValueError("crop_metadata is not ordered by crop_dataset_index.")

        input_stack, target_stack = dataset[crop_dataset_index]
        channel_groups = (
            (input_channels, input_stack),
            (target_channels, target_stack),
        )

        plate = _path_component(metadata["Metadata_Plate"])
        well = _path_component(metadata["Metadata_Well"])
        site = _site_label(metadata["Metadata_Site"])
        filename = (
            f"crop_{crop_dataset_index:06d}_x{int(metadata['x']):04d}_y{int(metadata['y']):04d}.tif"
        )

        for channel_names, image_stack in channel_groups:
            if image_stack.shape[0] != len(channel_names):
                raise ValueError("Tensor channel count does not match the dataset channel keys.")

            for channel_position, channel in enumerate(channel_names):
                channel_dir = tiff_root / plate / well / f"site_{site}" / channel
                channel_dir.mkdir(parents=True, exist_ok=True)
                output_path = channel_dir / filename

                image = image_stack[channel_position].detach().cpu().numpy()
                image = np.asarray(image, dtype=np.float32)
                if image.ndim != 2:
                    raise ValueError(f"Expected a 2D {channel} crop, found shape {image.shape}.")
                if overwrite or not output_path.exists():
                    imwrite(
                        output_path,
                        image,
                        photometric="minisblack",
                        metadata={"axes": "YX"},
                    )

                records.append(
                    {
                        "crop_dataset_index": crop_dataset_index,
                        "channel": channel,
                        "patch_path": output_path.relative_to(output_dir).as_posix(),
                        "source_image_file": str(metadata[channel]),
                    }
                )

    file_index = pd.DataFrame.from_records(records)
    if file_index.empty:
        raise ValueError("crop_indices did not contain any crops.")

    selected_metadata = crop_metadata.iloc[crop_indices]
    patch_index = file_index.merge(
        selected_metadata,
        on="crop_dataset_index",
        how="left",
        validate="many_to_one",
    )

    index_path = output_dir / index_filename
    temporary_index_path = index_path.with_name(f".{index_path.stem}.tmp.csv")
    patch_index.to_csv(temporary_index_path, index=False)
    temporary_index_path.replace(index_path)

    print(f"Indexed {len(patch_index):,} TIFF files in {index_path}")
    return patch_index
