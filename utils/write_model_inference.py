"""
Utilities for batched model inference over reference Lance records.

Given a collection of virtual staining models, corresponding catalogue, and
    previously written reference image records, this utility will run the models
    over the images and write the predictions to the specified output directory.

Mirrors the structure of all other write_* utilities in this project:
- defines the schema and fingerprint for model-prediction stacks.
- provides a main orchestration function receiving needed components to
    produce new images from existing ones (virtual staining models and input images here).
- writes the resulting model-prediction records to the output directory using shared
    Lance writer utility.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from tqdm.auto import tqdm

from .encoding import decode_pixel_record, encode_pixels
from .iter_data import iter_lance_fragments
from .parquet_writer import LanceWriter
from .write_reference_image import SHARD_SIZE


def _serialize_model_catalogue(model_catalogue: pd.DataFrame) -> tuple[str, str]:
    """
    Serialize the prediction-page catalogue and return it with its fingerprint.

    :param: model_catalogue: DataFrame containing the model catalogue to serialize.
    :return: A tuple containing the serialized JSON string of the catalogue and
        its SHA-256 fingerprint.
    """
    if model_catalogue.empty:
        raise ValueError("model_catalogue must contain at least one model.")

    catalogue = model_catalogue.reset_index(drop=True).copy()
    if "page_index" in catalogue.columns:
        expected_indexes = list(range(len(catalogue)))
        if catalogue["page_index"].tolist() != expected_indexes:
            raise ValueError("model_catalogue page_index must be consecutive and zero-based.")
    else:
        catalogue.insert(0, "page_index", range(len(catalogue)))

    catalogue_json = catalogue.to_json(
        orient="records",
        date_format="iso",
        double_precision=15,
    )
    fingerprint = hashlib.sha256(catalogue_json.encode("utf-8")).hexdigest()
    return catalogue_json, fingerprint


def _model_prediction_record_schema(
    reference_schema: pa.Schema,
    model_catalogue: pd.DataFrame,
) -> tuple[pa.Schema, str]:
    """
    Create the output schema and fingerprint for a model-prediction stack.

    :param reference_schema: The schema of the reference image stack.
    :param model_catalogue: DataFrame containing the model catalogue.
    :return: A tuple containing the output schema for the model-prediction stack and
        the SHA-256 fingerprint of the model catalogue.
    """
    if "model_catalogue_fingerprint" in reference_schema.names:
        raise ValueError("Reference schema already contains model_catalogue_fingerprint.")

    catalogue_json, fingerprint = _serialize_model_catalogue(model_catalogue)
    metadata = {
        **(reference_schema.metadata or {}),
        b"record_kind": b"model_prediction_stack",
        b"pixel_encoding": b"C-contiguous little-endian float32",
        b"stack_axes": b"MYX",
        b"model_catalogue": catalogue_json.encode("utf-8"),
        b"torch_version": torch.__version__.encode("utf-8"),
    }
    schema = reference_schema.append(
        pa.field("model_catalogue_fingerprint", pa.string(), nullable=False)
    ).with_metadata(metadata)
    return schema, fingerprint


def _prediction_batch(
    images: Sequence[np.ndarray],
    models: Sequence[torch.nn.Module],
    device: torch.device,
) -> np.ndarray:
    """
    Run every model over an image batch and return a ``BMYX`` array.

    :param images: A sequence of input images as NumPy arrays.
    :param models: A sequence of PyTorch models to run on the images.
    :param device: The device on which to run the models.
    :return: A NumPy array containing the predictions with shape ``BMYX``.
    """
    if not images:
        raise ValueError("Cannot run inference on an empty image batch.")

    image_shapes = {image.shape for image in images}
    if len(image_shapes) != 1:
        raise ValueError(f"All images in a batch must have one shape; received {image_shapes}.")

    input_tensor = torch.from_numpy(np.stack(images)).unsqueeze(1).to(device)
    predictions = []

    with torch.inference_mode():
        for model in models:
            output = model(input_tensor)
            if output.ndim == 4 and output.shape[1] == 1:
                output = output[:, 0]
            elif output.ndim != 3:
                raise ValueError(
                    "Each model must return B1YX or BYX output; "
                    f"received shape {tuple(output.shape)}."
                )

            if (
                output.shape[0] != input_tensor.shape[0]
                or output.shape[1:] != input_tensor.shape[2:]
            ):
                raise ValueError(
                    "Model output must preserve batch and spatial dimensions; "
                    f"input shape is {tuple(input_tensor.shape)}, output shape is {tuple(output.shape)}."
                )
            predictions.append(output.detach().to(device="cpu", dtype=torch.float32))

    return torch.stack(predictions, dim=1).numpy()


def write_model_predictions(
    *,
    models: Sequence[torch.nn.Module],
    model_catalogue: pd.DataFrame,
    output_dir: Path,
    reference_lance_dir: Path,
    device: torch.device | str | None = None,
    batch_size: int = 16,
    overwrite: bool = False,
) -> None:
    """
    Main orchestration function to run virtual staining models over reference images
        and write the resulting predictions to the specified output directory.

    :param models: A sequence of PyTorch models to run on the reference images.
    :param model_catalogue: A DataFrame containing metadata for the models.
    :param output_dir: The directory where the model predictions will be written.
    :param reference_lance_dir: The directory containing the reference image records.
    :param device: The device on which to run the models.
    :param batch_size: The number of images to process in each batch.
    :param overwrite: Whether to overwrite existing output files.
    """

    models = list(models)
    if not models:
        raise ValueError("models must contain at least one model.")
    if len(models) != len(model_catalogue):
        raise ValueError(
            f"models has {len(models)} entries but model_catalogue has {len(model_catalogue)} rows."
        )
    if any(model is None for model in models):
        raise ValueError("models cannot contain missing entries.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    inference_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    for model in models:
        model.eval()
        model.to(inference_device)

    reference_dataset = lance.dataset(reference_lance_dir)
    fragment_count = len(reference_dataset.get_fragments())
    output_schema, catalogue_fingerprint = _model_prediction_record_schema(
        reference_dataset.schema,
        model_catalogue,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(
        iter_lance_fragments(reference_lance_dir),
        total=fragment_count,
        desc="Writing model predictions",
    )

    with LanceWriter(
        output_dir=output_dir,
        schema=output_schema,
        overwrite=overwrite,
        shardsize=SHARD_SIZE,
        dataset_name="data.lance",
    ) as writer:
        for _, _, reference_table in progress:
            references = reference_table.to_pylist()
            for start in range(0, len(references), batch_size):
                batch_records = references[start : start + batch_size]
                batch_images = [
                    decode_pixel_record(record, expected_axes="YX")[0] for record in batch_records
                ]
                prediction_stacks = _prediction_batch(
                    batch_images,
                    models,
                    inference_device,
                )

                for reference, prediction_stack in zip(
                    batch_records,
                    prediction_stacks,
                    strict=True,
                ):
                    writer.add_row(
                        {
                            **reference,
                            "model_catalogue_fingerprint": catalogue_fingerprint,
                            **encode_pixels(
                                prediction_stack,
                                axes="MYX",
                                require_finite=True,
                            ),
                        }
                    )
