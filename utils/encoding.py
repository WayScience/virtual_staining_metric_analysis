"""
Utilities for encoding and decoding images as raw little-endian float32 bytes.
Intended use is for embedding images in Parquet records.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt

PixelDType = Literal["float32"]
PixelByteOrder = Literal["little"]

PIXEL_DTYPE: PixelDType = "float32"
PIXEL_BYTE_ORDER: PixelByteOrder = "little"

# The actual NumPy storage dtype:
#     <  = little-endian
#     f4 = 4-byte floating point
_FLOAT32_LE = np.dtype("<f4")

_SUPPORTED_NDIMS = frozenset({2, 3})


class EncodedFloat32Image(TypedDict):
    """
    Fields stored in each Parquet image record.
    The return type of `encode_pixels`.
    Add to pq.Table.from_pylist() row dictionary to write a
        full image record with all needed information for decoding.
    """

    pixels: bytes
    shape: list[int]
    axes: str
    dtype: PixelDType
    byte_order: PixelByteOrder


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """Normalize and validate a serialized image shape."""
    try:
        normalized = tuple(int(size) for size in shape)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid pixel shape: {shape!r}.") from error

    if len(normalized) not in _SUPPORTED_NDIMS:
        raise ValueError(
            f"Only 2D images and 3D image stacks are supported; received shape {normalized}."
        )

    if any(size <= 0 for size in normalized):
        raise ValueError(f"All pixel dimensions must be positive; received {normalized}.")

    return normalized


def _validate_axes(
    axes: str,
    ndim: int,
) -> str:
    """Validate an axis-label string against an array dimensionality."""
    if not isinstance(axes, str):
        raise TypeError(f"axes must be a string, received {type(axes).__name__}.")

    if len(axes) != ndim:
        raise ValueError(
            f"Axis label {axes!r} has length {len(axes)}, but the image has {ndim} dimensions."
        )

    if len(set(axes)) != len(axes):
        raise ValueError(f"Axis labels must be unique; received {axes!r}.")

    return axes


def encode_pixels(
    image: npt.ArrayLike,
    *,
    axes: str,
    require_finite: bool = False,
) -> EncodedFloat32Image:
    """
    Encode a 2D image or 3D image stack as raw little-endian float32 bytes.
    Add entire EncodedFloat32Image to pq.Table.from_pylist() row dictionary to
        write a  full image record with all needed information for decoding.

    e.g.:

        table = pa.Table.from_pylist(
            [{"record_id": record_id, **encode_pixels(image, axes="YX")}],
            schema=schema
        )
        pq.write_table(table, ...)
    """
    encoded = np.asarray(
        image,
        dtype=_FLOAT32_LE,
        order="C",
    )

    if encoded.ndim not in _SUPPORTED_NDIMS:
        raise ValueError(
            "Only 2D images and 3D image stacks are supported; "
            f"received an array with shape {encoded.shape}."
        )

    if any(size <= 0 for size in encoded.shape):
        raise ValueError(f"Image dimensions must be positive; received {encoded.shape}.")

    normalized_axes = _validate_axes(axes, encoded.ndim)

    if require_finite and not np.isfinite(encoded).all():
        raise ValueError("Image contains NaN or infinite pixel values.")

    return {
        "pixels": encoded.tobytes(order="C"),
        "shape": list(encoded.shape),
        "axes": normalized_axes,
        "dtype": PIXEL_DTYPE,
        "byte_order": PIXEL_BYTE_ORDER,
    }


def decode_pixels(
    pixel_bytes: bytes | bytearray | memoryview,
    shape: Sequence[int],
    dtype: str,
    byte_order: str,
    *,
    axes: str | None = None,
    expected_axes: str | None = None,
    copy: bool = False,
) -> npt.NDArray[np.float32]:
    """
    Decode a raw little-endian float32 image payload.

    :param pixel_bytes: Raw C-order pixel bytes.
    :param shape: Serialized image shape.
    :param dtype: Must be ``"float32"``.
    :param byte_order: Must be ``"little"``.
    :param axes: Optional serialized axis labels.
    :param expected_axes: If provided, require the serialized axes to match this value.
    :param copy: If False, return an array backed by the input buffer where possible.
                 If True, return an independent writable C-contiguous array.
    """
    if dtype != PIXEL_DTYPE or byte_order != PIXEL_BYTE_ORDER:
        raise ValueError(
            "Unsupported pixel encoding contract: "
            f"{dtype=}, {byte_order=}; expected "
            f"{PIXEL_DTYPE=}, {PIXEL_BYTE_ORDER=}."
        )

    normalized_shape = _normalize_shape(shape)

    if axes is not None:
        normalized_axes = _validate_axes(
            axes,
            len(normalized_shape),
        )

        if expected_axes is not None and normalized_axes != expected_axes:
            raise ValueError(f"Expected axes {expected_axes!r}, received {normalized_axes!r}.")
    elif expected_axes is not None:
        raise ValueError("expected_axes was provided, but the record has no axes value.")

    try:
        payload = memoryview(pixel_bytes)
    except TypeError as error:
        raise TypeError(
            "pixel_bytes must support the Python buffer protocol; "
            f"received {type(pixel_bytes).__name__}."
        ) from error

    expected_values = prod(normalized_shape)
    expected_bytes = expected_values * _FLOAT32_LE.itemsize

    if payload.nbytes != expected_bytes:
        raise ValueError(
            f"Pixel payload contains {payload.nbytes} bytes; "
            f"expected {expected_bytes} bytes for shape "
            f"{normalized_shape} and dtype {_FLOAT32_LE}."
        )

    decoded = np.frombuffer(
        payload,
        dtype=_FLOAT32_LE,
        count=expected_values,
    ).reshape(normalized_shape, order="C")

    if copy:
        return decoded.copy(order="C")

    return decoded
