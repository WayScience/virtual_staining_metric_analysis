from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and require a top-level mapping."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, Mapping):
        raise TypeError(f"Config file {config_path} must contain a YAML mapping.")
    return dict(config)


def _require_mapping(
    config: Mapping[str, Any],
    key: str,
    *,
    path: str | None = None,
) -> Mapping[str, Any]:
    value = require_config_value(config, key)
    value_path = path or key
    if not isinstance(value, Mapping):
        raise TypeError(f"Config value {value_path!r} must be a mapping.")
    return value


def _require_string_list(config: Mapping[str, Any], key: str, *, path: str) -> list[str]:
    value = require_config_value(config, key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"Config value {path!r} must be a list of non-empty strings.")
    return value


def _require_string_mapping(
    config: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> Mapping[str, str]:
    value = _require_mapping(config, key, path=path)
    if any(
        not isinstance(item_key, str)
        or not item_key
        or not isinstance(item_value, str)
        or not item_value
        for item_key, item_value in value.items()
    ):
        raise TypeError(f"Config value {path!r} must map non-empty strings to strings.")
    return value


def _require_order_coverage(
    order: list[str],
    settings: Mapping[str, str],
    *,
    path: str,
) -> None:
    missing = [item for item in order if item not in settings]
    if missing:
        raise ValueError(f"Config value {path!r} is missing settings for: {', '.join(missing)}.")


def load_degradation_plot_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate shared metric and degradation-transform plot settings."""
    config = load_yaml_config(config_path)
    metrics = _require_mapping(config, "metrics")
    transforms = _require_mapping(config, "transforms")

    metric_order = _require_string_list(metrics, "order", path="metrics.order")
    metric_labels = _require_string_mapping(metrics, "labels", path="metrics.labels")
    metric_palette = _require_string_mapping(metrics, "palette", path="metrics.palette")
    transform_order = _require_string_list(transforms, "order", path="transforms.order")
    transform_labels = _require_string_mapping(
        transforms,
        "labels",
        path="transforms.labels",
    )

    _require_order_coverage(metric_order, metric_labels, path="metrics.labels")
    _require_order_coverage(metric_order, metric_palette, path="metrics.palette")
    _require_order_coverage(transform_order, transform_labels, path="transforms.labels")
    return config


def require_config_value(config: Mapping[str, Any], key: str) -> Any:
    """Return a required, non-empty config value."""
    value = config.get(key)
    if value is None or isinstance(value, (str, list, tuple, dict, set)) and not value:
        raise ValueError(f"Required config value {key!r} is not set.")
    return value


def require_config_string(config: Mapping[str, Any], key: str) -> str:
    """Return a required, non-empty string config value."""
    value = require_config_value(config, key)
    if not isinstance(value, str):
        raise TypeError(f"Config value {key!r} must be a string.")
    return value


def require_config_directory(
    config: Mapping[str, Any],
    key: str,
    *,
    create: bool = False,
) -> Path:
    """Return a configured directory, optionally creating it."""
    directory = Path(require_config_value(config, key)).expanduser()
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    elif not directory.is_dir():
        raise FileNotFoundError(f"Configured directory {key!r} does not exist: {directory}")
    return directory


def require_config_membership(
    config: Mapping[str, Any],
    key: str,
    collection_key: str,
) -> Any:
    """Return a required value after checking membership in another config value."""
    value = require_config_value(config, key)
    collection = require_config_value(config, collection_key)
    try:
        is_member = value in collection
    except TypeError as error:
        raise TypeError(
            f"Config value {collection_key!r} must support membership checks."
        ) from error
    if not is_member:
        raise ValueError(f"Config value {key!r} ({value!r}) is absent from {collection_key!r}.")
    return value
