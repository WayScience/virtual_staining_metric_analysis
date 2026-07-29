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


def require_config_value(config: Mapping[str, Any], key: str) -> Any:
    """Return a required, non-empty config value."""
    value = config.get(key)
    if value is None or isinstance(value, (str, list, tuple, dict, set)) and not value:
        raise ValueError(f"Required config value {key!r} is not set.")
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
