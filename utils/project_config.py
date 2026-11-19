"""Configure `"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    metadata_root: Path
    large_output_root: Path
    image_root: Path | None = None
    outline_root: Path | None = None
    workers: int = 8
    preview_count: int = 3
    preview_seed: int = 42


def _mapping(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    return value


def load_project_config(path: str | Path | None = None) -> ProjectConfig:
    """Load the single YAML, resolving its paths relative to its own parent."""
    config_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else Path(__file__).resolve().parents[1] / "project_config.yaml"
    )
    with config_path.open(encoding="utf-8") as handle:
        data = _mapping(yaml.safe_load(handle), {"paths", "pedcp"}, "config")
    paths = _mapping(data["paths"], {"metadata_root", "large_output_root"}, "paths")
    pedcp = _mapping(
        data["pedcp"],
        {"image_root", "outline_root", "workers", "preview_count", "preview_seed"},
        "pedcp",
    )

    def resolve(value: object, nullable: bool = False) -> Path | None:
        if nullable and value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Paths must be nonempty strings (input roots may be null)")
        result = Path(value).expanduser()
        return (config_path.parent / result).resolve()

    for key, minimum in (("workers", 1), ("preview_count", 0), ("preview_seed", 0)):
        value = pedcp[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"pedcp.{key} must be an integer >= {minimum}")
    if pedcp["preview_seed"] > 2**32 - 1:
        raise ValueError("pedcp.preview_seed must be <= 2**32 - 1")
    metadata_root = resolve(paths["metadata_root"])
    large_output_root = resolve(paths["large_output_root"])
    assert metadata_root is not None and large_output_root is not None
    return ProjectConfig(
        metadata_root=metadata_root,
        large_output_root=large_output_root,
        image_root=resolve(pedcp["image_root"], nullable=True),
        outline_root=resolve(pedcp["outline_root"], nullable=True),
        workers=pedcp["workers"],
        preview_count=pedcp["preview_count"],
        preview_seed=pedcp["preview_seed"],
    )
