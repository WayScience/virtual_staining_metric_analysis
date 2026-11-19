from pathlib import Path


def resolve_root(
    caller_file: str | Path | None = None,
    markers: tuple[str, ...] = (".git", "pyproject.toml", "setup.py"),
) -> Path:
    # Start from the caller's __file__ if provided, otherwise fall back to cwd (for notebooks)
    start_path = Path(caller_file).resolve() if caller_file else Path.cwd().resolve()
    current = start_path if start_path.is_dir() else start_path.parent

    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            return parent

    raise FileNotFoundError(f"Could not find repository root containing any of {markers}")
