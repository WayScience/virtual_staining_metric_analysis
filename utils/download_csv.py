"""
Helper module for accessing and downloading CSV files from a GitHub repository.
"""

from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


def download_csv_files(
    source_paths: list[str],
    destination_dir: Path,
    repo_url: str,
    commit: str,
) -> list[Path]:
    """Download CSV files from a pinned GitHub commit into one flat directory.

    Each downloaded file keeps only its source name (stem and ``.csv``
    extension), so parent directories from the repository are not recreated.
    Existing destination files are replaced.
    """
    github_prefix = "https://github.com/"
    repository_url = repo_url.removesuffix(".git")
    if not repository_url.startswith(github_prefix):
        raise ValueError(f"Expected a GitHub repository URL, got: {repo_url}")

    repository_slug = repository_url.removeprefix(github_prefix).strip("/")
    if repository_slug.count("/") != 1:
        raise ValueError(f"Could not determine owner and repository from: {repo_url}")

    relative_paths = [source_path.lstrip("/") for source_path in source_paths]
    destination_names = [Path(source_path).name for source_path in relative_paths]

    non_csv_paths = [
        source_path
        for source_path in relative_paths
        if Path(source_path).suffix.lower() != ".csv"
    ]
    if non_csv_paths:
        raise ValueError(f"All source files must be CSV files: {non_csv_paths}")

    if len(destination_names) != len(set(destination_names)):
        raise ValueError("Source paths contain duplicate filenames after flattening")

    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []

    for source_path, destination_name in zip(relative_paths, destination_names):
        encoded_source_path = quote(source_path, safe="/")
        raw_url = (
            f"https://raw.githubusercontent.com/{repository_slug}/"
            f"{commit}/{encoded_source_path}"
        )
        destination_path = destination_dir / destination_name

        # Read the complete response before replacing any existing local file.
        with urlopen(raw_url, timeout=60) as response:
            file_contents = response.read()
        destination_path.write_bytes(file_contents)

        downloaded_paths.append(destination_path)
        print(f"Downloaded {source_path} -> {destination_path}")

    return downloaded_paths
