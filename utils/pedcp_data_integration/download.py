"""Download pinned metadata into a source-versioned cache."""

from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from utils.pedcp_data_integration.artifacts import Artifacts, atomic_write
from utils.pedcp_data_integration.recipe import SOURCE_COMMIT, SOURCE_FILES, SOURCE_REPO


def download_metadata(config) -> list[Path]:
    artifacts = Artifacts(config)
    slug = SOURCE_REPO.removesuffix(".git").removeprefix("https://github.com/")
    paths = []
    for local, source in SOURCE_FILES.items():
        path = artifacts.raw / local
        if not path.is_file() or not path.stat().st_size:
            url = f"https://raw.githubusercontent.com/{slug}/{SOURCE_COMMIT}/{quote(source, safe='/')}"
            with urlopen(url, timeout=60) as response:
                data = response.read()
                length = response.headers.get("Content-Length")
                if not data or (length is not None and len(data) != int(length)):
                    raise OSError(f"Incomplete download: {url}")
                atomic_write(path, lambda temporary, data=data: temporary.write_bytes(data))
        paths.append(path)
    return paths
